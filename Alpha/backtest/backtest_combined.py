"""
Combined Alpha-Risk Model Backtesting Script

Uses Alpha model for direction decisions and Risk model for SL/TP/position sizing.

Usage:
    python Alpha/backtest/backtest_combined.py \
        --alpha-model checkpoints/8.03.zip \
        --risk-model risk_model_final.zip \
        --data-dir Alpha/backtest/data \
        --output-dir Alpha/backtest/results \
        --episodes 1
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

# Add numpy 1.x/2.x compatibility shim for SB3 model loading
if not hasattr(np, "_core"):
    import sys
    sys.modules["numpy._core"] = np.core

import argparse
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
import tempfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Absolute imports from project root
from Alpha.src.trading_env import TradingEnv
from Alpha.backtest.backtest import BacktestMetrics, NumpyEncoder, generate_all_charts
import joblib
import torch
from RiskLayer.src.risk_model_sl import RiskModelSL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

initial_equity=10.0

class CombinedBacktest:
    """Combined backtest using Alpha model for direction and SL Risk model for SL/TP/sizing"""
    
    def __init__(self, alpha_model, risk_model, risk_scaler, data_dir, initial_equity=initial_equity, alpha_norm_env=None, env=None, verify_alpha=False):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.risk_scaler = risk_scaler
        self.alpha_norm_env = alpha_norm_env
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.verify_alpha = verify_alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create Alpha environment for data access (or reuse existing)
        if env is not None:
            self.env = env
        else:
            self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
            
        self.env.equity = initial_equity
        self.env.start_equity = initial_equity
        self.env.peak_equity = initial_equity
        
        # Risk model constants
        self.MAX_ALLOCATION_PCT = 0.60 # New User Request: 60% Max Allocation
        self.MAX_LEVERAGE = 100.0      # Updated to match RiskLayer logic
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000
        
        # Slippage Configuration (Match Training)
        self.ENABLE_SLIPPAGE = True
        self.SLIPPAGE_MIN_PIPS = 0.5
        self.SLIPPAGE_MAX_PIPS = 1.5
        
        # Per-asset history tracking
        self.asset_histories = {
            asset: {
                'pnl_history': deque([0.0] * 5, maxlen=5),
                'action_history': deque([np.zeros(2, dtype=np.float32) for _ in range(5)], maxlen=5)
            }
            for asset in self.env.assets
        }
        
        # Track current equity and peak
        self.equity = initial_equity
        self.peak_equity = initial_equity
        
    def build_risk_observation(self, asset, alpha_obs):
        """Build 60-feature observation for SL risk model"""
        # 1. Market State (40)
        market_obs = alpha_obs
        
        # 2. Account state (5 features)
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        equity_norm = self.equity / self.initial_equity
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        # Win streak normalization (track last 5 trades)
        recent_pnls = list(self.asset_histories[asset]['pnl_history'])[-5:]
        wins = sum(1 for p in recent_pnls if p > 0)
        win_streak_norm = wins / 5.0
        
        account_obs = np.array([
            equity_norm,
            drawdown,
            0.0,  # Leverage placeholder
            risk_cap_mult,
            win_streak_norm
        ], dtype=np.float32)
        
        # 3. History features (5 PnL + 10 Actions = 15)
        hist = self.asset_histories[asset]
        hist_pnl = np.array(list(hist['pnl_history']), dtype=np.float32)
        hist_acts = np.array(list(hist['action_history']), dtype=np.float32).flatten()
        
        # Combine: 40 + 5 + 5 + 10 = 60
        obs = np.concatenate([market_obs, account_obs, hist_pnl, hist_acts])
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 4. Scale
        obs_scaled = self.risk_scaler.transform(obs.reshape(1, -1)).astype(np.float32)
        return torch.from_numpy(obs_scaled).to(self.device)
    
    def predict_risk(self, obs_tensor):
        """Perform inference with SL risk model"""
        self.risk_model.eval()
        with torch.no_grad():
            preds = self.risk_model(obs_tensor)
            sl_mult = preds['sl'].item()
            tp_mult = preds['tp'].item()
            size_out = preds['size'].item()
        return sl_mult, tp_mult, size_out
    
    def calculate_position_size(self, asset, entry_price, size_out):
        """Calculate position size using Direct Model Allocation (No Cap)"""
        # Final_Size = Equity * size from model (0.0 to 1.0)
        position_size = self.equity * size_out
        position_value_usd = position_size * self.MAX_LEVERAGE
        
        # Calculate Lots
        is_usd_quote = asset in ['EURUSD', 'GBPUSD', 'XAUUSD']
        if is_usd_quote:
            lot_value_usd = self.CONTRACT_SIZE * entry_price
        else:
            lot_value_usd = self.CONTRACT_SIZE
            
        lots = position_value_usd / (lot_value_usd + 1e-9)
        lots = np.clip(lots, self.MIN_LOTS, 100.0)
        
        return size_out, lots, position_size
    
    def run_backtest(self, episodes=1, max_steps=5000):
        """Run combined backtest"""
        metrics_tracker = BacktestMetrics()
        logger.info(f"Running {episodes} episodes with max {max_steps} steps...")
        
        for episode in range(episodes):
            obs_dummy, _ = self.env.reset()
            done = False
            step_count = 0
            
            self.equity = self.initial_equity
            self.peak_equity = self.initial_equity
            self.env.equity = self.initial_equity
            self.env.peak_equity = self.initial_equity
            
            alpha_actions_sum = 0
            alpha_non_zero = 0
            
            if self.verify_alpha:
                logger.info("VERIFY ALPHA MODE: Risk Model Bypassed. Using fixed SL=2.0, TP=4.0, Size=25%")
            
            while not done:
                combined_actions = {}
                
                # For each asset, get Alpha direction
                for asset in self.env.assets:
                    self.env.set_asset(asset)
                    alpha_obs = self.env._get_observation()
                    
                    # Normalize alpha_obs if needed
                    pred_obs = alpha_obs
                    if self.alpha_norm_env is not None:
                        pred_obs = self.alpha_norm_env.normalize_obs(alpha_obs.reshape(1, -1)).flatten()
                        
                    alpha_action, _ = self.alpha_model.predict(pred_obs, deterministic=True)
                    
                    # Debug stats
                    alpha_actions_sum += abs(alpha_action[0])
                    if abs(alpha_action[0]) > 0.33:
                        alpha_non_zero += 1
                    
                    # Single-pair action to direction
                    # -1 to 1 continuous to discrete
                    direction = 1 if alpha_action[0] > 0.33 else (-1 if alpha_action[0] < -0.33 else 0)
                    
                    if direction == 0:
                        continue
                        
                    if self.verify_alpha:
                        # Mode: Verify Alpha only (Fixed Risk & SL/TP used in Alpha training)
                        sl_mult = 2.0
                        tp_mult = 4.0
                        size_pct = 0.25 # 25% fixed
                    else:
                        # Build and predict with SL Risk model
                        risk_obs_tensor = self.build_risk_observation(asset, alpha_obs)
                        sl_mult, tp_mult, size_out = self.predict_risk(risk_obs_tensor)
                        
                        # Calculate size (60% Max Allocation Rule)
                        current_prices = self.env._get_current_prices()
                        entry_price = current_prices[asset]
                        
                        size_pct, lots, pos_size = self.calculate_position_size(
                            asset, entry_price, size_out
                        )
                        
                        # Update action history
                        self.asset_histories[asset]['action_history'].append(np.array([sl_mult, tp_mult], dtype=np.float32))
                    
                    if size_pct > 0.01: # Filter out very small sizes
                        combined_actions[asset] = {
                            'direction': direction,
                            'size': size_pct, # This is the fraction used for rewards calculation in TradingEnv
                            'sl_mult': sl_mult,
                            'tp_mult': tp_mult,
                            'lots': lots if not self.verify_alpha else 0.1 # Placeholder if verify
                        }
                
                # Step 2: Execute all approved trades
                self.env.completed_trades = []
                current_prices = self.env._get_current_prices()
                atrs = self.env._get_current_atrs()
                
                for asset in self.env.assets:
                    act = combined_actions.get(asset)
                    current_pos = self.env.positions[asset]
                    price_raw = current_prices[asset]
                    atr = atrs[asset]
                    
                    if act:
                        direction = act['direction']
                        pip_scalar = 0.01 if 'JPY' in asset or 'XAU' in asset else 0.0001
                        
                        # Apply Slippage (Match Training 0.5 - 1.5 pips)
                        slippage_pips = np.random.uniform(0.5, 1.5)
                        price = price_raw + (direction * -1 * slippage_pips * pip_scalar)
                            
                        if current_pos is None:
                            self.env._open_position(asset, direction, act, price, atr)
                        elif current_pos['direction'] != direction:
                            self.env._close_position(asset, price)
                            self.env._open_position(asset, direction, act, price, atr)
                    else:
                        # No action (Hold): Keep existing position if any (user requested)
                        continue
                
                # Advance time and update
                self.env.current_step += 1
                self.env._update_positions()
                
                # Update peak equity and history
                self.env.peak_equity = max(self.env.peak_equity, self.env.equity)
                prev_equity = self.equity
                self.equity = self.env.equity
                self.peak_equity = self.env.peak_equity
                
                # Update per-asset PnL history
                if self.env.completed_trades:
                    for trade in self.env.completed_trades:
                        asset = trade['asset']
                        pnl_ratio = trade['pnl'] / max(prev_equity, 1e-6)
                        self.asset_histories[asset]['pnl_history'].append(pnl_ratio)
                
                info = {
                    'trades': self.env.completed_trades,
                    'equity': self.env.equity,
                    'timestamp': self.env._get_current_timestamp()
                }
                
                # Log metrics
                if info['trades']:
                    for trade in info['trades']:
                        metrics_tracker.add_trade(trade)
                metrics_tracker.add_equity_point(info['timestamp'], info['equity'])
                
                step_count += 1
                done = self.env.current_step >= self.env.max_steps
                if max_steps is not None and step_count >= max_steps:
                    done = True
                
                if step_count % 1000 == 0:
                    avg_alpha = alpha_actions_sum / (step_count * len(self.env.assets) + 1e-9)
                    logger.info(f"Step {step_count}, Equity: ${self.equity:.2f}, Avg Alpha Action: {avg_alpha:.4f}, Trades Triggered: {alpha_non_zero}")
            
            avg_alpha = alpha_actions_sum / (step_count * len(self.env.assets) + 1e-9)
            logger.info(f"Episode {episode + 1} complete. Final Equity: ${self.env.equity:.2f}, Avg Alpha: {avg_alpha:.4f}, Non-Zero Actions: {alpha_non_zero}")
        
        return metrics_tracker


def run_combined_backtest(args):
    """Main backtesting function"""
    project_root = Path(__file__).resolve().parent.parent.parent
    alpha_model_path = project_root / args.alpha_model
    risk_model_path = project_root / args.risk_model
    data_dir_path = project_root / args.data_dir
    output_dir_path = project_root / args.output_dir

    logger.info("Starting combined Alpha-Risk backtest")
    logger.info(f"Alpha model: {alpha_model_path}")
    logger.info(f"Risk model: {risk_model_path}")
    logger.info(f"Data directory: {data_dir_path}")
    
    # Create output directory
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Validate model files
    if not alpha_model_path.exists():
        logger.error(f"Alpha model file not found: {alpha_model_path}")
        sys.exit(1)
    
    if not risk_model_path.exists():
        logger.error(f"Risk model file not found: {risk_model_path}")
        sys.exit(1)
    
    # Create a SINGLE TradingEnv instance to be shared
    logger.info("Initializing shared Trading Environment and preprocessing data...")
    shared_env = TradingEnv(data_dir=data_dir_path, stage=1, is_training=False)
    dummy_vec_env = DummyVecEnv([lambda: shared_env])

    # Load Alpha normalizer
    alpha_norm_path = str(alpha_model_path).replace('.zip', '_vecnormalize.pkl')
    # FALLBACK: Handle ppo_final_model(1).zip -> ppo_final_model(1)_vecnormalize.pkl (already handled by above)
    # or handle cases where (1) might be missing in some naming conventions
    if not os.path.exists(alpha_norm_path):
        # Specific check for the user's special directory naming
        if "modej" in str(alpha_model_path):
             alpha_norm_path = alpha_model_path.parent / "ppo_final_vecnormalize(1).pkl"
        
        if not os.path.exists(alpha_norm_path):
             alpha_norm_path = str(alpha_model_path).replace('_model.zip', '_vecnormalize.pkl')
        
    alpha_norm_env = None
    if os.path.exists(alpha_norm_path):
        logger.info(f"Loading Alpha Normalizer from {alpha_norm_path}")
        # Reuse the already preprocessed dummy_vec_env
        alpha_norm_env = VecNormalize.load(alpha_norm_path, dummy_vec_env)
        alpha_norm_env.training = False
        alpha_norm_env.norm_reward = False
    else:
        logger.warning(f"Alpha Normalizer NOT found at {alpha_norm_path}. Model predictions may be inaccurate.")
    
    # Load Alpha model
    logger.info("Loading Alpha model...")
    alpha_model = PPO.load(alpha_model_path, env=dummy_vec_env)
    
    # Load SL Risk model and Scaler
    logger.info("Loading SL Risk model and Scaler...")
    try:
        # 1. Load Scaler
        scaler_path = risk_model_path.parent / "sl_risk_scaler.pkl"
        if not scaler_path.exists():
            scaler_path = project_root / "models" / "sl_risk_scaler.pkl"
        risk_scaler = joblib.load(scaler_path)
        
        # 2. Load Model
        risk_model = RiskModelSL(input_dim=60)
        # Handle weights file (best.pth)
        if str(risk_model_path).endswith('.pth'):
            weights_path = risk_model_path
        else:
            weights_path = risk_model_path.parent / "risk_model_sl_best.pth"
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(weights_path, map_location=device)
        risk_model.load_state_dict(state_dict)
        risk_model.to(device)
        risk_model.eval()
        
        logger.info(f"SL Risk model loaded from {weights_path}")
    except Exception as e:
        logger.error(f"Failed to load SL Risk model: {e}")
        sys.exit(1)
    
    # Create combined backtest
    backtest = CombinedBacktest(
        alpha_model, 
        risk_model, 
        risk_scaler,
        data_dir_path, 
        initial_equity=args.initial_equity,
        alpha_norm_env=alpha_norm_env,
        env=shared_env,
        verify_alpha=args.verify_alpha
    )

    # Run backtest
    metrics_tracker = backtest.run_backtest(episodes=args.episodes, max_steps=args.max_steps)



    
    # Calculate metrics
    logger.info("\n" + "="*60)
    logger.info("CALCULATING METRICS")
    logger.info("="*60)
    
    metrics = metrics_tracker.calculate_metrics()
    
    # Print metrics
    logger.info(f"\n{'BACKTEST RESULTS':^60}")
    logger.info("="*60)
    logger.info(f"{'PRIMARY METRIC - Profit Factor:':<40} {metrics.get('profit_factor', 0):.3f}")
    logger.info(f"{'Total Return:':<40} {metrics.get('total_return', 0):.2%}")
    logger.info(f"{'Sharpe Ratio:':<40} {metrics.get('sharpe_ratio', 0):.3f}")
    logger.info(f"{'Max Drawdown:':<40} {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"{'Win Rate:':<40} {metrics.get('win_rate', 0):.2%}")
    logger.info(f"{'Average RR Ratio:':<40} {metrics.get('avg_rr_ratio', 0):.2f}")
    logger.info(f"{'Trade Frequency (per day):':<40} {metrics.get('trade_frequency', 0):.2f}")
    logger.info(f"{'Average Hold Time (minutes):':<40} {metrics.get('avg_hold_time_minutes', 0):.1f}")
    logger.info(f"{'Total Trades:':<40} {metrics.get('total_trades', 0)}")
    logger.info(f"{'Winning Trades:':<40} {metrics.get('winning_trades', 0)}")
    logger.info(f"{'Losing Trades:':<40} {metrics.get('losing_trades', 0)}")
    logger.info("="*60)
    
    # Check PRD success criteria
    logger.info(f"\n{'PRD SUCCESS CRITERIA CHECK':^60}")
    logger.info("="*60)
    pf_pass = metrics.get('profit_factor', 0) > 1.3
    dd_pass = metrics.get('max_drawdown', 0) > -0.20
    sr_pass = metrics.get('sharpe_ratio', 0) > 1.0
    wr_pass = metrics.get('win_rate', 0) > 0.45
    
    logger.info(f"{'Profit Factor > 1.3:':<40} {'✅ PASS' if pf_pass else '❌ FAIL'}")
    logger.info(f"{'Max Drawdown < 20%:':<40} {'✅ PASS' if dd_pass else '❌ FAIL'}")
    logger.info(f"{'Sharpe Ratio > 1.0:':<40} {'✅ PASS' if sr_pass else '❌ FAIL'}")
    logger.info(f"{'Win Rate > 45%:':<40} {'✅ PASS' if wr_pass else '❌ FAIL'}")
    logger.info("="*60)
    
    all_pass = pf_pass and dd_pass and sr_pass and wr_pass
    logger.info(f"\n{'OVERALL: ' + ('✅ ALL CRITERIA MET' if all_pass else '❌ SOME CRITERIA NOT MET'):^60}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_asset = metrics_tracker.get_per_asset_metrics()
    
    # 1. Save metrics JSON
    metrics_file = output_dir_path / f"metrics_combined_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # 2. Save trade log
    if metrics_tracker.trades:
        trades_file = output_dir_path / f"trades_combined_{timestamp}.csv"
        pd.DataFrame(metrics_tracker.trades).to_csv(trades_file, index=False)
        logger.info(f"Saved trade log to {trades_file}")
    
    # 3. Save per-asset performance
    if per_asset:
        asset_file = output_dir_path / f"asset_breakdown_combined_{timestamp}.csv"
        pd.DataFrame(per_asset).T.to_csv(asset_file)
        logger.info(f"Saved per-asset breakdown to {asset_file}")
    
    # 3.5. Save Blocked Trades Analysis
    if hasattr(backtest, 'blocked_trades') and backtest.blocked_trades:
        blocked_file = output_dir_path / f"blocked_trades_combined_{timestamp}.csv"
        df_blocked = pd.DataFrame(backtest.blocked_trades)
        df_blocked.to_csv(blocked_file, index=False)
        logger.info(f"Saved blocked trades log to {blocked_file}")
        
        # Calculate Stats
        total_blocked = len(df_blocked)
        blocked_wins = len(df_blocked[df_blocked['outcome'] == 'WIN'])
        blocked_losses = len(df_blocked[df_blocked['outcome'] == 'LOSS'])
        
        # Avoided Loss (Sum of negative PnL that was blocked)
        # Note: 'theoretical_pnl' is raw value from simulation. Negative means we saved that loss.
        avoided_loss_sum = df_blocked[df_blocked['theoretical_pnl'] < 0]['theoretical_pnl'].sum()
        missed_profit_sum = df_blocked[df_blocked['theoretical_pnl'] > 0]['theoretical_pnl'].sum()
        
        logger.info(f"\n{'BLOCKED TRADES ANALYSIS':^60}")
        logger.info("="*60)
        logger.info(f"{'Total Blocked Trades:':<40} {total_blocked}")
        logger.info(f"{'Blocked LOSSES (Good Blocks):':<40} {blocked_losses} ({blocked_losses/total_blocked:.1%})")
        logger.info(f"{'Blocked WINS (Missed Opport.):':<40} {blocked_wins} ({blocked_wins/total_blocked:.1%})")
        logger.info(f"{'Total Avoided Loss Value (Red Saved):':<40} {abs(avoided_loss_sum):.4f}")
        logger.info(f"{'Total Missed Profit Value (Green Lost):':<40} {missed_profit_sum:.4f}")
        logger.info("="*60)

    
    # 4. Generate all visualizations
    if metrics_tracker.equity_curve and metrics_tracker.trades:
        logger.info("\nGenerating comprehensive charts...")
        generate_all_charts(metrics_tracker, per_asset, "combined", output_dir_path, timestamp)
    
    logger.info("\nBacktest complete!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Alpha-Risk Model Backtest")
    parser.add_argument("--alpha-model", type=str, default="models/checkpoints/ppo_final_model.zip",
                        help="Path to Alpha model (.zip file) relative to project root")
    parser.add_argument("--risk-model", type=str, default="models/risk/risk_model_final.zip",
                        help="Path to Risk model (.zip file) relative to project root")
    parser.add_argument("--data-dir", type=str, default="Alpha/backtest/data",
                        help="Path to backtest data directory relative to project root")
    parser.add_argument("--output-dir", type=str, default="Alpha/backtest/results",
                        help="Path to save results relative to project root")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Maximum steps per episode")
    parser.add_argument("--initial-equity", type=float, default=10.0,
                        help="Starting equity for the backtest")
    parser.add_argument("--verify-alpha", action="store_true",
                        help="Bypass Risk model and use Alpha training defaults (Fixed SL/TP/Size)")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir_path = project_root / args.data_dir

    # Validate data directory
    if not data_dir_path.exists():
        logger.error(f"Data directory not found: {data_dir_path}")
        sys.exit(1)
    
    parquet_files = list(data_dir_path.glob('*.parquet'))
    if not parquet_files:
        logger.error(f"No .parquet files found in {data_dir_path}")
        sys.exit(1)
    
    run_combined_backtest(args)


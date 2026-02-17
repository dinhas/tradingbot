"""
Combined Alpha-Risk Model Backtesting Script - OPTIMIZED VERSION

Uses Vectorized Batch Inference and Pre-calculated features for maximum speed.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent)
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
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import tempfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Absolute imports from project root
from Alpha.src.trading_env import TradingEnv
from backtest.backtest import BacktestMetrics, NumpyEncoder, generate_all_charts
import joblib
import torch
from RiskLayer.src.risk_model_sl import RiskModelSL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

initial_equity=10.0

class CombinedBacktest:
    """Combined backtest using Alpha model for direction and SL Risk model for SL/TP/sizing"""
    
    def __init__(self, alpha_model, risk_model, risk_scaler, data_dir, initial_equity=initial_equity, alpha_norm_env=None, env=None, verify_alpha=False, challenge_mode=False):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.risk_scaler = risk_scaler
        self.alpha_norm_env = alpha_norm_env
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.verify_alpha = verify_alpha
        self.challenge_mode = challenge_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Challenge Mode Tracking
        self.daily_high_water_mark = initial_equity
        self.daily_trades_count = 0
        self.is_halted_until_next_day = False
        self.current_day = None
        self.disqualified = False
        self.disqualification_reason = ""
        self.challenge_start_time = None
        self.challenge_end_time = None
        
        # Create Alpha environment for data access (or reuse existing)
        if env is not None:
            self.env = env
        else:
            self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
            
        self.env.equity = initial_equity
        self.env.start_equity = initial_equity
        self.env.peak_equity = initial_equity
        
        # Risk model constants
        self.MAX_ALLOCATION_PCT = 0.60 
        self.MAX_LEVERAGE = 100.0      
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000
        
        # Slippage Configuration
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
        
    def calculate_position_size(self, asset, entry_price, size_out):
        """Calculate position size using Direct Model Allocation"""
        # Challenge Mode Leverage
        leverage = self.MAX_LEVERAGE
        if self.challenge_mode:
            is_forex = asset in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
            leverage = 100.0 if is_forex else 30.0
            
        position_size = self.equity * size_out
        position_value_usd = position_size * leverage
        
        # Calculate Lots
        is_usd_quote = asset in ['EURUSD', 'GBPUSD', 'XAUUSD']
        if is_usd_quote:
            lot_value_usd = self.CONTRACT_SIZE * entry_price if asset != 'XAUUSD' else 100 * entry_price
        else:
            lot_value_usd = self.CONTRACT_SIZE
            
        lots = position_value_usd / (lot_value_usd + 1e-9)
        
        # Challenge Mode Lot Limits
        if self.challenge_mode:
            is_forex = asset in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
            max_lots = 5.0 if is_forex else 3.0
            lots = np.clip(lots, self.MIN_LOTS, max_lots)
        else:
            lots = np.clip(lots, self.MIN_LOTS, 100.0)
        
        return size_out, lots, position_size
    
    def _precalculate_signals(self):
        """Pre-calculate all model outputs in large batches for the entire dataset."""
        logger.info("Pre-calculating Alpha and Risk signals for all assets and steps...")
        
        master_obs = self.env.master_obs_matrix # (N, num_assets * 40)
        N, total_dims = master_obs.shape
        num_assets = len(self.env.assets)
        
        # Reshape to (N * num_assets, 40)
        obs_flat = master_obs.reshape(-1, 40)
        
        # 1. Alpha Batch Inference
        logger.info(f"Running Alpha inference on {len(obs_flat)} observations...")
        if self.alpha_norm_env is not None:
            # VecNormalize works on batches
            obs_norm = self.alpha_norm_env.normalize_obs(obs_flat)
        else:
            obs_norm = obs_flat
            
        alpha_actions_all = []
        batch_size = 16384
        for i in tqdm(range(0, len(obs_norm), batch_size), desc="Alpha Batch"):
            batch = obs_norm[i : i + batch_size]
            actions, _ = self.alpha_model.predict(batch, deterministic=True)
            alpha_actions_all.append(actions)
        
        self.alpha_actions_matrix = np.concatenate(alpha_actions_all, axis=0).reshape(N, num_assets)
        
        # 2. Risk Batch Inference
        if not self.verify_alpha:
            logger.info(f"Running Risk inference on {len(obs_flat)} observations...")
            risk_obs_scaled = self.risk_scaler.transform(obs_flat).astype(np.float32)
            
            sl_list, tp_list, size_list = [], [], []
            for i in tqdm(range(0, len(risk_obs_scaled), batch_size), desc="Risk Batch"):
                batch_tensor = torch.from_numpy(risk_obs_scaled[i : i + batch_size]).to(self.device)
                with torch.no_grad():
                    preds = self.risk_model(batch_tensor)
                    sl_list.append(preds['sl'].cpu().numpy())
                    tp_list.append(preds['tp'].cpu().numpy())
                    size_list.append(preds['size'].cpu().numpy())
            
            self.sl_matrix = np.concatenate(sl_list, axis=0).reshape(N, num_assets)
            self.tp_matrix = np.concatenate(tp_list, axis=0).reshape(N, num_assets)
            self.size_matrix = np.concatenate(size_list, axis=0).reshape(N, num_assets)
        
        logger.info("Signal pre-calculation complete.")

    def run_backtest(self, episodes=1, max_steps=None):
        """Optimized Backtest using PRE-CALCULATED signals"""
        # Pre-calculate everything first
        self._precalculate_signals()
        
        metrics_tracker = BacktestMetrics()
        
        # Pre-cache assets and price arrays
        assets = self.env.assets
        num_assets = len(assets)
        close_prices = {a: self.env.close_arrays[a] for a in assets}
        atr_values = {a: self.env.atr_arrays[a] for a in assets}
        
        logger.info(f"Running {episodes} episodes with max {max_steps or self.env.max_steps} steps...")

        for episode in range(episodes):
            self.env.reset()
            start_step = self.env.current_step
            end_step = min(start_step + (max_steps or self.env.max_steps), self.env.max_steps)
            
            alpha_non_zero = 0
            alpha_actions_sum = 0
            
            self.equity = self.initial_equity
            self.peak_equity = self.initial_equity
            self.env.equity = self.initial_equity
            
            if self.verify_alpha:
                logger.info("VERIFY ALPHA MODE: Risk Model Bypassed.")

            for current_idx in tqdm(range(start_step, end_step), desc=f"Ep {episode+1}"):
                self.env.current_step = current_idx
                current_time = self.env._get_current_timestamp()
                
                # --- CHALLENGE MODE DAY MANAGEMENT ---
                if self.challenge_mode:
                    if self.challenge_start_time is None:
                        self.challenge_start_time = current_time
                        self.challenge_end_time = self.challenge_start_time + timedelta(days=30)
                        logger.info(f"Challenge Started: {self.challenge_start_time}. Ends: {self.challenge_end_time}")

                    if current_time >= self.challenge_end_time:
                        logger.info(f"Challenge Period Ended (30 days reached). Final Time: {current_time}")
                        break

                    day_str = current_time.strftime("%Y-%m-%d")
                    if day_str != self.current_day:
                        # New Day Reset
                        if self.current_day is not None:
                            logger.info(f"New Day: {day_str}. Daily trades: {self.daily_trades_count}. Halted: {self.is_halted_until_next_day}")
                        
                        self.current_day = day_str
                        self.daily_trades_count = 0
                        self.is_halted_until_next_day = False
                        # High Water Mark is max(Balance, Equity) at start of day
                        # Since balance = equity in this simplified model (unless open trades exist)
                        # We use the current equity which includes unrealized P&L
                        self.daily_high_water_mark = self.equity
                    
                                    # 1. Overall Loss Check (10% of initial balance)
                                    if self.equity < (self.initial_equity * 0.90):
                                        self.disqualified = True
                                        self.disqualification_reason = f"Max Overall Loss Breached: Equity ${self.equity:.2f} < ${self.initial_equity * 0.90:.2f}"
                                        logger.error(self.disqualification_reason)
                                        break
                                    
                                    # 2. Daily Loss Check (5% of initial balance)
                                    daily_loss_amount = self.daily_high_water_mark - self.equity
                                    max_daily_loss = self.initial_equity * 0.05
                                    
                                    if daily_loss_amount >= max_daily_loss:
                                        self.disqualified = True
                                        self.disqualification_reason = f"Daily Loss Limit Breached: Loss ${daily_loss_amount:.2f} >= ${max_daily_loss:.2f}"
                                        logger.error(self.disqualification_reason)
                                        break
                                    
                                    # 3. Daily Drawdown Halt (4.5% of initial balance)
                                    if daily_loss_amount >= (self.initial_equity * 0.045):
                                        if not self.is_halted_until_next_day:
                                            logger.warning(f"4.5% Daily Drawdown reached. Halting trading until tomorrow. Time: {current_time}")
                                            self.is_halted_until_next_day = True
                                    
                                    # --- FAST SIGNAL LOOKUP ---
                                    combined_actions = {}
                                    
                                    # Check if we can open new positions
                                    can_trade = True
                                    if self.challenge_mode:
                                        if self.is_halted_until_next_day or self.daily_trades_count >= 50:
                                            can_trade = False
                                    
                                    # Pre-calculate current open positions count
                                    open_pos_count = sum(1 for p in self.env.positions.values() if p is not None)
                    
                                    for i, asset in enumerate(assets):
                                        if not can_trade: break
                    
                                        alpha_action = self.alpha_actions_matrix[current_idx, i]
                                        alpha_actions_sum += abs(alpha_action)
                                        
                                        # Direction check
                                        direction = 1 if alpha_action > 0.33 else (-1 if alpha_action < -0.33 else 0)
                                        if direction == 0:
                                            continue
                                        
                                        # Challenge Mode: Max 5 positions
                                        if self.challenge_mode:
                                            current_pos = self.env.positions.get(asset)
                                            if current_pos is None:
                                                if open_pos_count >= 5: continue
                                            elif current_pos['direction'] == direction:
                                                continue 
                                            
                                        alpha_non_zero += 1
                                        
                                        if self.verify_alpha:
                                            sl_mult, tp_mult, size_pct, lots = 2.0, 4.0, 0.25, 0.1
                                        else:
                                            sl_mult = self.sl_matrix[current_idx, i]
                                            tp_mult = self.tp_matrix[current_idx, i]
                                            size_out = self.size_matrix[current_idx, i]
                                            
                                            if size_out < 0.30: continue
                                                
                                            entry_price = close_prices[asset][current_idx]
                                            size_pct, lots, pos_size = self.calculate_position_size(asset, entry_price, size_out)
                                        
                                        if size_pct > 0.01:
                                            combined_actions[asset] = {
                                                'direction': direction,
                                                'size': size_pct,
                                                'sl_mult': sl_mult,
                                                'tp_mult': tp_mult,
                                                'lots': lots
                                            }
                                            
                                            if self.challenge_mode:
                                                current_pos = self.env.positions.get(asset)
                                                if current_pos is None or current_pos['direction'] != direction:
                                                    self.daily_trades_count += 1
                                                    if current_pos is None: open_pos_count += 1
                                    
                                    # Step 2: Execute Trades (at Close of Candle T)
                                    self.env.completed_trades = []
                                    for asset, act in combined_actions.items():
                                        current_pos = self.env.positions[asset]
                                        price_raw = close_prices[asset][current_idx]
                                        atr = atr_values[asset][current_idx]
                                        
                                        pip_scalar = 0.01 if 'JPY' in asset or 'XAU' in asset else 0.0001
                                        slippage = np.random.uniform(0.5, 1.5) * pip_scalar
                                        price = price_raw + (act['direction'] * -1 * slippage)
                                            
                                        if current_pos is None:
                                            self.env._open_position(asset, act['direction'], act, price, atr)
                                        elif current_pos['direction'] != act['direction']:
                                            self.env._close_position(asset, price)
                                            self.env._open_position(asset, act['direction'], act, price, atr)
                                    
                                    # Step 3: Advance to Candle T+1 and Update Positions
                                    self.env.current_step += 1
                                    if self.env.current_step >= self.env.max_steps:
                                        break
                                    self.env._update_positions()
                                    
                                    self.equity = self.env.equity
                                    self.peak_equity = max(self.peak_equity, self.equity)
                                    
                                    if self.env.completed_trades:
                                        for trade in self.env.completed_trades:
                                            metrics_tracker.add_trade(trade)
                                    
                                    metrics_tracker.add_equity_point(self.env._get_current_timestamp(), self.equity)
                                    
                                    if current_idx % 10000 == 0:
                                        logger.info(f"Step {current_idx}, Equity: ${self.equity:.2f}")
                                logger.info(f"Episode {episode + 1} complete. Final Equity: ${self.equity:.2f}")
            if self.challenge_mode:
                if self.disqualified:
                    logger.error(f"CHALLENGE FAILED: DISQUALIFIED. Reason: {self.disqualification_reason}")
                else:
                    profit_pct = (self.equity - self.initial_equity) / self.initial_equity
                    # Standard prop firm target is usually 10%
                    if profit_pct >= 0.10:
                        logger.info(f"CHALLENGE PASSED! Profit: {profit_pct:.2%}")
                    else:
                        logger.warning(f"CHALLENGE ENDED: Profit Target Not Met. Profit: {profit_pct:.2%}")
        
        return metrics_tracker


def run_combined_backtest(args):
    """Main backtesting function"""
    project_root = Path(__file__).resolve().parent.parent
    alpha_model_path = project_root / args.alpha_model
    risk_model_path = project_root / args.risk_model
    data_dir_path = project_root / args.data_dir
    output_dir_path = project_root / args.output_dir

    logger.info("Starting optimized Alpha-Risk backtest")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    if not alpha_model_path.exists() or not risk_model_path.exists():
        logger.error("Model files not found")
        sys.exit(1)
    
    logger.info("Initializing environment...")
    shared_env = TradingEnv(data_dir=data_dir_path, stage=1, is_training=False)
    dummy_vec_env = DummyVecEnv([lambda: shared_env])

    # Load Normalizer
    alpha_norm_path = str(alpha_model_path).replace('.zip', '_vecnormalize.pkl')
    if not os.path.exists(alpha_norm_path):
        alpha_norm_path = str(alpha_model_path).replace('_model.zip', '_vecnormalize.pkl')
        
    alpha_norm_env = None
    if os.path.exists(alpha_norm_path):
        logger.info(f"Loading Normalizer from {alpha_norm_path}")
        alpha_norm_env = VecNormalize.load(alpha_norm_path, dummy_vec_env)
        alpha_norm_env.training = False
        alpha_norm_env.norm_reward = False
    
    # Load Models
    alpha_model = PPO.load(alpha_model_path, env=dummy_vec_env)
    scaler_path = risk_model_path.parent / "sl_risk_scaler.pkl"
    risk_scaler = joblib.load(scaler_path)
    
    risk_model = RiskModelSL(input_dim=40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(risk_model_path, map_location=device)
    risk_model.load_state_dict(state_dict)
    risk_model.to(device)
    risk_model.eval()
    
    backtest = CombinedBacktest(
        alpha_model, risk_model, risk_scaler, data_dir_path, 
        initial_equity=args.initial_equity, alpha_norm_env=alpha_norm_env,
        env=shared_env, verify_alpha=args.verify_alpha,
        challenge_mode=args.challenge_mode
    )

    metrics_tracker = backtest.run_backtest(episodes=args.episodes, max_steps=args.max_steps)
    
    # Calculate and Print Metrics
    metrics = metrics_tracker.calculate_metrics()
    logger.info(f"\n{'BACKTEST RESULTS':^60}")
    logger.info("="*60)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k:<40} {v:.4f}")
        else:
            logger.info(f"{k:<40} {v}")
    logger.info("="*60)
    
    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = output_dir_path / f"metrics_combined_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    if metrics_tracker.trades:
        trades_file = output_dir_path / f"trades_combined_{timestamp}.csv"
        pd.DataFrame(metrics_tracker.trades).to_csv(trades_file, index=False)
        
        per_asset = metrics_tracker.get_per_asset_metrics()
        generate_all_charts(metrics_tracker, per_asset, "combined", output_dir_path, timestamp)
    
    logger.info("Backtest complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-model", type=str, default="models/checkpoints/ppo_final_model.zip")
    parser.add_argument("--risk-model", type=str, default="models/risk/risk_model_sl_final.pth")
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--output-dir", type=str, default="backtest/results")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--initial-equity", type=float, default=10.0)
    parser.add_argument("--verify-alpha", action="store_true")
    parser.add_argument("--challenge-mode", action="store_true", help="Enable prop-firm challenge risk rules")
    run_combined_backtest(parser.parse_args())

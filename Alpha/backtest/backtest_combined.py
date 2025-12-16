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
import argparse
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import deque
import tempfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Alpha.src.trading_env import TradingEnv
from Alpha.backtest.backtest import BacktestMetrics, NumpyEncoder, generate_all_charts

# Import risk model components
try:
    from RiskLayer.src.risk_env import RiskManagementEnv
except ImportError:
    # Try alternative path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../RiskLayer/src')))
    from risk_env import RiskManagementEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

initial_equity=50.0

class CombinedBacktest:
    """Combined backtest using Alpha model for direction and Risk model for SL/TP/sizing"""
    
    def __init__(self, alpha_model, risk_model, data_dir, initial_equity=initial_equity):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        
        # Create Alpha environment for data access
        self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
        self.env.equity = initial_equity
        self.env.start_equity = initial_equity
        self.env.peak_equity = initial_equity
        
        # Risk model constants (from RiskManagementEnv)
        self.MAX_RISK_PER_TRADE = 0.40  # 40% max risk per trade
        self.MAX_MARGIN_PER_TRADE_PCT = 0.80
        self.MAX_LEVERAGE = 400.0
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000
        
        # Per-asset history tracking
        self.asset_histories = {
            asset: {
                'pnl_history': deque([0.0] * 5, maxlen=5),
                'action_history': deque([np.zeros(3, dtype=np.float32) for _ in range(5)], maxlen=5)
            }
            for asset in self.env.assets
        }
        
        # Track current equity and peak
        self.equity = initial_equity
        self.peak_equity = initial_equity
        
    def build_risk_observation(self, asset):
        """Build 165-feature observation for risk model"""
        # 1. Market state (140 features) - same as Alpha observation
        alpha_obs = self.env._get_observation()
        market_obs = alpha_obs[:140]  # First 140 features are market state
        
        # 2. Account state (5 features)
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        equity_norm = self.equity / self.initial_equity
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        account_obs = np.array([
            equity_norm,
            drawdown,
            0.0,  # Leverage placeholder
            risk_cap_mult,
            0.0   # Padding
        ], dtype=np.float32)
        
        # 3. History (20 features)
        asset_history = self.asset_histories[asset]
        hist_pnl = np.array(asset_history['pnl_history'], dtype=np.float32)
        hist_acts = np.array(asset_history['action_history'], dtype=np.float32).flatten()
        
        # Combine all features
        obs = np.concatenate([market_obs, account_obs, hist_pnl, hist_acts])
        
        # Safety check
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    
    def parse_risk_action(self, action):
        """Parse risk model action to SL/TP/sizing"""
        # Parse action (3 values in [-1, 1])
        sl_mult = np.clip((action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)  # 0.2-2.0 ATR
        tp_mult = np.clip((action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)  # 0.5-4.0 ATR
        risk_raw = np.clip((action[2] + 1) / 2, 0.0, 1.0)  # 0-100% of max risk
        
        return sl_mult, tp_mult, risk_raw
    
    def calculate_position_size(self, asset, entry_price, atr, sl_mult, tp_mult, risk_raw, direction):
        """Calculate position size from risk percentage"""
        # Check if trade should be skipped
        if risk_raw < 1e-3:
            return 0.0, None
        
        # Calculate drawdown-based risk cap
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        # Actual risk percentage
        actual_risk_pct = risk_raw * self.MAX_RISK_PER_TRADE * risk_cap_mult
        
        # Calculate SL distance
        sl_dist_price = sl_mult * atr
        min_sl_dist = max(0.0001 * entry_price, 0.2 * atr)
        if sl_dist_price < min_sl_dist:
            sl_dist_price = min_sl_dist
        
        # Determine correct contract size
        # Standard Forex Lot = 100,000 units
        # Standard Gold Lot (XAUUSD) = 100 oz
        contract_size = 100 if asset == 'XAUUSD' else 100000
        
        # Determine currency pair type
        is_usd_quote = asset in ['EURUSD', 'GBPUSD', 'XAUUSD']
        is_usd_base = asset in ['USDJPY', 'USDCHF']
        
        # Calculate risk amount in cash
        risk_amount_cash = self.equity * actual_risk_pct
        
        # Calculate lots based on risk
        lots = 0.0
        if sl_dist_price > 0:
            if is_usd_quote:
                lots = risk_amount_cash / (sl_dist_price * contract_size)
            elif is_usd_base:
                lots = (risk_amount_cash * entry_price) / (sl_dist_price * contract_size)
            else:
                lots = risk_amount_cash / (sl_dist_price * contract_size)
        
        # Apply leverage constraints
        # Use appropriate contract size for value calculation
        if is_usd_quote:
            lot_value_usd = contract_size * entry_price
        elif is_usd_base:
            lot_value_usd = contract_size * 1.0
        else:
            lot_value_usd = contract_size * 1.0
        
        max_position_value = (self.equity * self.MAX_MARGIN_PER_TRADE_PCT) * self.MAX_LEVERAGE
        max_lots_leverage = max_position_value / lot_value_usd
        lots = min(lots, max_lots_leverage)
        
        # Check minimum lots
        if lots < self.MIN_LOTS:
            # User Feedback: "Everything under 1:400 leverage is tradeable." 
            # If the calculated risk-based lots are too small, check if we can afford MIN_LOTS with leverage.
            # 1. Calculate margin required for MIN_LOTS
            min_lot_value_usd = 0.0
            if is_usd_quote:
                min_lot_value_usd = self.MIN_LOTS * contract_size * entry_price
            else:
                 min_lot_value_usd = self.MIN_LOTS * contract_size # Simplified for base/cross pairs
            
            margin_required_min = min_lot_value_usd / self.MAX_LEVERAGE
            
            # 2. Check if we have enough equity for margin (plus a buffer)
            if self.equity > (margin_required_min * 1.05): # Lowered buffer to 5% for tight $10 account
                # We have leverage! Force MIN_LOTS.
                lots = self.MIN_LOTS
                if self.env.current_step % 1000 == 0:
                     logger.info(f"DEBUG: [{asset}] Enforcing MIN_LOTS {self.MIN_LOTS}. Margin: ${margin_required_min:.2f} < Equity ${self.equity:.2f}")
            else:
                if self.env.current_step % 1000 == 0:  
                     logger.info(f"DEBUG: [{asset}] Trade skipped. Calc lots {lots:.6f} < MIN_LOTS. Margin for min: ${margin_required_min:.2f} > Equity ${self.equity:.2f}")
                return 0.0, None
        
        lots = np.clip(lots, self.MIN_LOTS, 100.0)
        
        # Convert lots to position size for Alpha environment
        # Alpha environment expects size_pct in range [0, 1]
        
        # Calculate position value in USD (notional value before leverage)
        if is_usd_quote:
            notional_value = lots * contract_size * entry_price
        elif is_usd_base:
            notional_value = lots * contract_size  # 1 lot = 100,000 base currency (or 100 oz gold)
        else:
            notional_value = lots * contract_size
        
        # Position size in equity units (margin required)
        margin_required = notional_value / self.MAX_LEVERAGE
        
        # Convert to Alpha's size_pct format
        MAX_POS_SIZE_PCT = 0.50  # From Alpha environment
        size_pct = margin_required / (MAX_POS_SIZE_PCT * self.equity)
        size_pct = np.clip(size_pct, 0.0, 1.0)  # Ensure in valid range
        
        position_size = size_pct * MAX_POS_SIZE_PCT * self.equity
        
        return size_pct, {
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'risk_raw': risk_raw,
            'lots': lots,
            'position_size': position_size
        }
    
    def run_backtest(self, episodes=1):
        """Run combined backtest"""
        metrics_tracker = BacktestMetrics()
        
        logger.info(f"Running {episodes} episodes...")
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            done = False
            step_count = 0
            
            # Reset state
            self.equity = self.initial_equity
            self.peak_equity = self.initial_equity
            self.env.equity = self.initial_equity
            self.env.peak_equity = self.initial_equity
            
            # Reset histories
            for asset in self.env.assets:
                self.asset_histories[asset]['pnl_history'] = deque([0.0] * 5, maxlen=5)
                self.asset_histories[asset]['action_history'] = deque([np.zeros(3, dtype=np.float32) for _ in range(5)], maxlen=5)
            
            while not done:
                # Get direction from Alpha model
                alpha_action, _ = self.alpha_model.predict(obs, deterministic=True)
                
                # Parse Alpha action to get directions (Stage 1: 5 outputs)
                directions = {}
                for i, asset in enumerate(self.env.assets):
                    direction_raw = alpha_action[i]
                    direction = 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0)
                    directions[asset] = direction
                
                # For each asset with direction != 0, get risk parameters
                combined_actions = {}
                
                # Debug logging for Alpha outputs (occasional)
                if step_count % 1000 == 0:
                     logger.info(f"DEBUG Step {step_count}: Alpha raw outputs: {alpha_action}")
                     logger.info(f"DEBUG Step {step_count}: Directions: {directions}")
                
                for asset, direction in directions.items():
                    # REMOVE XAUUSD SKIP: Allow all assets to trade
                    # if asset == 'XAUUSD':
                    #     continue
                        
                    if direction == 0:
                        continue
                    
                    # Build risk model observation
                    risk_obs = self.build_risk_observation(asset)
                    
                    # Get risk model prediction
                    risk_action, _ = self.risk_model.predict(risk_obs, deterministic=True)
                    
                    # Parse risk action
                    sl_mult, tp_mult, risk_raw = self.parse_risk_action(risk_action)
                    
                    # Debug logging for Risk outputs (occasional)
                    if step_count % 1000 == 0:
                         logger.info(f"DEBUG Step {step_count} [{asset}]: Dir {direction}, Risk Raw {risk_raw:.4f}, SL {sl_mult:.2f}, TP {tp_mult:.2f}")

                    # Get current price and ATR
                    current_prices = self.env._get_current_prices()
                    atrs = self.env._get_current_atrs()
                    entry_price = current_prices[asset]
                    atr = atrs[asset]
                    
                    # Calculate position size
                    size_pct, risk_info = self.calculate_position_size(
                        asset, entry_price, atr, sl_mult, tp_mult, risk_raw, direction
                    )
                    
                    if size_pct > 0 and risk_info:
                        # Store action in history
                        self.asset_histories[asset]['action_history'].append(
                            np.array([sl_mult, tp_mult, risk_raw], dtype=np.float32)
                        )
                        
                        # Create combined action for Alpha environment
                        combined_actions[asset] = {
                            'direction': direction,
                            'size': size_pct,
                            'sl_mult': sl_mult,
                            'tp_mult': tp_mult
                        }
                    elif step_count % 1000 == 0:
                         logger.info(f"DEBUG Step {step_count} [{asset}]: Trade skipped. Size pct: {size_pct}")
                
                # Reset step trackers
                self.env.peeked_pnl_step = 0.0
                self.env.completed_trades = []
                
                # Execute trades manually with combined parameters
                current_prices = self.env._get_current_prices()
                atrs = self.env._get_current_atrs()
                
                for asset, act in combined_actions.items():
                    direction = act['direction']
                    current_pos = self.env.positions[asset]
                    price = current_prices[asset]
                    atr = atrs[asset]
                    
                    # Execute trade manually
                    if current_pos is None:
                        # No position: open if direction != 0
                        if direction != 0:
                            position_act = {
                                'direction': direction,
                                'size': act['size'],
                                'sl_mult': act['sl_mult'],
                                'tp_mult': act['tp_mult']
                            }
                            self.env._open_position(asset, direction, position_act, price, atr)
                    elif current_pos['direction'] == direction:
                        # Same direction: hold
                        pass
                    elif direction != 0 and current_pos['direction'] != direction:
                        # Opposite direction: close and reverse
                        self.env._close_position(asset, price)
                        position_act = {
                            'direction': direction,
                            'size': act['size'],
                            'sl_mult': act['sl_mult'],
                            'tp_mult': act['tp_mult']
                        }
                        self.env._open_position(asset, direction, position_act, price, atr)
                    elif direction == 0 and current_pos is not None:
                        # Flat signal: close position
                        self.env._close_position(asset, price)
                
                # Also close positions for assets with direction == 0 that aren't in combined_actions
                for asset in self.env.assets:
                    if asset not in combined_actions and self.env.positions[asset] is not None:
                        self.env._close_position(asset, current_prices[asset])
                
                # Advance time and check SL/TP (manually call internal methods)
                self.env.current_step += 1
                self.env._update_positions()
                
                # Calculate reward
                reward = self.env._calculate_reward()
                
                # Termination checks
                terminated = False
                truncated = self.env.current_step >= self.env.max_steps
                
                # Update peak equity
                self.env.peak_equity = max(self.env.peak_equity, self.env.equity)
                drawdown = 1.0 - (self.env.equity / self.env.peak_equity)
                
                # Build info dict
                info = {
                    'trades': self.env.completed_trades,
                    'equity': self.env.equity,
                    'drawdown': drawdown,
                    'timestamp': self.env._get_current_timestamp()
                }
                
                # Get next observation
                obs = self.env._validate_observation(self.env._get_observation())
                done = terminated or truncated
                
                # Update equity tracking
                self.equity = self.env.equity
                self.peak_equity = max(self.peak_equity, self.equity)
                
                # Update histories with completed trades
                if info and 'trades' in info:
                    for trade in info['trades']:
                        asset = trade['asset']
                        pnl_ratio = trade.get('net_pnl', 0) / max(self.initial_equity, 1e-6)
                        self.asset_histories[asset]['pnl_history'].append(pnl_ratio)
                        metrics_tracker.add_trade(trade)
                
                # Log equity
                if 'equity' in info and 'timestamp' in info:
                    metrics_tracker.add_equity_point(info['timestamp'], info['equity'])
                
                step_count += 1
                
                if step_count % 1000 == 0:
                    logger.info(f"Step {step_count}, Equity: ${self.equity:.2f}")
            
            final_equity = self.env.equity
            logger.info(f"Episode {episode + 1}/{episodes} complete. Steps: {step_count}, Final Equity: ${final_equity:.2f}")
        
        return metrics_tracker


def run_combined_backtest(args):
    """Main backtesting function"""
    logger.info("Starting combined Alpha-Risk backtest")
    logger.info(f"Alpha model: {args.alpha_model}")
    logger.info(f"Risk model: {args.risk_model}")
    logger.info(f"Data directory: {args.data_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validate model files
    if not os.path.exists(args.alpha_model):
        logger.error(f"Alpha model file not found: {args.alpha_model}")
        sys.exit(1)
    
    if not os.path.exists(args.risk_model):
        logger.error(f"Risk model file not found: {args.risk_model}")
        sys.exit(1)
    
    # Load Alpha model
    logger.info("Loading Alpha model...")
    env = DummyVecEnv([lambda: TradingEnv(data_dir=args.data_dir, stage=1, is_training=False)])
    
    # Load VecNormalize stats if available
    vecnorm_path = args.alpha_model.replace('.zip', '_vecnormalize.pkl')
    if os.path.exists(vecnorm_path):
        logger.info(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    
    alpha_model = PPO.load(args.alpha_model, env=env)
    logger.info("Alpha model loaded successfully")
    
    # Load Risk model
    logger.info("Loading Risk model...")
    # Risk model needs an environment for loading, but we can use a dummy dataset path
    # The environment won't be used for inference, just for model loading
    from RiskLayer.src.risk_env import RiskManagementEnv
    
    # Try to find risk dataset, use dummy if not found
    risk_dataset_paths = [
        os.path.join(os.path.dirname(__file__), '../../RiskLayer/risk_dataset.parquet'),
        os.path.join(os.path.dirname(__file__), '../../../RiskLayer/risk_dataset.parquet'),
        'RiskLayer/risk_dataset.parquet',
        'risk_dataset.parquet'
    ]
    
    risk_dataset_path = None
    for path in risk_dataset_paths:
        if os.path.exists(path):
            risk_dataset_path = path
            break
    
    if risk_dataset_path is None:
        # Create a minimal dummy dataset for loading
        logger.warning("Risk dataset not found, creating minimal dummy dataset for model loading")
        import tempfile
        dummy_data = pd.DataFrame({
            'direction': [1] * 100,
            'entry_price': [1.0] * 100,
            'atr_14': [0.001] * 100,
            'max_profit_pct': [0.01] * 100,
            'max_loss_pct': [-0.01] * 100,
            'close_1000_price': [1.0] * 100,
            'features': [np.zeros(140, dtype=np.float32)] * 100
        })
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
        dummy_data.to_parquet(temp_file.name)
        risk_dataset_path = temp_file.name
    
    risk_env = RiskManagementEnv(
        dataset_path=risk_dataset_path,
        initial_equity=initial_equity,
        is_training=False
    )
    risk_model = PPO.load(args.risk_model, env=risk_env)
    logger.info("Risk model loaded successfully")
    
    # Create combined backtest
    backtest = CombinedBacktest(alpha_model, risk_model, args.data_dir, initial_equity=initial_equity)

    
    # Run backtest
    metrics_tracker = backtest.run_backtest(episodes=args.episodes)
    
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
    
    # 1. Save metrics JSON
    metrics_file = os.path.join(args.output_dir, f"metrics_combined_{timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # 2. Save trade log
    if metrics_tracker.trades:
        trades_file = os.path.join(args.output_dir, f"trades_combined_{timestamp}.csv")
        pd.DataFrame(metrics_tracker.trades).to_csv(trades_file, index=False)
        logger.info(f"Saved trade log to {trades_file}")
    
    # 3. Save per-asset performance
    per_asset = metrics_tracker.get_per_asset_metrics()
    if per_asset:
        asset_file = os.path.join(args.output_dir, f"asset_breakdown_combined_{timestamp}.csv")
        pd.DataFrame(per_asset).T.to_csv(asset_file)
        logger.info(f"Saved per-asset breakdown to {asset_file}")
    
    # 4. Generate all visualizations
    if metrics_tracker.equity_curve and metrics_tracker.trades:
        logger.info("\nGenerating comprehensive charts...")
        generate_all_charts(metrics_tracker, per_asset, "combined", args.output_dir, timestamp)
    
    logger.info("\nBacktest complete!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Alpha-Risk Model Backtest")
    parser.add_argument("--alpha-model", type=str, required=True,
                       help="Path to Alpha model (.zip file)")
    parser.add_argument("--risk-model", type=str, required=True,
                       help="Path to Risk model (.zip file)")
    parser.add_argument("--data-dir", type=str, default="Alpha/backtest/data",
                       help="Path to backtest data directory")
    parser.add_argument("--output-dir", type=str, default="Alpha/backtest/results",
                       help="Path to save results")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run")
    
    args = parser.parse_args()
    
    # Validate data directory
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    parquet_files = [f for f in os.listdir(args.data_dir) if f.endswith('.parquet')]
    if not parquet_files:
        logger.error(f"No .parquet files found in {args.data_dir}")
        sys.exit(1)
    
    run_combined_backtest(args)


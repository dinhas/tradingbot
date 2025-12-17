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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from ..src.trading_env import TradingEnv
from .backtest import BacktestMetrics, NumpyEncoder, generate_all_charts
from RiskLayer.src.risk_env import RiskManagementEnv

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
        
        # Track blocked trades
        self.blocked_trades = []
        
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
                    
                    # --- BLOCKING LOGIC ---
                    # Threshold: 10% risk knob (0.10).
                    # If risk_raw < 0.10, we BLOCK.
                    
                    if risk_raw < 0.10:
                        # BLOCKED TRADE
                        # Simulate what would have happened (Virtual Trade) since we have the direction and SL/TP
                        
                        # 1. Get current market data for simulation
                        current_prices = self.env._get_current_prices()
                        atrs = self.env._get_current_atrs()
                        entry_price = current_prices[asset]
                        atr = atrs[asset]
                        
                        # Handle zero ATR (rare)
                        if atr <= 0: atr = entry_price * 0.0001

                        # Calculate absolute SL/TP prices for simulation
                        sl_dist = sl_mult * atr
                        tp_dist = tp_mult * atr
                        sl_price = entry_price - (direction * sl_dist)
                        tp_price = entry_price + (direction * tp_dist)
                        
                        # 2. Create Virtual Position (Notional Size 1.0 for PnL %)
                        # We just want the % return, size doesn't matter for sign checking
                        virtual_pos = {
                            'direction': direction,
                            'entry_price': entry_price,
                            'size': 1.0, # Dummy size
                            'sl': sl_price,
                            'tp': tp_price,
                            'entry_step': self.env.current_step,
                            'sl_dist': sl_dist,
                            'tp_dist': tp_dist
                        }
                        
                        # 3. Inject Virtual Position
                        # Store existing (should be None if we are checking new entry, but safety first)
                        original_pos = self.env.positions[asset]
                        self.env.positions[asset] = virtual_pos
                        
                        # 4. Simulate Outcome (Peek Ahead)
                        # This looks ahead up to 1000 steps to see if SL or TP is hit
                        # Returns PnL amount. Since size is 1.0, this is roughly PnL % (times leverage if calc used it)
                        # In _simulate_trade_outcome: pnl = price_change_pct * (pos['size'] * self.leverage)
                        # Env leverage is 100.
                        simulated_pnl_value = self.env._simulate_trade_outcome(asset)
                        
                        # 5. Restore State
                        self.env.positions[asset] = original_pos
                        
                        # 6. Record Blocked Trade
                        # PnL > 0 means we blocked a WIN (Bad Block / Missed Opportunity)
                        # PnL < 0 means we blocked a LOSS (Good Block / Saved Money)
                        self.blocked_trades.append({
                            'timestamp': self.env._get_current_timestamp(),
                            'asset': asset,
                            'direction': direction,
                            'risk_raw': risk_raw,
                            'sl_mult': sl_mult,
                            'tp_mult': tp_mult,
                            'theoretical_pnl': simulated_pnl_value,
                            'outcome': 'WIN' if simulated_pnl_value > 0 else ('LOSS' if simulated_pnl_value < 0 else 'NEUTRAL')
                        })
                        
                        if step_count % 1000 == 0:
                             logger.info(f"DEBUG Step {step_count} [{asset}]: BLOCKED. Risk {risk_raw:.4f} < 0.10. Avoided: {simulated_pnl_value:.4f}")
                        
                        continue # Skip actual trade execution logic for this asset

                    # If NOT blocked, Rescale Risk to use full range [0, 1]
                    # Map [0.10, 1.0] -> [0.0, 1.0]
                    risk_raw = (risk_raw - 0.10) / 0.90
                    
                    # Debug logging for Risk outputs (occasional)
                    if step_count % 1000 == 0:
                         logger.info(f"DEBUG Step {step_count} [{asset}]: Dir {direction}, Risk Raw (Rescaled) {risk_raw:.4f}, SL {sl_mult:.2f}, TP {tp_mult:.2f}")

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
                         logger.info(f"DEBUG Step {step_count} [{asset}]: Trade skipped (Size 0). Size pct: {size_pct}")
                
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
    
    # Load Alpha model
    logger.info("Loading Alpha model...")
    env = DummyVecEnv([lambda: TradingEnv(data_dir=data_dir_path, stage=1, is_training=False)])
    
    # Load VecNormalize stats if available
    vecnorm_path = str(alpha_model_path).replace('.zip', '_vecnormalize.pkl')
    if os.path.exists(vecnorm_path):
        logger.info(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    
    alpha_model = PPO.load(alpha_model_path, env=env)
    logger.info("Alpha model loaded successfully")
    
    # Load Risk model
    logger.info("Loading Risk model...")
    
    # FORCE DUMMY DATASET to avoid OSError and speed up loading
    # The environment is only used for model initialization, not for actual backtest data
    import tempfile
    
    logger.info("Creating dummy dataset for Risk Model loading (bypassing full dataset cache)...")
    dummy_data = pd.DataFrame({
        'direction': [1] * 100,
        'entry_price': [1.0] * 100,
        'atr': [0.001] * 100, # Note: 'atr' column name expected by RiskEnv defaults
        'atr_14': [0.001] * 100, # Backup
        'max_profit_pct': [0.01] * 100,
        'max_loss_pct': [-0.01] * 100,
        'close_1000_price': [1.0] * 100,
        'features': [np.zeros(140, dtype=np.float32)] * 100,
        'pair': ['EURUSD'] * 100
    })
    
    # specific columns required by risk_env _load_data sanity check
    # 'max_profit_pct', 'max_loss_pct', 'close_1000_price'
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as temp_file:
        dummy_data.to_parquet(temp_file.name)
        dummy_risk_dataset_path = Path(temp_file.name)
    
    try:
        risk_env = RiskManagementEnv(
            dataset_path=str(dummy_risk_dataset_path),
            initial_equity=initial_equity,
            is_training=False
        )
        
        # Check for VecNormalize for Risk Model
        # Assuming it might be named similar to model or generic 'vec_normalize.pkl' in model dir
        risk_model_dir = risk_model_path.parent
        possible_vec_paths = [
            str(risk_model_path).replace('.zip', '_vecnormalize.pkl'),
            risk_model_dir / 'vec_normalize.pkl',
            risk_model_dir / 'risk_model_final_vecnormalize.pkl'
        ]
        
        found_vec = False
        for vp in possible_vec_paths:
            if os.path.exists(vp):
                logger.info(f"Loading Risk Model VecNormalize stats from {vp}")
                risk_env = DummyVecEnv([lambda: risk_env]) # Wrap in VecEnv first
                risk_env = VecNormalize.load(vp, risk_env)
                risk_env.training = False
                risk_env.norm_reward = False
                found_vec = True
                break
        
        if not found_vec:
            logger.warning("CRITICAL: No VecNormalize stats found for Risk Model! Predictions may be garbage if model expects normalized inputs.")
            logger.warning(f"Checked: {possible_vec_paths}")

        risk_model = PPO.load(risk_model_path, env=risk_env)
        logger.info("Risk model loaded successfully")
        
    finally:
        # Cleanup dummy file
        try:
            os.remove(dummy_risk_dataset_path)
        except:
            pass
    
    # Create combined backtest
    backtest = CombinedBacktest(alpha_model, risk_model, data_dir_path, initial_equity=initial_equity)

    
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
    parser.add_argument("--alpha-model", type=str, required=True,
                        help="Path to Alpha model (.zip file) relative to project root")
    parser.add_argument("--risk-model", type=str, required=True,
                        help="Path to Risk model (.zip file) relative to project root")
    parser.add_argument("--data-dir", type=str, default="Alpha/backtest/data",
                        help="Path to backtest data directory relative to project root")
    parser.add_argument("--output-dir", type=str, default="Alpha/backtest/results",
                        help="Path to save results relative to project root")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    
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


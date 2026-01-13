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
# Fix for WinError 1114: Import torch before other libraries (especially numpy/scipy)
import torch
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
try:
    from RiskLayer.env.risk_env import RiskTradingEnv
    from RiskLayer.src.feature_engine import RiskFeatureEngine
except ImportError:
    # Fallback paths
    sys.path.append(os.path.join(project_root, "RiskLayer"))
    from RiskLayer.env.risk_env import RiskTradingEnv
    from RiskLayer.src.feature_engine import RiskFeatureEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

initial_equity=10000.0

class CombinedBacktest:
    """Combined backtest using Alpha model for direction and Risk model for SL/TP/sizing"""
    
    def __init__(self, alpha_model, risk_model, data_dir, initial_equity=initial_equity, alpha_norm_env=None, risk_norm_env=None, env=None, use_spreads=True):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.alpha_norm_env = alpha_norm_env
        self.risk_norm_env = risk_norm_env
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.use_spreads = use_spreads
        
        # Create Alpha environment for data access (or reuse existing)
        if env is not None:
            self.env = env
        else:
            self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
            
        self.env.equity = initial_equity
        self.env.start_equity = initial_equity
        self.env.peak_equity = initial_equity
        
        # Initialize Risk Feature Engine
        self.risk_feature_engine = RiskFeatureEngine()
        logger.info("Preprocessing Risk features for backtest...")
        
        # FIX: Align Backtest Data Structure with Training Data Structure
        # Training data had [OHLCV, alpha_signal, alpha_confidence].
        # If we don't add them here, they end up at the END of the feature list, shifting all other features.
        # We also need alpha_confidence to match the 84 feature count (Training had alpha_confidence + generated alpha_conf).
        for asset in self.env.data:
            if 'alpha_signal' not in self.env.data[asset].columns:
                self.env.data[asset]['alpha_signal'] = 0.0
            if 'alpha_confidence' not in self.env.data[asset].columns:
                self.env.data[asset]['alpha_confidence'] = 0.0
                
        # TradingEnv already has raw data in self.env.data
        self.risk_processed_data = self.risk_feature_engine.preprocess_data(self.env.data)
        
        # Build numeric mapping for Risk Model (matching RiskTradingEnv)
        numeric_df = self.risk_processed_data.select_dtypes(include=[np.number])
        self.risk_master_matrix = numeric_df.astype(np.float32).values
        self.risk_column_map = {col: i for i, col in enumerate(numeric_df.columns)}
        
        # Cache indices for each asset to speed up obs building
        self.risk_asset_indices = {}
        self.risk_asset_cols = {}  # Also cache the column names for signal injection
        for asset in self.env.assets:
            asset_cols = [c for c in numeric_df.columns if c.startswith(f"{asset}_")]
            self.risk_asset_indices[asset] = [self.risk_column_map[c] for c in asset_cols]
            self.risk_asset_cols[asset] = asset_cols
        
        # Risk model parameters (match RiskTradingEnv)
        self.ATR_SL_MIN = 3.0
        self.ATR_SL_MAX = 7.0
        self.ATR_TP_MIN = 1.0
        self.ATR_TP_MAX = 15.0
        self.EXECUTION_THRESHOLD = 0.2
        
        # Spreads (match RiskTradingEnv)
        if self.use_spreads:
            self.spreads = {
                'EURUSD': 0.0001,
                'GBPUSD': 0.00015,
                'USDJPY': 0.01,
                'USDCHF': 0.00015,
                'XAUUSD': 0.20
            }
        else:
            logger.info("⚠️ SPREADS DISABLED for this backtest")
            self.spreads = {
                'EURUSD': 0.0,
                'GBPUSD': 0.0,
                'USDJPY': 0.0,
                'USDCHF': 0.0,
                'XAUUSD': 0.0
            }
            
        # Asset ID mapping (Match RiskTradingEnv)
        self.risk_assets_list = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.asset_ids = {asset: float(i) for i, asset in enumerate(self.risk_assets_list)}
        
        # Per-asset history tracking
        self.asset_histories = {
            asset: {
                'pnl_history': deque([0.0] * 5, maxlen=5),
                'action_history': deque([np.zeros(2, dtype=np.float32) for _ in range(5)], maxlen=5)
            }
            for asset in self.env.assets
        }
        
        # Position sizing parameters (match TradingEnv)
        self.MAX_MARGIN_PER_TRADE_PCT = 0.25
        self.MAX_LEVERAGE = 30.0
        self.MIN_LOTS = 0.01
        self.MAX_POS_SIZE_PCT = 0.50
        
        self.equity = initial_equity
        self.peak_equity = initial_equity

    def _map_range(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def build_risk_observation(self, asset, current_step, alpha_signal, alpha_conf):
        """Build observation vector for the new risk model"""
        # 1. Get pre-calculated features for this step
        indices = self.risk_asset_indices[asset]
        obs = self.risk_master_matrix[current_step, indices].copy()
        
        # 2. Inject REAL-TIME Alpha signals into the observation
        # We need to find where alpha_signal and alpha_conf are in the vector
        # They are usually named f"{asset}_alpha_signal" and f"{asset}_alpha_conf"
        
        sig_col = f"{asset}_alpha_signal"
        conf_col = f"{asset}_alpha_conf"
        
        # Use cached column list for consistent ordering
        asset_cols = self.risk_asset_cols[asset]
        
        sig_idx = -1
        conf_idx = -1
        
        if sig_col in asset_cols:
            sig_idx = asset_cols.index(sig_col)
        elif "alpha_signal" in asset_cols:
            sig_idx = asset_cols.index("alpha_signal")
            
        if conf_col in asset_cols:
            conf_idx = asset_cols.index(conf_col)
        elif "alpha_conf" in asset_cols:
            conf_idx = asset_cols.index("alpha_conf")
        
        # Also check for alpha_confidence (from our fix)
        conf_idx_2 = -1
        if f"{asset}_alpha_confidence" in asset_cols:
            conf_idx_2 = asset_cols.index(f"{asset}_alpha_confidence")
            
        if sig_idx != -1:
            obs[sig_idx] = alpha_signal
        if conf_idx != -1:
            obs[conf_idx] = alpha_conf
        if conf_idx_2 != -1:
            obs[conf_idx_2] = alpha_conf
            
        # Append Static Features (Spread, Asset ID) - MATCHING RISK ENV
        spread = self.spreads.get(asset, 0.0)
        asset_id = self.asset_ids.get(asset, -1.0)
        
        obs = np.concatenate([obs, [spread, asset_id]])
            
        # Debug injection
        if current_step % 1000 == 0 or current_step < 10:
             logger.info(f"  [DEBUG] Injecting into {asset}: SigIdx={sig_idx}, ConfIdx={conf_idx} | Vals: {alpha_signal}, {alpha_conf} | Added Spread={spread}, ID={asset_id}")
             if sig_idx != -1:
                 logger.info(f"    > Verified Injected Sig: {obs[sig_idx]}")

        # Handle NaNs
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
            
        if self.risk_norm_env is not None:
            obs_normalized = self.risk_norm_env.normalize_obs(obs.reshape(1, -1)).flatten()
            return obs_normalized
            
        return obs
    
    def parse_risk_action(self, action):
        """Parse risk model action (3 outputs) to Execution/SL/TP"""
        # [0] Execution Confidence, [1] SL Mult, [2] TP Mult
        exec_conf = action[0]
        sl_norm = action[1]
        tp_norm = action[2]
        
        sl_mult = self._map_range(sl_norm, -1, 1, self.ATR_SL_MIN, self.ATR_SL_MAX)
        tp_mult = self._map_range(tp_norm, -1, 1, self.ATR_TP_MIN, self.ATR_TP_MAX)
        
        return exec_conf, sl_mult, tp_mult
    
    def calculate_position_size(self, asset, entry_price, atr, sl_mult, tp_mult, risk_raw, direction):
        """Calculate position size from risk percentage"""
        if risk_raw < 1e-3:
            return 0.0, None
        
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        actual_risk_pct = risk_raw * risk_cap_mult
        
        sl_dist_price = sl_mult * atr
        min_sl_dist = max(0.0001 * entry_price, 0.2 * atr)
        if sl_dist_price < min_sl_dist:
            sl_dist_price = min_sl_dist
        
        contract_size = 100 if asset == 'XAUUSD' else 100000
        is_usd_quote = asset in ['EURUSD', 'GBPUSD', 'XAUUSD']
        is_usd_base = asset in ['USDJPY', 'USDCHF']
        
        risk_amount_cash = self.equity * actual_risk_pct
        
        lots = 0.0
        if sl_dist_price > 0:
            if is_usd_quote:
                lots = risk_amount_cash / (sl_dist_price * contract_size)
            elif is_usd_base:
                lots = (risk_amount_cash * entry_price) / (sl_dist_price * contract_size)
            else:
                lots = risk_amount_cash / (sl_dist_price * contract_size)
        
        if is_usd_quote:
            lot_value_usd = contract_size * entry_price
        elif is_usd_base:
            lot_value_usd = contract_size * 1.0
        else:
            lot_value_usd = contract_size * 1.0
        
        max_position_value = (self.equity * self.MAX_MARGIN_PER_TRADE_PCT) * self.MAX_LEVERAGE
        max_lots_leverage = max_position_value / (lot_value_usd + 1e-9)
        lots = min(lots, max_lots_leverage)
        
        if lots < self.MIN_LOTS:
            min_lot_value_usd = self.MIN_LOTS * lot_value_usd / (entry_price if is_usd_quote else 1.0)
            margin_required_min = min_lot_value_usd / self.MAX_LEVERAGE
            
            if self.equity > (margin_required_min * 1.05):
                lots = self.MIN_LOTS
            else:
                return 0.0, None
        
        lots = np.clip(lots, self.MIN_LOTS, 100.0)
        
        # Convert to Alpha's size_pct
        if is_usd_quote:
            notional_value = lots * contract_size * entry_price
        else:
            notional_value = lots * contract_size
            
        margin_required = notional_value / self.MAX_LEVERAGE
        MAX_POS_SIZE_PCT = 0.50
        size_pct = margin_required / (MAX_POS_SIZE_PCT * self.equity + 1e-9)
        size_pct = np.clip(size_pct, 0.0, 1.0)
        
        position_size = size_pct * MAX_POS_SIZE_PCT * self.equity
        
        return size_pct, {
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'risk_raw': risk_raw,
            'lots': lots,
            'position_size': position_size
        }
    
    def run_backtest(self, episodes=1, max_steps=None):
        """Run combined backtest"""
        metrics_tracker = BacktestMetrics()
        logger.info(f"Running {episodes} episodes...")
        
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
            
            total_steps = 0
            alpha_signals_count = 0
            risk_approvals_count = 0
            risk_blocked_count = 0
            alpha_min = 100.0
            alpha_max = -100.0
            
            while not done:
                # Check explicit step limit
                if max_steps is not None and step_count >= max_steps:
                    logger.info(f"Reached max steps limit: {max_steps}")
                    break

                combined_actions = {}
                alpha_predictions = {}
                
                # 1. Get Alpha and Risk predictions
                current_prices = self.env._get_current_prices()
                atrs = self.env._get_current_atrs()
                
                for asset in self.env.assets:
                    self.env.set_asset(asset)
                    alpha_obs = self.env._get_observation()
                    
                    # Alpha Prediction
                    pred_obs = alpha_obs
                    if self.alpha_norm_env is not None:
                        pred_obs = self.alpha_norm_env.normalize_obs(alpha_obs.reshape(1, -1)).flatten()
                        
                    alpha_action, _ = self.alpha_model.predict(pred_obs, deterministic=True)
                    alpha_val = alpha_action[0]
                    
                    # Track Alpha values
                    alpha_min = min(alpha_min, alpha_val)
                    alpha_max = max(alpha_max, alpha_val)
                    
                    # Determine Direction
                    direction = 1 if alpha_val > 0.33 else (-1 if alpha_val < -0.33 else 0)
                    alpha_predictions[asset] = direction
                    
                    if direction == 0:
                        continue
                        
                    alpha_signals_count += 1
                        
                    # Build and predict with Risk model
                    risk_obs = self.build_risk_observation(asset, self.env.current_step, direction, alpha_val)
                    risk_action, _ = self.risk_model.predict(risk_obs, deterministic=True)
                    
                    exec_conf, sl_mult, tp_mult = self.parse_risk_action(risk_action)
                    
                    # Risk Decision: Only OPEN if confidence > THRESHOLD
                    if exec_conf > self.EXECUTION_THRESHOLD:
                        risk_approvals_count += 1
                        combined_actions[asset] = {
                            'direction': direction,
                            'size': 0.5, # Fixed size placeholder (could use sizing logic)
                            'sl_mult': sl_mult,
                            'tp_mult': tp_mult
                        }
                    else:
                        risk_blocked_count += 1
                
                # 2. Execute trades based on Alpha Signal + Risk Model Approval
                for asset in self.env.assets:
                    act = combined_actions.get(asset) 
                    current_pos = self.env.positions.get(asset)
                    price_bid = current_prices[asset]
                    atr = atrs[asset]
                    spread = self.spreads.get(asset, 0.0)
                    direction = alpha_predictions.get(asset, 0)
                    
                    if act:
                        direction = act['direction']
                        entry_price = (price_bid + spread) if direction == 1 else price_bid
                        
                        if current_pos is None:
                            # New Position
                            self.env._open_position(asset, direction, act, entry_price, atr)
                        elif current_pos['direction'] != direction:
                            # Reversal
                            exit_price = price_bid if current_pos['direction'] == 1 else (price_bid + spread)
                            self.env._close_position(asset, exit_price)
                            self.env._open_position(asset, direction, act, entry_price, atr)
                    
                    elif current_pos is not None:
                        # No Risk approval for a NEW trade, but we have an OLD trade.
                        # Check if Alpha has REVERSED according to its latest signal.
                        if direction != 0 and direction != current_pos['direction']:
                            # Alpha flipped to opposite, but Risk didn't approve the new trade.
                            # We should still exit the old trade.
                            exit_price = price_bid if current_pos['direction'] == 1 else (price_bid + spread)
                            self.env._close_position(asset, exit_price)
                        # else: Hold until SL/TP or reversal.
                
                # 3. Advance time and update positions (SL/TP)
                self.env.current_step += 1
                self._update_positions_with_spread()
                
                # Update metrics
                self.equity = self.env.equity
                self.peak_equity = max(self.peak_equity, self.equity)
                
                if self.env.completed_trades:
                    for trade in self.env.completed_trades:
                        metrics_tracker.add_trade(trade)
                    self.env.completed_trades = []
                    
                metrics_tracker.add_equity_point(self.env._get_current_timestamp(), self.equity)
                
                step_count += 1
                done = self.env.current_step >= self.env.max_steps
                
                if step_count % 1000 == 0:
                    logger.info(f"Step {step_count} | Equity ${self.equity:.0f} | Trades: {len(metrics_tracker.trades)}")
            
            logger.info(f"Episode {episode + 1} complete. Final Equity: ${self.env.equity:.2f}")
            logger.info("\n" + "-"*30)
            logger.info("RISK FILTER SUMMARY")
            logger.info(f"Total Alpha Signals:  {alpha_signals_count}")
            logger.info(f"Risk Approved:        {risk_approvals_count}")
            logger.info(f"Risk Blocked:         {risk_blocked_count}")
            filter_rate = (risk_blocked_count / alpha_signals_count * 100) if alpha_signals_count > 0 else 0
            logger.info(f"Filter Rate:          {filter_rate:.2f}%")
            logger.info("-"*30 + "\n")
        
        return metrics_tracker

    def _update_positions_with_spread(self):
        """Check SL/TP for all open positions using Bid/Ask logic with High/Low prices."""
        step = self.env.current_step
        
        for asset, pos in list(self.env.positions.items()):
            if pos is None: continue
            
            # Use High/Low for more accurate SL/TP simulation
            high_bid = self.env.high_arrays[asset][step]
            low_bid = self.env.low_arrays[asset][step]
            spread = self.spreads.get(asset, 0.0)
            
            if pos['direction'] == 1: # Long
                # SL hit if Low Bid <= SL
                if low_bid <= pos['sl']:
                    self.env._close_position(asset, pos['sl'])
                # TP hit if High Bid >= TP
                elif high_bid >= pos['tp']:
                    self.env._close_position(asset, pos['tp'])
            else: # Short
                # Ask prices for short checking
                high_ask = high_bid + spread
                low_ask = low_bid + spread
                
                # SL hit if High Ask >= SL
                if high_ask >= pos['sl']:
                    self.env._close_position(asset, pos['sl'])
                # TP hit if Low Ask <= TP
                elif low_ask <= pos['tp']:
                    self.env._close_position(asset, pos['tp'])


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
    if args.steps:
        logger.info(f"Step limit: {args.steps}")
    
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
    alpha_norm_env = None
    alpha_norm_path = Path(str(alpha_model_path).replace('.zip', '_vecnormalize.pkl'))
    if not alpha_norm_path.exists():
        alpha_norm_path = Path(str(alpha_model_path).replace('_model.zip', '_vecnormalize.pkl'))
    
    if alpha_norm_path.exists():
        logger.info(f"Loading Alpha Normalizer from {alpha_norm_path}")
        alpha_norm_env = VecNormalize.load(str(alpha_norm_path), dummy_vec_env)
        alpha_norm_env.training = False
        alpha_norm_env.norm_reward = False
    else:
        logger.warning(f"Alpha Normalizer NOT found at {alpha_norm_path}")
    
    # Load Alpha model
    logger.info("Loading Alpha model...")
    alpha_model = PPO.load(alpha_model_path, env=dummy_vec_env)
    logger.info(f"Alpha model loaded. Obs Space: {alpha_model.observation_space}")
    
    # Load Risk model
    logger.info("Loading Risk model...")
    risk_model = PPO.load(risk_model_path, env=None)
    logger.info(f"Risk model loaded. Obs Space: {risk_model.observation_space}")
    
    # Load Risk normalizer
    risk_norm_path = risk_model_path.parent / "vec_normalize.pkl"
    if not risk_norm_path.exists():
        risk_norm_path = Path(str(risk_model_path).replace('.zip', '_vecnormalize.pkl'))
    
    # NEW FALLBACK: handle 10M.zip -> 10M.pkl
    if not risk_norm_path.exists():
        risk_norm_path = Path(str(risk_model_path).replace('.zip', '.pkl'))
        
    risk_norm_env = None
    if risk_norm_path.exists():
        logger.info(f"Loading Risk Normalizer from {risk_norm_path}")
        try:
            # Create a simple dummy vec env with the correct obs space for loading
            import gymnasium as gym
            from gymnasium import spaces
            class SimpleObsEnv(gym.Env):
                def __init__(self, obs_space):
                    super().__init__()
                    self.observation_space = obs_space
                    self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
                def reset(self, seed=None): return np.zeros(self.observation_space.shape), {}
                def step(self, action): return np.zeros(self.observation_space.shape), 0, False, False, {}

            dummy_risk_vec = DummyVecEnv([lambda: SimpleObsEnv(risk_model.observation_space)])
            risk_norm_env = VecNormalize.load(str(risk_norm_path), dummy_risk_vec)
            risk_norm_env.training = False
            risk_norm_env.norm_reward = False
            logger.info("Risk Normalizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Risk Normalizer: {e}")
    else:
        logger.warning(f"Risk Normalizer NOT found at {risk_norm_path}")
    
    # Create combined backtest using the SHARED environment
    backtest = CombinedBacktest(
        alpha_model, 
        risk_model, 
        data_dir_path, 
        initial_equity=initial_equity,
        alpha_norm_env=alpha_norm_env,
        risk_norm_env=risk_norm_env,
        env=shared_env,
        use_spreads=not args.no_spreads
    )

    # Run backtest
    metrics_tracker = backtest.run_backtest(episodes=args.episodes, max_steps=args.steps)



    
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
        
    # 4. Generate all visualizations
    if metrics_tracker.equity_curve and metrics_tracker.trades:
        logger.info("\nGenerating comprehensive charts...")
        generate_all_charts(metrics_tracker, per_asset, "combined", output_dir_path, timestamp)
    
    logger.info("\nBacktest complete!")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Alpha-Risk Model Backtest")
    parser.add_argument("--alpha-model", type=str, default="models/checkpoints/alpha/ppo_final_model.zip",
                        help="Path to Alpha model (.zip file) relative to project root")
    parser.add_argument("--risk-model", type=str, default="models/checkpoints/risk/risk_model_final.zip",
                        help="Path to Risk model (.zip file) relative to project root")
    parser.add_argument("--data-dir", type=str, default="Alpha/backtest/data",
                        help="Path to backtest data directory relative to project root")
    parser.add_argument("--output-dir", type=str, default="Alpha/backtest/results",
                        help="Path to save results relative to project root")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=None,
                        help="Limit number of steps to run (optional)")
    parser.add_argument("--no-spreads", action="store_true",
                        help="Disable spreads (set to 0.0) for debugging")
    
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



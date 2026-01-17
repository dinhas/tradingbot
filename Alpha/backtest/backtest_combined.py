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
    from RiskLayer.src.risk_env import RiskManagementEnv
    from RiskLayer.src.feature_engine import RiskFeatureEngine
except ImportError:
    # Fallback paths
    sys.path.append(os.path.join(project_root, "RiskLayer"))
    from RiskLayer.src.risk_env import RiskManagementEnv
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
        
        # TradingEnv already has raw data in self.env.data
        self.risk_raw_data, self.risk_processed_data = self.risk_feature_engine.preprocess_data(self.env.data)
        
        # Build numeric mapping for Risk Model
        numeric_df = self.risk_processed_data.select_dtypes(include=[np.number])
        self.risk_master_matrix = numeric_df.astype(np.float32).values
        self.risk_column_map = {col: i for i, col in enumerate(numeric_df.columns)}
        
        # Spreads (match RiskManagementEnv)
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
            
        # Per-asset history tracking (Match RiskManagementEnv history)
        self.asset_histories = {
            asset: {
                'history_pnl': deque([0.0] * 5, maxlen=5),
                'history_actions': deque([np.zeros(3, dtype=np.float32) for _ in range(5)], maxlen=5)
            }
            for asset in self.env.assets
        }
        
        # Sizing parameters (match RiskManagementEnv)
        self.MAX_RISK_PER_TRADE = 0.40
        self.MAX_MARGIN_PER_TRADE_PCT = 0.80
        self.MAX_LEVERAGE = 400.0
        self.MIN_LOTS = 0.01
        
        self.equity = initial_equity
        self.peak_equity = initial_equity

    def build_risk_observation(self, asset, current_step):
        """Build 65-dim observation vector for the new risk model"""
        # 1. Market State (40 features from RiskFeatureEngine)
        # portfolio_state for RiskFeatureEngine
        portfolio_state = {
            'equity': self.equity,
            'margin_usage_pct': sum(pos['size'] for pos in self.env.positions.values() if pos is not None) / self.equity if self.equity > 0 else 0,
            'drawdown': 1.0 - (self.equity / self.peak_equity),
            'num_open_positions': sum(1 for p in self.env.positions.values() if p is not None)
        }
        
        current_prices = self.env._get_current_prices()
        for a in self.env.assets:
            pos = self.env.positions[a]
            if pos:
                price_change = (current_prices[a] - pos['entry_price']) * pos['direction']
                price_change_pct = price_change / pos['entry_price'] if pos['entry_price'] != 0 else 0
                unrealized_pnl = price_change_pct * (pos['size'] * self.env.leverage)
                
                portfolio_state[a] = {
                    'has_position': 1,
                    'position_size': pos['size'] / self.equity,
                    'unrealized_pnl': unrealized_pnl,
                    'position_age': self.env.current_step - pos['entry_step'],
                    'entry_price': pos['entry_price'],
                    'current_sl': pos['sl'],
                    'current_tp': pos['tp']
                }
            else:
                portfolio_state[a] = {
                    'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0,
                    'position_age': 0, 'entry_price': 0, 'current_sl': 0, 'current_tp': 0
                }
        
        current_row = self.risk_processed_data.iloc[current_step]
        market_obs = self.risk_feature_engine.get_observation(current_row, portfolio_state, asset)
        
        # 2. Account State (5)
        drawdown = 1.0 - (self.equity / self.peak_equity)
        equity_norm = self.equity / self.initial_equity
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        account_obs = np.array([equity_norm, drawdown, 0.0, risk_cap_mult, 0.0], dtype=np.float32)
        
        # 3. History (20) - PER ASSET
        hist = self.asset_histories[asset]
        hist_pnl = np.array(hist['history_pnl'], dtype=np.float32)
        hist_acts = np.array(hist['history_actions'], dtype=np.float32).flatten()
        
        obs = np.concatenate([market_obs, account_obs, hist_pnl, hist_acts])
        
        if self.risk_norm_env is not None:
            obs = self.risk_norm_env.normalize_obs(obs.reshape(1, -1)).flatten()
            
        return obs
    
    def parse_risk_action(self, action):
        """Parse risk model action (3 outputs) to SL/TP/Risk"""
        # Exact logic from RiskManagementEnv.step
        sl_mult = np.clip((action[0] + 1) / 2 * 1.75 + 0.75, 0.75, 2.5)   # 0.75 - 2.5 ATR
        tp_mult = np.clip((action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)   # 0.5 - 4.0 ATR
        risk_raw = np.clip((action[2] + 1) / 2, 0.0, 1.0)              # 0.0 - 1.0
        
        return sl_mult, tp_mult, risk_raw
    
    def calculate_position_size(self, asset, entry_price, atr, sl_mult, tp_mult, risk_raw, direction):
        """Calculate position size matching RiskManagementEnv logic"""
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        # Actual Risk %
        actual_risk_pct = risk_raw * self.MAX_RISK_PER_TRADE * risk_cap_mult
        
        # Calculate Lots
        sl_dist_price = max(sl_mult * atr, 1e-9)
        min_sl_dist = max(0.0001 * entry_price, 0.20 * atr)
        if sl_dist_price < min_sl_dist: sl_dist_price = min_sl_dist
        
        risk_amount_cash = self.equity * actual_risk_pct
        
        contract_size = 100 if asset == 'XAUUSD' else 100000
        is_usd_quote = asset in ['EURUSD', 'GBPUSD', 'XAUUSD']
        is_usd_base = asset in ['USDJPY', 'USDCHF']
        
        lots = 0.0
        if sl_dist_price > 0:
            if is_usd_quote:
                lots = risk_amount_cash / (sl_dist_price * contract_size)
            elif is_usd_base:
                lots = (risk_amount_cash * entry_price) / (sl_dist_price * contract_size)
            else:
                lots = risk_amount_cash / (sl_dist_price * contract_size)
        
        # Leverage Clamping
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
            return 0.0, None
            
        lots = np.clip(lots, self.MIN_LOTS, 100.0)
        
        # Calculate size_pct for TradingEnv
        if is_usd_quote:
            required_position_size = lots * sl_dist_price * contract_size
        elif is_usd_base:
            required_position_size = (lots * sl_dist_price * contract_size) / entry_price
        else:
            required_position_size = lots * sl_dist_price * contract_size
            
        size_pct = required_position_size / (self.env.MAX_POS_SIZE_PCT * self.equity + 1e-9)
        size_pct = np.clip(size_pct, 0.0, 1.0)
        
        return size_pct, {
            'sl_mult': sl_mult, 'tp_mult': tp_mult, 'risk_raw': risk_raw,
            'lots': lots, 'position_size': required_position_size, 'actual_risk_pct': actual_risk_pct
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
            
            # Reset History
            self.asset_histories = {
                asset: {
                    'history_pnl': deque([0.0] * 5, maxlen=5),
                    'history_actions': deque([np.zeros(3, dtype=np.float32) for _ in range(5)], maxlen=5)
                }
                for asset in self.env.assets
            }
            
            total_signals = 0
            risk_approved = 0
            risk_blocked = 0
            
            while not done:
                if max_steps is not None and step_count >= max_steps: break

                combined_actions = {}
                alpha_directions = {}
                current_prices = self.env._get_current_prices()
                atrs = self.env._get_current_atrs()
                
                for asset in self.env.assets:
                    self.env.set_asset(asset)
                    alpha_obs = self.env._get_observation()
                    
                    if self.alpha_norm_env is not None:
                        alpha_obs = self.alpha_norm_env.normalize_obs(alpha_obs.reshape(1, -1)).flatten()
                        
                    alpha_action, _ = self.alpha_model.predict(alpha_obs, deterministic=True)
                    alpha_val = alpha_action[0]
                    direction = 1 if alpha_val > 0.33 else (-1 if alpha_val < -0.33 else 0)
                    alpha_directions[asset] = direction
                    
                    if direction == 0: continue
                    total_signals += 1
                        
                    # Risk Model Prediction
                    risk_obs = self.build_risk_observation(asset, self.env.current_step)
                    risk_action, _ = self.risk_model.predict(risk_obs, deterministic=True)
                    sl_mult, tp_mult, risk_raw = self.parse_risk_action(risk_action)
                    
                    # BLOCKING logic: Bottom 10% of risk knob = BLOCK
                    if risk_raw < 0.10:
                        risk_blocked += 1
                        self.asset_histories[asset]['history_actions'].append(np.zeros(3))
                        self.asset_histories[asset]['history_pnl'].append(0.0)
                        continue
                    
                    risk_approved += 1
                    risk_raw_scaled = (risk_raw - 0.10) / (1.0 - 0.10)
                    
                    spread = self.spreads.get(asset, 0.0)
                    entry_price = (current_prices[asset] + spread) if direction == 1 else current_prices[asset]
                    
                    size_pct, size_info = self.calculate_position_size(
                        asset, entry_price, atrs[asset], sl_mult, tp_mult, risk_raw_scaled, direction
                    )
                    
                    if size_info:
                        combined_actions[asset] = {
                            'direction': direction, 'size': size_pct,
                            'sl_mult': sl_mult, 'tp_mult': tp_mult, 'lots': size_info['lots'],
                            'risk_actual': size_info['actual_risk_pct']
                        }
                    else:
                        risk_blocked += 1
                        self.asset_histories[asset]['history_actions'].append(np.zeros(3))
                        self.asset_histories[asset]['history_pnl'].append(0.0)
                
                # Execution
                for asset in self.env.assets:
                    act = combined_actions.get(asset) 
                    current_pos = self.env.positions.get(asset)
                    price_mid = current_prices[asset]
                    spread = self.spreads.get(asset, 0.0)
                    
                    if act:
                        entry_price = (price_mid + spread) if act['direction'] == 1 else price_mid
                        if current_pos is None:
                            self.env._open_position(asset, act['direction'], act, entry_price, atrs[asset])
                            self.asset_histories[asset]['history_actions'].append(np.array([act['sl_mult'], act['tp_mult'], act['risk_actual']], dtype=np.float32))
                        elif current_pos['direction'] != act['direction']:
                            exit_price = price_mid if current_pos['direction'] == 1 else (price_mid + spread)
                            self.env._close_position(asset, exit_price)
                            self.env._open_position(asset, act['direction'], act, entry_price, atrs[asset])
                            self.asset_histories[asset]['history_actions'].append(np.array([act['sl_mult'], act['tp_mult'], act['risk_actual']], dtype=np.float32))
                    elif current_pos is not None:
                        # Alpha reversal exit
                        alpha_dir = alpha_directions.get(asset, 0)
                        if alpha_dir != 0 and alpha_dir != current_pos['direction']:
                            exit_price = price_mid if current_pos['direction'] == 1 else (price_mid + spread)
                            self.env._close_position(asset, exit_price)
                
                # Update PNL history after step
                trades_this_step = self.env.completed_trades.copy()
                if trades_this_step:
                    for trade in trades_this_step:
                        asset = trade['asset']
                        pnl_ratio = trade['net_pnl'] / self.equity if self.equity > 0 else 0
                        self.asset_histories[asset]['history_pnl'].append(pnl_ratio)
                        metrics_tracker.add_trade(trade)
                    self.env.completed_trades = []
                
                # For assets that didn't close a trade, we might want to append 0 to pnl history 
                # OR we only append when a trade IS closed (which matches RiskManagementEnv.step better
                # if each step in RiskManagementEnv is a signal, then each signal gets a PnL).
                # Wait, in RiskManagementEnv, history_pnl is appended EVERY step.
                # If it's blocked, it's 0. If it's a trade, it's the outcome.
                # So I should append 0 for signals that were BLOCKED (already done above).
                # For trades that are OPEN, we don't append to pnl history yet.
                # This is a bit tricky since RiskManagementEnv is episodic per trade.
                # In CombinedBacktest, one "step" can have signals for multiple assets.
                
                self.env.current_step += 1
                self._update_positions_with_spread()
                self.equity = self.env.equity
                self.peak_equity = max(self.peak_equity, self.equity)
                metrics_tracker.add_equity_point(self.env._get_current_timestamp(), self.equity)
                
                step_count += 1
                done = self.env.current_step >= self.env.max_steps or self.equity < 10.0
                if step_count % 5000 == 0: logger.info(f"Step {step_count} | Equity ${self.equity:.2f}")
            
            logger.info(f"Episode {episode + 1} complete. Final Equity: ${self.env.equity:.2f}")
            logger.info(f"Signals: {total_signals}, Approved: {risk_approved}, Blocked: {risk_blocked}")
        
        return metrics_tracker

    def _update_positions_with_spread(self):
        """Check SL/TP for all open positions using High/Low prices."""
        step = self.env.current_step
        
        for asset, pos in list(self.env.positions.items()):
            if pos is None: continue
            
            # Use High/Low for more accurate SL/TP simulation
            high_price = self.env.high_arrays[asset][step]
            low_price = self.env.low_arrays[asset][step]
            spread = self.spreads.get(asset, 0.0)
            
            if pos['direction'] == 1: # Long
                # SL hit if Bid Low <= SL
                if low_price <= pos['sl']:
                    self.env._close_position(asset, pos['sl'])
                # TP hit if Bid High >= TP
                elif high_price >= pos['tp']:
                    self.env._close_position(asset, pos['tp'])
            else: # Short
                # SL hit if Ask High >= SL => Bid High + Spread >= SL
                if high_price + spread >= pos['sl']:
                    self.env._close_position(asset, pos['sl'])
                # TP hit if Ask Low <= TP => Bid Low + Spread <= TP
                elif low_price + spread <= pos['tp']:
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
    
    # Determine if we should use spreads
    use_spreads = not args.no_spreads

    # Create combined backtest using the SHARED environment
    backtest = CombinedBacktest(
        alpha_model, 
        risk_model, 
        data_dir_path, 
        initial_equity=initial_equity,
        alpha_norm_env=alpha_norm_env,
        risk_norm_env=risk_norm_env,
        env=shared_env,
        use_spreads=use_spreads
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
                        help="Disable spreads for testing (default: False, spreads are ON)")
    
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



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
from RiskLayer.src.risk_env import RiskManagementEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

initial_equity=10000.0

class CombinedBacktest:
    """Combined backtest using Alpha model for direction and Risk model for SL/TP/sizing"""
    
    def __init__(self, alpha_model, risk_model, data_dir, initial_equity=initial_equity, alpha_norm_env=None, risk_norm_env=None, env=None):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.alpha_norm_env = alpha_norm_env
        self.risk_norm_env = risk_norm_env
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        
        # Create Alpha environment for data access (or reuse existing)
        if env is not None:
            self.env = env
        else:
            self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
            
        self.env.equity = initial_equity
        self.env.start_equity = initial_equity
        self.env.peak_equity = initial_equity
        
        # Risk model constants (from RiskManagementEnv)
        self.MAX_RISK_PER_TRADE = 0.40 
        self.MAX_MARGIN_PER_TRADE_PCT = 0.80
        self.MAX_LEVERAGE = 400.0
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000
        
        # Slippage Configuration (Match RiskLayer logic)
        self.ENABLE_SLIPPAGE = True
        self.SLIPPAGE_MIN_PIPS = 0.5
        self.SLIPPAGE_MAX_PIPS = 1.5
        
        # SPREAD CONFIGURATION (Dynamic, matching risk_env.py)
        # RiskEnv: SPREAD_MIN_PIPS + (SPREAD_ATR_FACTOR * ATR)
        self.SPREAD_MIN_PIPS = 0.5
        self.SPREAD_ATR_FACTOR = 0.05
        
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

        self.SYMBOL_IDS = {
            'EURUSD': 1,
            'GBPUSD': 2,
            'XAUUSD': 41,
            'USDCHF': 6,
            'USDJPY': 4
        }

    def calculate_dynamic_spread(self, price, atr):
        """Calculate dynamic spread matching RiskEnv logic"""
        # Determine pip size (0.01 for JPY/large prices, 0.0001 for others)
        pip_size = 0.01 if price > 20.0 else 0.0001
        
        # Calculate full spread in price units
        full_spread_price = (self.SPREAD_MIN_PIPS * pip_size) + (self.SPREAD_ATR_FACTOR * atr)
        return full_spread_price
        
    def _strict_update_positions(self):
        """
        Check SL/TP with strict Spread application (Exit Spread).
        Matches RiskEnv: Hit SL if Mid <= SL + HalfSpread (Long)
        """
        current_prices = self.env._get_current_prices()
        current_atrs = self.env._get_current_atrs()
        
        # Iterate over copy
        for asset, pos in list(self.env.positions.items()):
            if pos is None:
                continue
            
            price = current_prices[asset]
            atr = current_atrs[asset]
            
            # Calculate Dynamic Spread for Exit
            full_spread = self.calculate_dynamic_spread(price, atr)
            half_spread = full_spread / 2.0
            
            # Check SL/TP (Trigger on Bid/Ask, not Mid)
            if pos['direction'] == 1:  # Long
                # Exit Sell at Bid = Mid - HalfSpread.
                # Trigger if Bid <= SL.
                # Mid - HalfSpread <= SL  =>  Mid <= SL + HalfSpread
                if price <= (pos['sl'] + half_spread):
                    self.env._close_position(asset, pos['sl'])
                
                # Trigger if Bid >= TP ?? No, Limit order (TP) usually sells at Ask? 
                # Or Bid? Long TP is Sell. Sell at Bid.
                # RiskEnv implies TP is hard to hit too?
                # RiskEnv: Hit TP if MaxFav >= TP_Dist + Spread.
                # i.e. Price must go HIGHER by spread to cover valid exit?
                # Actually, standard MT4: TP for Long is Sell Limit. Executed at Bid.
                # Values:
                # Bid = Mid - 0.5*S.
                # If Bid >= TP. Mid - 0.5*S >= TP. Mid >= TP + 0.5*S.
                elif price >= (pos['tp'] + half_spread):
                    self.env._close_position(asset, pos['tp'])
                    
            else:  # Short
                # Exit Buy at Ask = Mid + HalfSpread.
                # Trigger if Ask >= SL.
                # Mid + HalfSpread >= SL  =>  Mid >= SL - HalfSpread
                if price >= (pos['sl'] - half_spread):
                    self.env._close_position(asset, pos['sl'])
                
                # Trigger if Ask <= TP.
                # Mid + HalfSpread <= TP  =>  Mid <= TP - HalfSpread
                elif price <= (pos['tp'] - half_spread):
                    self.env._close_position(asset, pos['tp'])
        
    def build_risk_observation(self, asset, alpha_obs):
        """Build 45-feature observation for risk model"""
        # Alpha features are already the 40 features [25 asset + 15 global]
        
        # Account state (5 features)
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        equity_norm = self.equity / self.initial_equity
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        # Get current market data for spread/symbol calc
        current_prices = self.env._get_current_prices()
        current_atrs = self.env._get_current_atrs()
        price = current_prices[asset]
        atr = current_atrs[asset]
        
        # Calculate Spread Ratio
        full_spread = self.calculate_dynamic_spread(price, atr)
        spread_ratio = full_spread / atr if atr > 1e-9 else 0.0
        
        # Get Symbol ID
        symbol_id = self.SYMBOL_IDS.get(asset, 0.0)
        
        account_obs = np.array([
            equity_norm,
            drawdown,
            float(symbol_id),  # Symbol ID
            risk_cap_mult,
            spread_ratio   # Spread Ratio
        ], dtype=np.float32)
        
        # Combine all features (40 Market + 5 Account = 45)
        obs = np.concatenate([alpha_obs, account_obs])
        
        # Safety check
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Normalize if normalizer is available
        if self.risk_norm_env is not None:
            obs_reshaped = obs.reshape(1, -1)
            obs_norm = self.risk_norm_env.normalize_obs(obs_reshaped)
            obs = obs_norm.flatten()
        
        return obs
    
    def parse_risk_action(self, action):
        """Parse risk model action to SL/TP/sizing"""
        # Parse action (2 values in [-1, 1])
        # [SL_Mult, TP_Mult]
        sl_mult = np.clip((action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)  # 0.2-2.0 ATR
        tp_mult = np.clip((action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)  # 0.5-4.0 ATR
        
        # Fixed Risk (7%) - matching training logic
        risk_raw = 0.07
        
        return sl_mult, tp_mult, risk_raw
    
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
            
            while not done:
                # Check explicit step limit
                if max_steps is not None and step_count >= max_steps:
                    logger.info(f"Reached max steps limit: {max_steps}")
                    break

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
                        
                    # Build and predict with Risk model
                    risk_obs = self.build_risk_observation(asset, alpha_obs)
                    risk_action, _ = self.risk_model.predict(risk_obs, deterministic=True)
                    sl_mult, tp_mult, risk_raw = self.parse_risk_action(risk_action)
                    
                    # Calculate size
                    current_prices = self.env._get_current_prices()
                    atrs = self.env._get_current_atrs()
                    entry_price = current_prices[asset]
                    atr = atrs[asset]
                    
                    size_pct, risk_info = self.calculate_position_size(
                        asset, entry_price, atr, sl_mult, tp_mult, risk_raw, direction
                    )
                    
                    if size_pct > 0:
                        combined_actions[asset] = {
                            'direction': direction,
                            'size': size_pct,
                            'sl_mult': sl_mult,
                            'tp_mult': tp_mult
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
                        # Slippage + Spread (Adverse movement)
                        if self.ENABLE_SLIPPAGE:
                            slippage_pips = np.random.uniform(self.SLIPPAGE_MIN_PIPS, self.SLIPPAGE_MAX_PIPS)
                            
                            # FULL DYNAMIC SPREAD
                            full_spread_price = self.calculate_dynamic_spread(price_raw, atr)
                            half_spread_price = full_spread_price / 2.0
                            
                            # Friction = Slippage + Half Spread
                            if self.ENABLE_SLIPPAGE:
                                pip_val = 0.01 if price_raw > 20 else 0.0001
                                slippage_price = slippage_pips * pip_val
                            else:
                                slippage_price = 0.0
                                
                            friction_price = slippage_price + half_spread_price
                            price = price_raw + (direction * friction_price)
                        else:
                            # Still pay spread
                            full_spread_price = self.calculate_dynamic_spread(price_raw, atr)
                            price = price_raw + (direction * (full_spread_price / 2.0))
                            
                        if current_pos is None:
                            self.env._open_position(asset, direction, act, price, atr)
                        elif current_pos['direction'] != direction:
                            self.env._close_position(asset, price)
                            self.env._open_position(asset, direction, act, price, atr)
                    else:
                        # No action for this asset: close existing position if any
                        if current_pos is not None:
                            # Apply exit friction (slippage + spread)
                            slippage_pips = np.random.uniform(self.SLIPPAGE_MIN_PIPS, self.SLIPPAGE_MAX_PIPS)
                            
                            full_spread_price = self.calculate_dynamic_spread(price_raw, atr)
                            half_spread_price = full_spread_price / 2.0
                            
                            slippage_price = slippage_pips * (0.01 if price_raw > 20 else 0.0001) if self.ENABLE_SLIPPAGE else 0.0
                            friction_price = slippage_price + half_spread_price
                            
                            # Market close means trading against the current position
                            exit_direction = -current_pos['direction']
                            exit_price = price_raw + (exit_direction * friction_price)
                            self.env._close_position(asset, exit_price)
                
                # Advance time and update
                self.env.current_step += 1
                
                # USE STRICT UPDATE (Accounts for Exit Spread)
                self._strict_update_positions()
                
                
                # Update peak equity and check termination
                self.env.peak_equity = max(self.env.peak_equity, self.env.equity)
                self.equity = self.env.equity
                self.peak_equity = self.env.peak_equity
                
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
    alpha_norm_path = str(alpha_model_path).replace('.zip', '_vecnormalize.pkl')
    # FALLBACK: Handle ppo_final_model.zip -> ppo_final_vecnormalize.pkl
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
    logger.info("Alpha model loaded successfully")
    
    # Load Risk normalizer
    risk_norm_path = risk_model_path.parent / "vec_normalize.pkl"
    if not risk_norm_path.exists():
        risk_norm_path = Path(str(risk_model_path).replace('.zip', '_vecnormalize.pkl'))
    
    # NEW FALLBACK: handle 10M.zip -> 10M.pkl
    if not risk_norm_path.exists():
        risk_norm_path = Path(str(risk_model_path).replace('.zip', '.pkl'))
        
    risk_norm_env = None
    
    # Function to create a valid dummy Risk dataset (small, for loading only)
    def create_dummy_risk_dataset():
        import tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
        dummy_data = pd.DataFrame({
            'direction': [1.0] * 10,
            'entry_price': [1.0] * 10,
            'atr': [0.01] * 10,
            'max_profit_pct': [0.02] * 10,
            'max_loss_pct': [-0.01] * 10,
            'close_1000_price': [1.01] * 10,
            'features': [np.zeros(40, dtype=np.float32) for _ in range(10)],
            'pair': ['EURUSD'] * 10
        })
        dummy_data.to_parquet(tmp.name)
        return tmp.name

    if risk_norm_path.exists():
        logger.info(f"Loading Risk Normalizer from {risk_norm_path}")
        try:
             dummy_path = create_dummy_risk_dataset()
             dummy_risk_env = RiskManagementEnv(dataset_path=dummy_path)
             risk_norm_env = VecNormalize.load(str(risk_norm_path), DummyVecEnv([lambda: dummy_risk_env]))
             risk_norm_env.training = False 
             risk_norm_env.norm_reward = False
             os.unlink(dummy_path)
             logger.info("Risk Normalizer loaded.")
        except Exception as e:
             logger.error(f"Failed to load Risk Normalizer: {e}")
    
    # Load Risk model
    logger.info("Loading Risk model...")
    try:
        dummy_path = create_dummy_risk_dataset()
        risk_env_dummy = DummyVecEnv([lambda: RiskManagementEnv(dataset_path=dummy_path)])
        risk_model = PPO.load(risk_model_path, env=risk_env_dummy)
        os.unlink(dummy_path)
        logger.info("Risk model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Risk model: {e}")
        sys.exit(1)
    
    # Create combined backtest using the SHARED environment
    backtest = CombinedBacktest(
        alpha_model, 
        risk_model, 
        data_dir_path, 
        initial_equity=initial_equity,
        alpha_norm_env=alpha_norm_env,
        risk_norm_env=risk_norm_env,
        env=shared_env
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


import os
import sys
import json
import logging
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add numpy 1.x/2.x compatibility shim for SB3 model loading
if not hasattr(np, "_core"):
    import sys
    sys.modules["numpy._core"] = np.core

from stable_baselines3 import PPO
from Alpha.backtest.backtest_combined import CombinedBacktest
from Alpha.backtest.backtest import BacktestMetrics, FullSystemMetrics, generate_full_system_charts
from Alpha.backtest.tradeguard_feature_builder import TradeGuardFeatureBuilder

logger = logging.getLogger(__name__)

def load_tradeguard_model(model_path, metadata_path):
    """
    Loads the TradeGuard model and its metadata.
    Fails fast if either is missing or invalid.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TradeGuard model not found at {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"TradeGuard metadata not found at {metadata_path}")
    
    try:
        model = lgb.Booster(model_file=model_path)
    except Exception as e:
        raise ValueError(f"Failed to load TradeGuard model: {e}")
        
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load TradeGuard metadata: {e}")
        
    if 'threshold' not in metadata:
        raise KeyError("Metadata must contain 'threshold'")
        
    return model, metadata

class FullSystemBacktest(CombinedBacktest):
    """
    Extends CombinedBacktest to include TradeGuard filtering.
    """
    def __init__(self, alpha_model_path, risk_model_path, guard_model_path, guard_metadata_path, data_dir, initial_equity=10):
        # Load models
        alpha_model = PPO.load(alpha_model_path)
        risk_model = PPO.load(risk_model_path)
        guard_model, guard_metadata = load_tradeguard_model(guard_model_path, guard_metadata_path)
        
        super().__init__(alpha_model, risk_model, data_dir, initial_equity)
        
        self.guard_model = guard_model
        self.guard_threshold = guard_metadata['threshold']
        self.guard_metadata = guard_metadata
        
        # Initialize feature builder (requires data from env)
        # TradingEnv stores the dictionary of dataframes in self.env.data
        self.feature_builder = TradeGuardFeatureBuilder(self.env.data)
        
        # Tracking for Shadow Portfolio and blocked trades
        self.shadow_equity = initial_equity
        self.shadow_peak_equity = initial_equity
        self.blocked_trades = []
        
        # Initialize last_alpha_actions (needed for Group A features)
        self.last_alpha_actions = np.zeros(self.env.action_dim, dtype=np.float32)
        
        logger.info(f"Initialized FullSystemBacktest with TradeGuard threshold: {self.guard_threshold}")

    def evaluate_tradeguard(self, asset, action_info):
        """
        Calculates TradeGuard probability and applies threshold filtering.
        """
        # 1. Build portfolio state
        total_exposure = sum(pos['size'] for pos in self.env.positions.values() if pos is not None)
        
        # Build asset_recent_actions for this asset
        asset_recent_actions = list(self.asset_histories[asset]['action_history'])
        # Map to raw action values (first component for direction)
        asset_recent_actions_raw = [a[0] for a in asset_recent_actions]
        if len(asset_recent_actions_raw) < 5:
            asset_recent_actions_raw = [0.0] * (5 - len(asset_recent_actions_raw)) + asset_recent_actions_raw
            
        persistence, reversal = self._calculate_persistence_reversal(asset, action_info['direction'])
        
        portfolio_state = {
            'equity': self.env.equity,
            'peak_equity': self.env.peak_equity,
            'total_exposure': total_exposure,
            'open_positions_count': sum(1 for p in self.env.positions.values() if p is not None),
            'recent_trades': [{'pnl': t['net_pnl']} for t in self.env.all_trades[-10:]],
            'asset_action_raw': self.last_alpha_actions[self.env.assets.index(asset)] if self.last_alpha_actions is not None else 0.0,
            'asset_recent_actions': asset_recent_actions_raw,
            'asset_signal_persistence': persistence,
            'asset_signal_reversal': reversal,
            'position_value': self.env.equity * action_info['size'] * 0.5 # Approximate
        }
        
        trade_info = {
            'entry_price': self.env._get_current_prices()[asset],
            'sl': 0, # Will be filled below if needed
            'tp': 0,
            'direction': action_info['direction']
        }
        
        # We need SL/TP to build features (Group E)
        atr = self.env._get_current_atrs()[asset]
        atr_val = max(atr, trade_info['entry_price'] * self.env.MIN_ATR_MULTIPLIER)
        trade_info['sl'] = trade_info['entry_price'] - (action_info['direction'] * action_info['sl_mult'] * atr_val)
        trade_info['tp'] = trade_info['entry_price'] + (action_info['direction'] * action_info['tp_mult'] * atr_val)

        # 2. Build feature vector
        features = self.feature_builder.build_features(asset, self.env.current_step, trade_info, portfolio_state)
        
        # 3. Predict
        prob = self.guard_model.predict(np.array([features]))[0]
        
        return prob >= self.guard_threshold, prob

    def _calculate_persistence_reversal(self, asset, current_direction):
        """Helper to match generate_dataset logic"""
        history = list(self.asset_histories[asset]['action_history'])
        # Directions are sign(action[0])
        sig_history = [np.sign(a[0]) if abs(a[0]) > 0.33 else 0 for a in history]
        
        if len(sig_history) < 1:
            return 1.0, 0.0
            
        count = 0
        for sig in reversed(sig_history):
            if sig == current_direction:
                count += 1
            else:
                break
        persistence = count + 1.0
        
        prev_signal = 0
        for sig in reversed(sig_history):
            if sig != 0:
                prev_signal = sig
                break
        
        reversal = 1.0 if prev_signal != 0 and prev_signal != current_direction else 0.0
        return persistence, reversal

    def run_backtest(self, episodes=1):
        """
        Full integrated backtest loop with Alpha, Risk, and TradeGuard.
        Tracks both Full System and Shadow Portfolio (Baseline).
        """
        metrics_tracker = FullSystemMetrics()
        
        logger.info(f"Running {episodes} episodes with Full System...")
        
        for episode in range(episodes):
            obs, _ = self.env.reset()
            done = False
            step_count = 0
            
            # Reset state
            self.equity = self.initial_equity
            self.peak_equity = self.initial_equity
            self.env.equity = self.initial_equity
            self.env.peak_equity = self.initial_equity
            
            # Reset Shadow Portfolio
            self.shadow_equity = self.initial_equity
            self.shadow_peak_equity = self.initial_equity
            self.active_virtual_trades = [] # Trades that are active in shadow but blocked in real
            
            # Reset histories
            for asset in self.env.assets:
                self.asset_histories[asset]['pnl_history'] = deque([0.0] * 5, maxlen=5)
                self.asset_histories[asset]['action_history'] = deque([np.zeros(3, dtype=np.float32) for _ in range(5)], maxlen=5)
            
            while not done:
                # 1. Alpha Prediction
                alpha_action, _ = self.alpha_model.predict(obs, deterministic=True)
                self.last_alpha_actions = alpha_action
                
                # Parse Alpha action to get directions (Stage 1: 5 outputs)
                directions = {}
                for i, asset in enumerate(self.env.assets):
                    direction_raw = alpha_action[i]
                    direction = 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0)
                    directions[asset] = direction
                
                combined_actions = {}
                
                # 2. Risk Evaluation for each asset
                for asset, direction in directions.items():
                    if direction == 0:
                        continue
                    
                    # Build risk model observation
                    risk_obs = self.build_risk_observation(asset)
                    
                    # Get risk model prediction
                    risk_action, _ = self.risk_model.predict(risk_obs, deterministic=True)
                    
                    # Parse risk action
                    sl_mult, tp_mult, risk_raw = self.parse_risk_action(risk_action)
                    
                    # Risk-level Blocking Logic (Threshold: 0.20)
                    if risk_raw < 0.20:
                        continue

                    # If NOT blocked by Risk, Rescale Risk
                    risk_raw_scaled = (risk_raw - 0.20) / 0.80
                    
                    # Get market data for position sizing
                    current_prices = self.env._get_current_prices()
                    atrs = self.env._get_current_atrs()
                    entry_price = current_prices[asset]
                    atr = atrs[asset]
                    
                    # Calculate position size
                    size_pct, risk_info = self.calculate_position_size(
                        asset, entry_price, atr, sl_mult, tp_mult, risk_raw_scaled, direction
                    )
                    
                    if size_pct > 0 and risk_info:
                        # Prepare action info for TradeGuard
                        act_info = {
                            'direction': direction,
                            'size': size_pct,
                            'sl_mult': sl_mult,
                            'tp_mult': tp_mult,
                            'risk_raw': risk_raw_scaled
                        }
                        
                        # 3. TradeGuard Evaluation
                        is_approved, prob = self.evaluate_tradeguard(asset, act_info)
                        
                        if is_approved:
                            # Store action in history
                            self.asset_histories[asset]['action_history'].append(
                                np.array([sl_mult, tp_mult, risk_raw_scaled], dtype=np.float32)
                            )
                            
                            combined_actions[asset] = act_info
                        else:
                            # BLOCKED by TradeGuard
                            blocked_record = self._simulate_blocked_trade(asset, direction, act_info, prob)
                            metrics_tracker.add_blocked_trade(blocked_record)
                            
                            # Update Shadow Portfolio (compounding PnL)
                            # theoretical_pnl is a ratio (e.g. 0.01 for 1%)
                            self.shadow_equity *= (1 + blocked_record['theoretical_pnl'])
                
                # 4. Execute approved trades in environment
                prev_real_equity = self.env.equity
                self._execute_approved_trades(combined_actions)
                
                # 5. Advance Environment
                self.env.current_step += 1
                self.env._update_positions()
                
                # Capture Results
                info = {
                    'trades': self.env.completed_trades,
                    'equity': self.env.equity,
                    'timestamp': self.env._get_current_timestamp()
                }
                
                # 6. Synchronize Shadow Portfolio for approved trades
                # The shadow portfolio also executed these trades.
                # We apply the same PnL ratio as seen in the real portfolio.
                real_pnl_ratio = (self.env.equity - prev_real_equity) / prev_real_equity if prev_real_equity > 0 else 0
                self.shadow_equity *= (1 + real_pnl_ratio)
                
                self.shadow_peak_equity = max(self.shadow_peak_equity, self.shadow_equity)
                
                # Advance tracking
                obs = self.env._validate_observation(self.env._get_observation())
                done = self.env.current_step >= self.env.max_steps
                
                self.equity = self.env.equity
                self.peak_equity = max(self.peak_equity, self.equity)
                
                # Update trackers
                if info['trades']:
                    for trade in info['trades']:
                        asset = trade['asset']
                        pnl_ratio = trade.get('net_pnl', 0) / max(self.initial_equity, 1e-6)
                        self.asset_histories[asset]['pnl_history'].append(pnl_ratio)
                        metrics_tracker.add_trade(trade)
                
                metrics_tracker.add_equity_point(info['timestamp'], info['equity'])
                metrics_tracker.add_shadow_equity_point(info['timestamp'], self.shadow_equity)
                
                step_count += 1
                
                if step_count % 1000 == 0:
                    logger.info(f"Step {step_count}, Equity: ${self.equity:.2f}, Shadow: ${self.shadow_equity:.2f}")
                    
            logger.info(f"Episode {episode + 1}/{episodes} complete. Steps: {step_count}, Final Equity: ${self.equity:.2f}, Shadow: ${self.shadow_equity:.2f}")
            
        return metrics_tracker


    def _execute_approved_trades(self, combined_actions):
        """Helper to execute approved trades and close others"""
        current_prices = self.env._get_current_prices()
        atrs = self.env._get_current_atrs()
        
        for asset, act in combined_actions.items():
            direction = act['direction']
            current_pos = self.env.positions[asset]
            price_raw = current_prices[asset]
            atr = atrs[asset]
            
            # Slippage
            if self.ENABLE_SLIPPAGE:
                slippage_pips = np.random.uniform(self.SLIPPAGE_MIN_PIPS, self.SLIPPAGE_MAX_PIPS)
                slippage_price = slippage_pips * 0.0001 * price_raw
                price = price_raw + (direction * -1 * slippage_price)
            else:
                price = price_raw
            
            if current_pos is None:
                if direction != 0:
                    self.env._open_position(asset, direction, act, price, atr)
            elif current_pos['direction'] != direction:
                self.env._close_position(asset, price)
                if direction != 0:
                    self.env._open_position(asset, direction, act, price, atr)
                    
        # Close positions for flat signals
        for asset in self.env.assets:
            if asset not in combined_actions and self.env.positions[asset] is not None:
                self.env._close_position(asset, current_prices[asset])

    def _simulate_blocked_trade(self, asset, direction, act, prob):
        """
        Simulates the outcome of a trade that was blocked by TradeGuard.
        Uses the environment's peek-ahead simulation logic.
        """
        # Save current position to restore it after simulation
        real_pos = self.env.positions[asset]
        
        # Calculate SL/TP for virtual labeling
        price = self.env._get_current_prices()[asset]
        atr = self.env._get_current_atrs()[asset]
        atr_val = max(atr, price * self.env.MIN_ATR_MULTIPLIER)
        
        sl_dist = act['sl_mult'] * atr_val
        tp_dist = act['tp_mult'] * atr_val
        
        # Create virtual position for simulation
        self.env.positions[asset] = {
            'direction': direction,
            'entry_price': price,
            'size': 1.0, 
            'sl': price - (direction * sl_dist),
            'tp': price + (direction * tp_dist),
            'entry_step': self.env.current_step,
            'sl_dist': sl_dist,
            'tp_dist': tp_dist
        }
        
        try:
            # Simulate outcome using peek-ahead
            outcome = self.env._simulate_trade_outcome_with_timing(asset)
            
            record = {
                'timestamp': self.env._get_current_timestamp(),
                'asset': asset,
                'direction': direction,
                'prob': prob,
                'threshold': self.guard_threshold,
                'theoretical_pnl': outcome['pnl'],
                'outcome': outcome.get('reason', 'unknown'),
                'exit_step': outcome.get('exit_step', self.env.current_step),
                'sl': self.env.positions[asset]['sl'],
                'tp': self.env.positions[asset]['tp'],
                'entry_price': price
            }
            self.blocked_trades.append(record)
            return record
        finally:
            # Restore the actual environment position
            self.env.positions[asset] = real_pos

if __name__ == "__main__":
    pass
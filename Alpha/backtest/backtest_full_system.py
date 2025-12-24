import os
import sys
import json
import logging
from pathlib import Path
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
from stable_baselines3.common.vec_env import DummyVecEnv
from Alpha.backtest.backtest_combined import CombinedBacktest
from Alpha.backtest.backtest import BacktestMetrics, FullSystemMetrics, generate_full_system_charts
from Alpha.backtest.tradeguard_feature_builder import TradeGuardFeatureBuilder

logger = logging.getLogger(__name__)

class FullSystemBacktest(CombinedBacktest):
    """
    Extends CombinedBacktest to include TradeGuard filtering (RL Version).
    """
    def __init__(self, alpha_model_path, risk_model_path, guard_model_path, data_dir, initial_equity=10):
        # Load models
        alpha_model = PPO.load(alpha_model_path)
        risk_model = PPO.load(risk_model_path)
        
        # Load TradeGuard Model (PPO)
        if not os.path.exists(guard_model_path):
             raise FileNotFoundError(f"TradeGuard model not found at {guard_model_path}")
        
        # We need to construct a dummy env for PPO load if using custom features, 
        # but usually PPO.load works fine without env for inference if observation space matches.
        # However, to be safe, we can wrap it.
        try:
             guard_model = PPO.load(guard_model_path)
        except Exception as e:
             raise ValueError(f"Failed to load TradeGuard PPO model: {e}")

        super().__init__(alpha_model, risk_model, data_dir, initial_equity)
        
        self.guard_model = guard_model
        
        # Initialize feature builder (requires data from env)
        # TradingEnv stores the dictionary of dataframes in self.env.data
        self.feature_builder = TradeGuardFeatureBuilder(self.env.data)
        
        # Tracking for Shadow Portfolio and blocked trades
        self.shadow_equity = initial_equity
        self.shadow_peak_equity = initial_equity
        self.blocked_trades = []
        
        # Initialize last_alpha_actions (needed for Group A features)
        self.last_alpha_actions = np.zeros(self.env.action_dim, dtype=np.float32)
        
        logger.info(f"Initialized FullSystemBacktest with TradeGuard RL Model")

    def _get_portfolio_state_for_features(self, alpha_actions_parsed):
        """
        Constructs the portfolio state dictionary required by TradeGuardFeatureBuilder.
        """
        state = {
            'total_drawdown': 1.0 - (self.env.equity / self.env.peak_equity),
            'total_exposure': sum(p['size'] for p in self.env.positions.values() if p is not None) / (self.env.equity + 1e-6)
        }
        
        for asset in self.env.assets:
            # Calculate persistence/reversal
            history = list(self.asset_histories[asset]['action_history'])
            # Directions are sign(action[0]) - wait, action_history stores [sl, tp, risk] in CombinedBacktest
            # We need directional history. 
            # In CombinedBacktest, action_history stores the RISK actions.
            # We need to track ALPHA directional history separately if we want persistence.
            # Let's derive it from recent trades or use a new tracker.
            
            # For now, let's use the current direction from alpha_actions_parsed
            current_dir = alpha_actions_parsed[asset]
            
            # We don't have a dedicated direction history in CombinedBacktest base class easily accessible 
            # in the format we want (it has pnl_history).
            # Let's implement a simple local tracker for direction history in this class.
            # See self.direction_histories initialized in run_backtest
            
            hist = self.direction_histories[asset]
            
            persistence = 0
            if hist:
                 for a in reversed(hist):
                      if a == current_dir and a != 0:
                           persistence += 1
                      else:
                           break
            
            reversal = 1 if (len(hist) > 0 and hist[-1] != current_dir and current_dir != 0) else 0

            state[asset] = {
                'action_raw': current_dir, # Using integer direction as raw proxy
                'signal_persistence': persistence,
                'signal_reversal': reversal
            }
            
        return state

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
            self.active_virtual_trades = [] 
            
            # Reset histories
            self.direction_histories = {asset: [] for asset in self.env.assets}
            for asset in self.env.assets:
                self.asset_histories[asset]['pnl_history'] = deque([0.0] * 5, maxlen=5)
                self.asset_histories[asset]['action_history'] = deque([np.zeros(3, dtype=np.float32) for _ in range(5)], maxlen=5)
            
            while not done:
                # Clear completed trades from previous step to avoid duplication
                self.env.completed_trades = []
                
                # 1. Alpha Prediction
                alpha_action, _ = self.alpha_model.predict(obs, deterministic=True)
                self.last_alpha_actions = alpha_action
                
                # Parse Alpha action to get directions
                directions = {}
                for i, asset in enumerate(self.env.assets):
                    direction_raw = alpha_action[i]
                    direction = 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0)
                    directions[asset] = direction
                
                # 2. Risk Evaluation for each asset
                pending_new_trades = {}
                continuing_trades = {}
                
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
                    
                    # Check for valid trade
                    if size_pct > 0 and risk_info:
                        # Prepare action info
                        act_info = {
                            'direction': direction,
                            'size': size_pct,
                            'sl_mult': sl_mult,
                            'tp_mult': tp_mult,
                            'risk_raw': risk_raw_scaled,
                            # Additional info for Feature Builder
                            'entry': entry_price,
                            'sl': entry_price - (direction * sl_mult * atr),
                            'tp': entry_price + (direction * tp_mult * atr)
                        }
                        
                        # Entry Filter Logic: Only new entries or reversals go through TradeGuard
                        current_pos = self.env.positions[asset]
                        if current_pos is not None and current_pos['direction'] == direction:
                            continuing_trades[asset] = act_info
                        else:
                            pending_new_trades[asset] = act_info

                # Initialize combined_actions with continuing trades (allowed by default to prevent churn)
                combined_actions = continuing_trades.copy()
                for asset, act_info in continuing_trades.items():
                    self.asset_histories[asset]['action_history'].append(
                        np.array([act_info['sl_mult'], act_info['tp_mult'], act_info['risk_raw']], dtype=np.float32)
                    )
                
                # 3. TradeGuard Global Evaluation (Only for NEW or REVERSAL entries)
                if pending_new_trades:
                    # Build Portfolio State
                    portfolio_state = self._get_portfolio_state_for_features(directions)
                    
                    # Use all trades (continuing + new) for guard context, but only block 'new' ones
                    evaluation_set = {**continuing_trades, **pending_new_trades}
                    
                    features = self.feature_builder.get_multi_asset_obs(
                        self.env.current_step, 
                        evaluation_set, 
                        portfolio_state
                    )
                    
                    # Predict (Allow/Block)
                    # Action 1 = Allow, 0 = Block
                    guard_action, _ = self.guard_model.predict(features, deterministic=True)
                    
                    # Probability (Optional, if supported by policy)
                    # For PPO, we can get distribution, but let's stick to deterministic action for now.
                    prob = 1.0 if guard_action == 1 else 0.0 
                    
                    if guard_action == 1:
                        # APPROVED: Add all pending new trades to combined_actions
                        for asset, act_info in pending_new_trades.items():
                            self.asset_histories[asset]['action_history'].append(
                                np.array([act_info['sl_mult'], act_info['tp_mult'], act_info['risk_raw']], dtype=np.float32)
                            )
                            combined_actions[asset] = act_info
                    else:
                        # BLOCKED: Simulate all pending new trades for metrics
                        for asset, act_info in pending_new_trades.items():
                            blocked_record = self._simulate_blocked_trade(asset, act_info['direction'], act_info, prob)
                            metrics_tracker.add_blocked_trade(blocked_record)
                            
                            # Update Shadow Portfolio (compounding PnL for baseline)
                            self.shadow_equity *= (1 + blocked_record['theoretical_pnl'])

                # Update Direction History (for persistence features)
                for asset in self.env.assets:
                    self.direction_histories[asset].append(directions[asset])
                    if len(self.direction_histories[asset]) > 20:
                        self.direction_histories[asset].pop(0)

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
                    
        # Close positions for flat signals OR if not in combined_actions (closed by Guard or Alpha)
        # Note: If Alpha says trade, but Guard says Block, combined_actions won't have it.
        # But if we had a position, and Alpha says trade (same dir), we should HOLD?
        # Logic check:
        # If Alpha says direction X.
        # Guard says BLOCK.
        # We do NOT trade.
        # If we already had a position in direction X:
        # Should we close it? Or hold it?
        # Standard logic: If signal is blocked, we treat it as "No Trade".
        # If "No Trade" (Alpha says 0), we close.
        # So if Blocked, we should probably close or at least not open new.
        # The current implementation of _execute_approved_trades iterates combined_actions (Approved).
        # Any asset NOT in combined_actions is handled in the next loop:
        
        for asset in self.env.assets:
            # If asset is not in approved actions
            if asset not in combined_actions:
                # If we have a position, CLOSE IT.
                # This implies that if Guard blocks a "continuation" signal, we close the position.
                # This is a safe "Guard" behavior (if confidence drops, exit).
                if self.env.positions[asset] is not None:
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
        
        sl_dist = act['sl_mult'] * self.env._get_current_atrs()[asset]
        tp_dist = act['tp_mult'] * self.env._get_current_atrs()[asset]
        
        # Calculate ACTUAL intended size pct and notional size
        size_pct = act['size'] * self.env.MAX_POS_SIZE_PCT
        position_size = size_pct * self.env.equity
        
        # Create virtual position for simulation
        self.env.positions[asset] = {
            'direction': direction,
            'entry_price': price,
            'size': position_size, 
            'sl': act['sl'], # Already calculated
            'tp': act['tp'], # Already calculated
            'entry_step': self.env.current_step,
            'sl_dist': sl_dist,
            'tp_dist': tp_dist
        }
        
        try:
            # Simulate outcome using peek-ahead
            outcome = self.env._simulate_trade_outcome_with_timing(asset)
            
            # Calculate Transaction Costs (approximate round trip)
            # Matching TradingEnv costs: 0.00002 per side on notional size
            entry_cost = position_size * 0.00002
            # Use entry size for exit cost approximation
            exit_cost = position_size * 0.00002 
            
            net_pnl = outcome['pnl'] - entry_cost - exit_cost
            
            # Calculate PnL as a ratio of current equity for compounding shadow portfolio
            theoretical_pnl_ratio = net_pnl / self.env.equity if self.env.equity > 0 else 0
            
            record = {
                'timestamp': self.env._get_current_timestamp(),
                'asset': asset,
                'direction': direction,
                'prob': prob,
                'threshold': 0.5, # Implicit binary threshold
                'theoretical_pnl': theoretical_pnl_ratio,
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def run_full_system_backtest(args):
    """Main function to run the full three-layer system backtest"""
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Setup paths
    alpha_path = project_root / args.alpha_model
    risk_path = project_root / args.risk_model
    guard_path = project_root / args.guard_model
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Initializing Full System Backtest...")
    bt = FullSystemBacktest(
        alpha_model_path=str(alpha_path),
        risk_model_path=str(risk_path),
        guard_model_path=str(guard_path),
        data_dir=str(data_dir),
        initial_equity=args.initial_equity
    )
    
    # Run backtest
    metrics_tracker = bt.run_backtest(episodes=args.episodes)
    
    # Calculate and save metrics
    logger.info("Calculating metrics...")
    metrics = metrics_tracker.calculate_metrics()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = output_dir / f"metrics_full_system_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    # Save trades
    if metrics_tracker.trades:
        trades_file = output_dir / f"trades_full_system_{timestamp}.csv"
        pd.DataFrame(metrics_tracker.trades).to_csv(trades_file, index=False)
        
    # Save blocked trades
    if metrics_tracker.blocked_trades:
        blocked_file = output_dir / f"blocked_trades_full_system_{timestamp}.csv"
        pd.DataFrame(metrics_tracker.blocked_trades).to_csv(blocked_file, index=False)
        
    # Generate charts
    logger.info("Generating visualization suite...")
    per_asset = metrics_tracker.get_per_asset_metrics()
    generate_full_system_charts(metrics_tracker, per_asset, 3, output_dir, timestamp)
    
    logger.info(f"Backtest complete. Results saved to {output_dir}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info(f"{ 'FULL SYSTEM BACKTEST RESULTS':^60}")
    logger.info("="*60)
    logger.info(f"{ 'Total Return:':<40} {metrics.get('total_return', 0):.2%}")
    logger.info(f"{ 'Baseline Return:':<40} {metrics.get('baseline_return', 0):.2%}")
    logger.info(f"{ 'Net Value-Add:':<40} {metrics.get('net_value_add_vs_baseline', 0):.2%}")
    logger.info(f"{ 'Max Drawdown:':<40} {metrics.get('max_drawdown', 0):.2%}")
    if 'tradeguard' in metrics:
        tg = metrics['tradeguard']
        logger.info(f"{ 'TradeGuard Approval Rate:':<40} {tg.get('approval_rate', 0):.2%}")
        logger.info(f"{ 'TradeGuard Block Accuracy:':<40} {tg.get('block_accuracy', 0):.2%}")
    logger.info("="*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Three-Layer Full System Backtest")
    parser.add_argument("--alpha-model", type=str, default="Alpha/models/checkpoints/8.03.zip")
    parser.add_argument("--risk-model", type=str, default="RiskLayer/models/2.15.zip")
    parser.add_argument("--guard-model", type=str, default="TradeGuard/models/manual_test_model.zip")
    # guard-meta is no longer required for PPO, but keeping it optional or removing it
    parser.add_argument("--data-dir", type=str, default="Alpha/backtest/data")
    parser.add_argument("--output-dir", type=str, default="Alpha/backtest/results/full_system")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--initial-equity", type=float, default=10.0)
    
    args = parser.parse_args()
    run_full_system_backtest(args)

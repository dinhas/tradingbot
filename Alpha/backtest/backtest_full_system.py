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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Alpha.backtest.backtest_combined import CombinedBacktest
from Alpha.backtest.backtest import BacktestMetrics, FullSystemMetrics, generate_full_system_charts
from TradeGuard.src.feature_calculator import TradeGuardFeatureCalculator
from Alpha.src.trading_env import TradingEnv
import pickle

logger = logging.getLogger(__name__)

class FullSystemBacktest(CombinedBacktest):
    """
    Extends CombinedBacktest to include TradeGuard filtering (RL Version).
    """
    def __init__(self, alpha_model_path, risk_model_path, guard_model_path, data_dir, initial_equity=10.0):
        # 1. Initialize Shared Trading Environment
        shared_env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
        dummy_vec_env = DummyVecEnv([lambda: shared_env])

        # 2. Load Alpha Model & Normalizer
        alpha_norm_path = str(alpha_model_path).replace('.zip', '_vecnormalize.pkl')
        if not os.path.exists(alpha_norm_path):
            alpha_norm_path = str(alpha_model_path).replace('_model.zip', '_vecnormalize.pkl')
            
        alpha_norm_env = None
        if os.path.exists(alpha_norm_path):
            logger.info(f"Loading Alpha Normalizer from {alpha_norm_path}")
            alpha_norm_env = VecNormalize.load(alpha_norm_path, dummy_vec_env)
            alpha_norm_env.training = False
            alpha_norm_env.norm_reward = False
        
        alpha_model = PPO.load(alpha_model_path, env=dummy_vec_env)

        # 3. Load Risk Model & Normalizer
        risk_norm_path = Path(risk_model_path).parent / "vec_normalize.pkl"
        if not risk_norm_path.exists():
            risk_norm_path = Path(str(risk_model_path).replace('.zip', '_vecnormalize.pkl'))
            
        risk_norm_env = None
        
        # Helper to create dummy risk env for loading normalizer/model
        def create_dummy_risk_env():
            from RiskLayer.src.risk_env import RiskManagementEnv
            import tempfile
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
            dummy_data = pd.DataFrame({
                'direction': [1.0] * 10, 'entry_price': [1.0] * 10, 'atr': [0.01] * 10,
                'max_profit_pct': [0.02] * 10, 'max_loss_pct': [-0.01] * 10, 'close_1000_price': [1.01] * 10,
                'features': [np.zeros(40, dtype=np.float32) for _ in range(10)], 'pair': ['EURUSD'] * 10
            })
            dummy_data.to_parquet(tmp.name)
            env = RiskManagementEnv(dataset_path=tmp.name)
            return env, tmp.name

        if risk_norm_path.exists():
            logger.info(f"Loading Risk Normalizer from {risk_norm_path}")
            r_env, tmp_path = create_dummy_risk_env()
            risk_norm_env = VecNormalize.load(str(risk_norm_path), DummyVecEnv([lambda: r_env]))
            risk_norm_env.training = False
            risk_norm_env.norm_reward = False
            os.unlink(tmp_path)

        r_env, tmp_path = create_dummy_risk_env()
        risk_model = PPO.load(risk_model_path, env=DummyVecEnv([lambda: r_env]))
        os.unlink(tmp_path)
        
        # 4. Load TradeGuard Model (PPO)
        if not os.path.exists(guard_model_path):
             raise FileNotFoundError(f"TradeGuard model not found at {guard_model_path}")
        
        try:
             guard_model = PPO.load(guard_model_path)
        except Exception as e:
             raise ValueError(f"Failed to load TradeGuard PPO model: {e}")

        # Initialize base class
        super().__init__(
            alpha_model=alpha_model, 
            risk_model=risk_model, 
            data_dir=data_dir, 
            initial_equity=initial_equity,
            alpha_norm_env=alpha_norm_env,
            risk_norm_env=risk_norm_env,
            env=shared_env
        )
        
        self.guard_model = guard_model
        
        # Initialize feature builder (requires data from env)
        self.feature_builder = TradeGuardFeatureCalculator(self.env.data)
        
        # Load TradeGuard Normalization Stats
        self.guard_norm_stats = None
        norm_path = str(guard_model_path) + "_norm_stats.pkl"
        if os.path.exists(norm_path):
            logger.info(f"Loading TradeGuard Normalization Stats from {norm_path}")
            with open(norm_path, 'rb') as f:
                self.guard_norm_stats = pickle.load(f)
        
        # Tracking for Shadow Portfolio and blocked trades
        self.shadow_equity = initial_equity
        self.shadow_peak_equity = initial_equity
        self.blocked_trades = []
        self.shadow_positions = {asset: None for asset in self.env.assets}
        self.active_blocked_signals = {asset: False for asset in self.env.assets}
        
        # Initialize last_alpha_actions (needed for Group A features)
        self.last_alpha_actions = {} 
        
        logger.info(f"Initialized FullSystemBacktest with TradeGuard RL Model and Fixed $25 Sizing")

    def calculate_position_size(self, asset, entry_price, atr, sl_mult, tp_mult, risk_raw, direction):
        """
        Calculates position size based on the 5% of account rule.
        """
        # User Rule: 5% of account per trade
        position_size = self.equity * 0.05
        
        # Determine contract size
        contract_size = 100 if asset == 'XAUUSD' else 100000
        is_usd_quote = asset in ['EURUSD', 'GBPUSD', 'XAUUSD']
        
        # Determine lot value
        if is_usd_quote:
            lot_value = contract_size * entry_price
        else:
            lot_value = contract_size # USD base pairs like USDJPY
            
        # Convert to size_pct (ratio of MAX_POS_SIZE_PCT * equity)
        MAX_POS_SIZE_PCT = 0.50
        size_pct = position_size / (MAX_POS_SIZE_PCT * self.equity + 1e-9)
        size_pct = np.clip(size_pct, 0.0, 1.0)
        
        return size_pct, {
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'risk_raw': 0.0,
            'lots': position_size / (lot_value / self.MAX_LEVERAGE + 1e-9), # Rough estimation for logging
            'position_size': position_size
        }

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
            self.shadow_positions = {asset: None for asset in self.env.assets}
            self.active_blocked_signals = {asset: False for asset in self.env.assets}
            
            # Reset histories
            self.direction_histories = {asset: [] for asset in self.env.assets}
            for asset in self.env.assets:
                self.asset_histories[asset]['pnl_history'] = deque([0.0] * 5, maxlen=5)
                self.asset_histories[asset]['action_history'] = deque([np.zeros(3, dtype=np.float32) for _ in range(5)], maxlen=5)
            
            while not done:
                # Clear completed trades from previous step to avoid duplication
                self.env.completed_trades = []
                
                # 1. Prediction Loop (Per Asset)
                pending_new_trades = {}
                continuing_trades = {}
                directions = {}

                for asset in self.env.assets:
                    self.env.set_asset(asset)
                    alpha_obs = self.env._get_observation()
                    
                    # Normalize Alpha Obs
                    alpha_pred_obs = alpha_obs
                    if self.alpha_norm_env:
                        alpha_pred_obs = self.alpha_norm_env.normalize_obs(alpha_obs.reshape(1, -1)).flatten()
                    
                    alpha_action, _ = self.alpha_model.predict(alpha_pred_obs, deterministic=True)
                    direction = 1 if alpha_action[0] > 0.33 else (-1 if alpha_action[0] < -0.33 else 0)
                    directions[asset] = direction
                    
                    if direction == 0:
                        continue
                        
                    # 2. Risk Evaluation
                    risk_obs = self.build_risk_observation(asset, alpha_obs)
                    risk_action, _ = self.risk_model.predict(risk_obs, deterministic=True)
                    sl_mult, tp_mult, risk_raw = self.parse_risk_action(risk_action)
                    
                    current_prices = self.env._get_current_prices()
                    atrs = self.env._get_current_atrs()
                    entry_price = current_prices[asset]
                    atr = atrs[asset]
                    
                    # Size calculation (Fixed 5% Rule)
                    size_pct, risk_info = self.calculate_position_size(
                        asset, entry_price, atr, sl_mult, tp_mult, risk_raw, direction
                    )
                    
                    # Calculate realistic entry price for SL/TP placement (Spread + Slippage)
                    if self.ENABLE_SLIPPAGE:
                        slippage_pips = np.random.uniform(self.SLIPPAGE_MIN_PIPS, self.SLIPPAGE_MAX_PIPS)
                        slippage_price = slippage_pips * 0.0001 * entry_price
                        realistic_price = entry_price + (direction * -1 * slippage_price)
                    else:
                        realistic_price = entry_price
                    
                    if self.ENABLE_SPREAD:
                        spread_val = (self.SPREAD_MIN_PIPS * 0.0001) + (self.SPREAD_ATR_FACTOR * atr)
                        realistic_price += direction * spread_val

                    if size_pct > 0:
                        act_info = {
                            'direction': direction,
                            'size': size_pct,
                            'sl_mult': sl_mult,
                            'tp_mult': tp_mult,
                            'risk_raw': risk_raw,
                            'entry': realistic_price,
                            'sl': realistic_price - (direction * sl_mult * atr),
                            'tp': realistic_price + (direction * tp_mult * atr),
                            'risk_val': 0.5 # Default Risk Conviction
                        }
                        
                        # Filter Continuation vs New
                        current_pos = self.env.positions[asset]
                        if current_pos is not None and current_pos['direction'] == direction:
                            continuing_trades[asset] = act_info
                        else:
                            pending_new_trades[asset] = act_info

                # 3. TradeGuard Global Evaluation (Per Pending Trade)
                portfolio_state = self._get_portfolio_state_for_features(directions)
                combined_actions = continuing_trades.copy()
                
                # Approved trades go to combined_actions
                for asset, act_info in pending_new_trades.items():
                    # Build 25 Features for THIS trade
                    # Asset portfolio state:
                    asset_p_state = portfolio_state.get(asset, {})
                    
                    tg_features = self.feature_builder.get_single_asset_obs(
                        asset, 
                        self.env.current_step, 
                        act_info, 
                        asset_p_state, 
                        portfolio_state
                    )
                    
                    # Normalize TradeGuard features
                    if self.guard_norm_stats:
                        mean = self.guard_norm_stats['feature_mean']
                        std = self.guard_norm_stats['feature_std']
                        tg_features = (tg_features - mean) / std
                    
                    # Predict Allow (1) or Block (0)
                    guard_action, _ = self.guard_model.predict(tg_features, deterministic=True)
                    
                    if guard_action == 1:
                        combined_actions[asset] = act_info
                        self.active_blocked_signals[asset] = False
                    else:
                        # Blocked: Record metrics ONLY once per new signal window
                        if not self.active_blocked_signals[asset]:
                            blocked_record = self._simulate_blocked_trade(asset, act_info['direction'], act_info, 0.0)
                            metrics_tracker.add_blocked_trade(blocked_record)
                            
                            # Update Shadow Portfolio Position (Virtual Entry)
                            if self.shadow_positions[asset] is None:
                                self.shadow_positions[asset] = {
                                    'exit_step': blocked_record['exit_step'],
                                    'theoretical_pnl': blocked_record['theoretical_pnl']
                                }
                            
                            self.active_blocked_signals[asset] = True

                # Update Direction History
                for asset in self.env.assets:
                    self.direction_histories[asset].append(directions[asset])
                    if len(self.direction_histories[asset]) > 20: self.direction_histories[asset].pop(0)

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
                
                # 6. Synchronize Shadow Portfolio
                # Approved trades are handled by env, their PnL flows to shadow_equity via real_pnl_ratio
                real_pnl_ratio = (self.env.equity - prev_real_equity) / prev_real_equity if prev_real_equity > 0 else 0
                self.shadow_equity *= (1 + real_pnl_ratio)
                
                # Blocked trades in shadow portfolio realize PnL when their simulated exit_step is reached
                for asset in self.env.assets:
                    s_pos = self.shadow_positions[asset]
                    if s_pos is not None and self.env.current_step >= s_pos['exit_step']:
                        self.shadow_equity *= (1 + s_pos['theoretical_pnl'])
                        self.shadow_positions[asset] = None
                
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
                        
                        # Add spread info for metrics
                        if True: # Always calculate if we are in this system
                            atr = self.env._get_current_atrs()[asset]
                            spread_val = (0.5 * 0.0001) + (0.05 * atr)
                            trade['spread_pips'] = spread_val / 0.0001
                        
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
            
            if self.ENABLE_SLIPPAGE:
                slippage_pips = np.random.uniform(self.SLIPPAGE_MIN_PIPS, self.SLIPPAGE_MAX_PIPS)
                slippage_price = slippage_pips * 0.0001 * price_raw
                price = price_raw + (direction * -1 * slippage_price)
            else:
                price = price_raw

            # Spread Simulation (Bid/Ask Logic)
            if self.ENABLE_SPREAD:
                spread_val = (self.SPREAD_MIN_PIPS * 0.0001) + (self.SPREAD_ATR_FACTOR * atr)
                price += direction * spread_val
            
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
        
        # Calculate SL/TP for virtual labeling (Include Slippage and Spread)
        price_raw = self.env._get_current_prices()[asset]
        atr = self.env._get_current_atrs()[asset]
        
        if self.ENABLE_SLIPPAGE:
            slippage_pips = np.random.uniform(self.SLIPPAGE_MIN_PIPS, self.SLIPPAGE_MAX_PIPS)
            slippage_price = slippage_pips * 0.0001 * price_raw
            price = price_raw + (direction * -1 * slippage_price)
        else:
            price = price_raw
            
        if self.ENABLE_SPREAD:
            spread_val = (self.SPREAD_MIN_PIPS * 0.0001) + (self.SPREAD_ATR_FACTOR * atr)
            price += direction * spread_val
            
        sl_dist = act['sl_mult'] * atr
        tp_dist = act['tp_mult'] * atr
        
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
                'entry_price': price,
                'spread_pips': (spread_val / 0.0001) if self.ENABLE_SPREAD else 0.0
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
    parser.add_argument("--alpha-model", type=str, default="models/checkpoints/alpha/ppo_final_model.zip")
    parser.add_argument("--risk-model", type=str, default="models/checkpoints/risk/model10M.zip")
    parser.add_argument("--guard-model", type=str, default="models/checkpoints/tradegurd/tradeguard_ppo.zip")
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--output-dir", type=str, default="Alpha/backtest/results/full_system")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--initial-equity", type=float, default=10.0)
    
    args = parser.parse_args()
    run_full_system_backtest(args)

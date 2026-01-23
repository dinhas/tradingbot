import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import Shared Execution Engine
from Shared.execution import ExecutionEngine, TradeConfig

# SB3 Compatibility
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Alpha.src.trading_env import TradingEnv
import gymnasium as gym
from gymnasium import spaces

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class DummyEnv(gym.Env):
    def __init__(self, obs_shape):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    def reset(self, seed=None): return np.zeros(self.observation_space.shape), {}
    def step(self, action): return np.zeros(self.observation_space.shape), 0, False, False, {}

class RiskMetrics:
    def __init__(self):
        self.total_signals = 0
        self.approved_trades = 0
        self.skipped_trades = 0
        self.exec_conf_history = []
        self.sl_mult_history = []
        self.tp_mult_history = []
        self.pnl_efficiency_history = []
        
    def to_dict(self):
        return {
            'total_signals': self.total_signals,
            'approved_trades': self.approved_trades,
            'skipped_trades': self.skipped_trades,
            'approval_rate': self.approved_trades / max(1, self.total_signals),
            'avg_sl_mult': np.mean(self.sl_mult_history) if self.sl_mult_history else 0,
            'avg_tp_mult': np.mean(self.tp_mult_history) if self.tp_mult_history else 0,
            'avg_efficiency': np.mean(self.pnl_efficiency_history) if self.pnl_efficiency_history else 0
        }

class CombinedBacktester:
    def __init__(self, args):
        self.args = args
        self.initial_equity = args.initial_equity
        self.equity = args.initial_equity
        self.peak_equity = args.initial_equity
        self.equity_curve = [self.initial_equity]
        self.timestamps = []
        
        # Initialize Execution Engine
        self.engine = ExecutionEngine()
        self.config = self.engine.config
        
        # Load Models
        self._load_models()
        
        # Initialize Environment
        self.env = TradingEnv(data_dir=args.data_dir, is_training=False)
        self.assets = self.env.assets
        self.max_steps = min(len(self.env.processed_data) - 1, args.max_steps) if args.max_steps else len(self.env.processed_data) - 1
        
        # Metrics
        self.risk_metrics = RiskMetrics()
        self.trades = []
        self.total_fees = 0.0
        self.total_lots = 0.0
        self.signals_log = []
        
        # History tracking for SAC observations (matching env behavior)
        self.recent_pnls = deque([0.0]*5, maxlen=5)
        self.recent_actions = deque([np.zeros(3) for _ in range(5)], maxlen=5)

    def _load_models(self):
        logger.info(f"Loading Alpha Model: {self.args.alpha_model}")
        self.alpha_model = PPO.load(self.args.alpha_model, device='cpu')
        
        self.alpha_norm = None
        if self.args.alpha_norm and os.path.exists(self.args.alpha_norm):
            logger.info(f"Loading Alpha Normalizer: {self.args.alpha_norm}")
            dummy_env = DummyVecEnv([lambda: DummyEnv((40,))])
            self.alpha_norm = VecNormalize.load(self.args.alpha_norm, dummy_env)
            self.alpha_norm.training = False
            self.alpha_norm.norm_reward = False

        logger.info(f"Loading Risk Model: {self.args.risk_model}")
        self.risk_model = PPO.load(self.args.risk_model, device='cpu')
        
        self.risk_norm = None
        if self.args.risk_norm and os.path.exists(self.args.risk_norm):
            logger.info(f"Loading Risk Normalizer: {self.args.risk_norm}")
            dummy_env = DummyVecEnv([lambda: DummyEnv((65,))])  # Updated to 65 for SAC
            self.risk_norm = VecNormalize.load(self.args.risk_norm, dummy_env)
            self.risk_norm.training = False
            self.risk_norm.norm_reward = False

    def build_risk_observation(self, asset_obs):
        """Build 65-dim observation for Risk model (40 market + 5 account + 5 PnL history + 15 action history)"""
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        equity_norm = self.equity / self.initial_equity
        risk_cap_mult = 1.0 # Fixed for consistency with training
        
        account_state = np.array([
            equity_norm,
            drawdown,
            0.0,  # Leverage placeholder
            risk_cap_mult,
            0.0   # Padding
        ], dtype=np.float32)
        
        # 3. PnL History (5) - Last 5 trades normalized PnL
        pnl_hist = np.array(list(self.recent_pnls), dtype=np.float32)
        
        # 4. Action History (15) - Last 5 actions flattened [SL, TP, Size] * 5
        action_hist = np.concatenate([np.array(a, dtype=np.float32) for a in self.recent_actions])
        
        return np.concatenate([asset_obs, account_state, pnl_hist, action_hist])

    def run(self):
        logger.info(f"Starting Combined Backtest (2025 Data)... total assets: {len(self.assets)}")
        
        for asset in self.assets:
            logger.info(f"--- Asset: {asset} ---")
            self.env.current_asset = asset
            obs, _ = self.env.reset()
            # Ensure reset doesn't change current_asset if we set it manually
            self.env.current_asset = asset 
            
            # Reset step for each asset
            self.env.current_step = 0
            asset_max_steps = min(len(self.env.processed_data) - 1, self.args.max_steps) if self.args.max_steps else len(self.env.processed_data) - 1
            
            for step in range(asset_max_steps):
                if step % 10000 == 0 and step > 0:
                    logger.info(f"  {asset} Step {step}/{asset_max_steps} | Equity: ${self.equity:.2f}")
                
                # 1. Alpha Inference
                if self.alpha_norm:
                    norm_obs = self.alpha_norm.normalize_obs(obs.reshape(1, -1))[0]
                else:
                    norm_obs = obs
                    
                alpha_raw_action, _ = self.alpha_model.predict(norm_obs, deterministic=True)
                alpha_signal = float(alpha_raw_action[0])
                
                # Signal Thresholding
                direction = 0
                if alpha_signal > 0.2: direction = 1
                elif alpha_signal < -0.2: direction = -1
                
                if direction != 0:
                    self.risk_metrics.total_signals += 1
                    
                    # 2. Risk Inference
                    risk_obs = self.build_risk_observation(obs)
                    if self.risk_norm:
                        risk_obs = self.risk_norm.normalize_obs(risk_obs.reshape(1, -1))[0]
                    
                    risk_action, _ = self.risk_model.predict(risk_obs, deterministic=True)
                    
                    # Decode SL/TP/Size from SAC Risk Model (3 actions)
                    # SL: [-1, 1] -> [0.5, 3.0] ATR
                    # TP: [-1, 1] -> [1.0, 10.0] ATR
                    # Size: [-1, 1] -> [0.01, 0.10] (1% to 10%)
                    sl_mult = np.clip((risk_action[0] + 1) / 2 * (3.0 - 0.5) + 0.5, 0.5, 3.0)
                    tp_mult = np.clip((risk_action[1] + 1) / 2 * (10.0 - 1.0) + 1.0, 1.0, 10.0)
                    risk_pct = np.clip((risk_action[2] + 1) / 2 * (0.10 - 0.01) + 0.01, 0.01, 0.10)
                    
                    # Track action for history
                    self.recent_actions.append(np.array([float(sl_mult), float(tp_mult), float(risk_pct)], dtype=np.float32))
                    
                    # 3. Position Sizing (using SAC model's risk_pct)
                    atr = self.env.atr_arrays[asset][self.env.current_step]
                    raw_price = self.env.close_arrays[asset][self.env.current_step]
                    
                    sl_dist_price = sl_mult * atr
                    
                    # Determine Asset Type
                    is_usd_quote = "USD" in asset and asset.endswith("USD")
                    is_usd_base = "USD" in asset and asset.startswith("USD")
                    
                    if not is_usd_quote and not is_usd_base:
                        is_usd_quote = True
                    
                    # Calculate lots using model's risk percentage
                    risk_amount_cash = self.equity * risk_pct
                    
                    if sl_dist_price > 1e-9:
                        if is_usd_quote:
                            lots = risk_amount_cash / (sl_dist_price * 100000)
                        elif is_usd_base:
                            lots = (risk_amount_cash * raw_price) / (sl_dist_price * 100000)
                        else:
                            lots = risk_amount_cash / (sl_dist_price * 100000)
                    else:
                        lots = 0.0
                    
                    # Leverage constraints
                    if is_usd_quote:
                        lot_value_usd = 100000 * raw_price
                    else:
                        lot_value_usd = 100000
                    
                    max_position_value = (self.equity * 0.80) * 400.0
                    max_lots_leverage = max_position_value / max(lot_value_usd, 1e-9)
                    lots = min(lots, max_lots_leverage)
                    lots = float(np.clip(lots, 0.01, 100.0))
                    
                    if lots > 0.0:
                        # Execute Trade
                        # Calculate Actual Entry Price (Spread + Slippage)
                        entry_price = self.engine.get_entry_price(raw_price, direction, atr)
                        
                        self._execute_trade(asset, direction, entry_price, raw_price, sl_mult, tp_mult, lots, atr, alpha_signal, is_usd_quote, is_usd_base)
                        self.risk_metrics.approved_trades += 1
                        
                        self.risk_metrics.sl_mult_history.append(sl_mult)
                        self.risk_metrics.tp_mult_history.append(tp_mult)
                    else:
                        self.risk_metrics.skipped_trades += 1

                # Advance
                self.env.current_step += 1
                if self.env.current_step >= asset_max_steps:
                    break
                obs = self.env._get_observation()
                
                # Update equity curve per step - use subset to keep plot readable
                if step % 288 == 0: # Daily point
                    self.equity_curve.append(self.equity)

        self._finalize()

    def _execute_trade(self, asset, direction, entry_price, raw_entry_price, sl_mult, tp_mult, lots, atr, alpha_conf, is_usd_quote, is_usd_base):
        """Simulate a single trade outcome based on ATR-relative exits using Shared Execution Engine"""
        idx = self.env.current_step
        max_lookahead = 576  # 48 hours
        
        # Use Engine for Exit Prices
        sl_price, tp_price, _, _ = self.engine.get_exit_prices(entry_price, direction, sl_mult, tp_mult, atr)
        
        exit_price = entry_price
        exit_reason = 'TIME_LIMIT'
        
        # Tracking for efficiency
        max_favorable_pct = 0.0
        
        exit_idx = idx
        for i in range(idx + 1, min(idx + max_lookahead, len(self.env.close_arrays[asset]))):
            exit_idx = i
            high = self.env.high_arrays[asset][i]
            low = self.env.low_arrays[asset][i]
            close = self.env.close_arrays[asset][i]
            
            # Max Favorable Calc
            if direction == 1:
                favorable = (high - entry_price) / entry_price
            else:
                favorable = (entry_price - low) / entry_price
            max_favorable_pct = max(max_favorable_pct, favorable)
            
            if direction == 1:
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
            else:
                if high >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'SL'
                    break
                if low <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'TP'
                    break
            
            # Update default exit to current close (will be used if Time Limit reached)
            exit_price = close

        # Advance env current_step to the exit index to avoid opening overlapping trades
        try:
            # Clamp to valid range
            self.env.current_step = min(exit_idx, len(self.env.close_arrays[asset]) - 1)
        except Exception:
            # If env arrays not available or unexpected error, leave current_step unchanged
            pass

        # Calculate Fees (Spread Cost)
        # Entry Fee
        entry_fee_dist = abs(entry_price - raw_entry_price)
        
        # Exit Fee
        if exit_reason == 'TIME_LIMIT':
            # We exited at close_price +/- spread
            # exit_price was set to 'close' (mid) then modified by get_close_price
            # So we need to calculate the actual execution price first
            exit_mid = exit_price
            exit_price = self.engine.get_close_price(exit_mid, direction, atr)
            exit_fee_dist = abs(exit_price - exit_mid)
        else:
            # SL/TP Hit
            # We assume exit_price (Trigger) was the execution price
            # The mid price would have been Trigger +/- Spread
            # So fee is just the spread at that price
            # Note: SL/TP prices in Shared/execution.py are the EXECUTION prices (e.g. Bid for Long SL)
            # So we effectively paid the spread to get there.
            exit_fee_dist = self.engine.get_spread(exit_price, atr)

        # Convert Fee Distance to USD
        # We use calculate_pnl logic: Price Delta * Lots * Contract * [Conversion]
        # Treat fee distance as a "profit" to get the magnitude in USD
        # We pass direction=1 always to get absolute value
        entry_fee_usd = self.engine.calculate_pnl(0, entry_fee_dist, lots, 1, is_usd_quote, is_usd_base)
        exit_fee_usd = self.engine.calculate_pnl(0, exit_fee_dist, lots, 1, is_usd_quote, is_usd_base)
        
        total_fee_usd = entry_fee_usd + exit_fee_usd
        self.total_fees += total_fee_usd
        self.total_lots += lots

        # Calculate P&L using Engine
        net_pnl = self.engine.calculate_pnl(entry_price, exit_price, lots, direction, is_usd_quote, is_usd_base)
        
        # Efficiency Metric (matching REWARD_SYSTEM.md)
        realized_pct = (exit_price - entry_price) / entry_price * direction
        denom = max(max_favorable_pct, atr / entry_price, 1e-5)
        efficiency = (realized_pct / denom) * 10.0
        self.risk_metrics.pnl_efficiency_history.append(efficiency)
        
        # Track PnL for history (scaled to match env expectations if needed)
        norm_pnl = net_pnl / max(self.initial_equity, 1e-6)
        self.recent_pnls.append(norm_pnl)
        
        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        
        self.trades.append({
            'asset': asset,
            'direction': direction,
            'entry': entry_price,
            'exit': exit_price,
            'reason': exit_reason,
            'pnl': net_pnl,
            'fee': total_fee_usd,
            'efficiency': efficiency,
            'lots': lots,
            'sl_mult': sl_mult,
            'tp_mult': tp_mult,
            'equity': self.equity
        })

    def _finalize(self):
        logger.info("Finalizing Backtest...")
        results = {
            'metrics': self._calculate_metrics(),
            'risk_metrics': self.risk_metrics.to_dict(),
            'total_trades': len(self.trades)
        }
        
        # Save JSON
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(self.args.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        with open(out_path / f"metrics_combined_v2_{ts}.json", 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
            
        # Save Trades CSV
        if self.trades:
            df_trades = pd.DataFrame(self.trades)
            df_trades.to_csv(out_path / f"trades_combined_v2_{ts}.csv", index=False)
            
        self._generate_charts(ts)
        logger.info(f"Results saved to {self.args.output_dir}")

    def _calculate_metrics(self):
        if not self.trades: return {}
        
        pnls = [t['pnl'] for t in self.trades]
        total_pnl = sum(pnls)
        win_rate = len([p for p in pnls if p > 0]) / len(pnls)
        profit_factor = sum([p for p in pnls if p > 0]) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float('inf')
        
        return {
            'total_pnl': total_pnl,
            'total_fees': self.total_fees,
            'avg_fee': self.total_fees / len(self.trades) if self.trades else 0.0,
            'avg_lots': self.total_lots / len(self.trades) if self.trades else 0.0,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'final_equity': self.equity,
            'max_drawdown': 1.0 - (min(self.equity_curve) / max(self.equity_curve))
        }

    def _generate_charts(self, ts):
        plt.figure(figsize=(15, 10))
        
        # 1. Equity Curve
        plt.subplot(2, 1, 1)
        plt.plot(self.equity_curve, label='Equity (Combined)')
        plt.title('Combined Backend v2 Strategy - 2025')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Risk Metrics Pie
        plt.subplot(2, 2, 3)
        labels = ['Approved', 'Skipped']
        sizes = [self.risk_metrics.approved_trades, self.risk_metrics.skipped_trades]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        plt.title('Risk Model Sniper Decisions')
        
        # 3. P&L Dist
        plt.subplot(2, 2, 4)
        if self.trades:
            pnls = [t['pnl'] for t in self.trades]
            sns.histplot(pnls, kde=True)
            plt.title('P&L Distribution')
        
        plt.tight_layout()
        plt.savefig(Path(self.args.output_dir) / f"charts_combined_v2_{ts}.png")
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-model", type=str, default="models/checkpoints/alpha/ppo_final_model.zip")
    parser.add_argument("--alpha-norm", type=str, default="models/checkpoints/alpha/ppo_final_vecnormalize.pkl")
    parser.add_argument("--risk-model", type=str, default="models/risk/risk_model_final.zip")
    parser.add_argument("--risk-norm", type=str, default="models/risk/vec_normalize.pkl")
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--output-dir", type=str, default="backtest/results_combined_v2")
    parser.add_argument("--initial-equity", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=None)
    
    args = parser.parse_args()
    
    backtester = CombinedBacktester(args)
    backtester.run()
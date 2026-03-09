"""
Combined Alpha-Risk Model Backtesting Script - OPTIMIZED VERSION
Resolved Merge Conflict: Vectorized PPO Inference with Flexible Thresholds
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
import torch
from tqdm import tqdm
import joblib

from Alpha.src.model import AlphaSLModel
from Alpha.src.trading_env import TradingEnv
from stable_baselines3 import PPO
from backtest.rl_backtest import BacktestMetrics, NumpyEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SL_CHOICES = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75, 3.0]
DEFAULT_INITIAL_EQUITY = 10000.0

class CombinedBacktest:
    """Combined backtest using Alpha model for direction and PPO Risk model for SL/TP/sizing"""

    def __init__(
        self,
        alpha_model,
        risk_model,
        risk_scaler,
        data_dir,
        initial_equity=DEFAULT_INITIAL_EQUITY,
        env=None,
        verify_alpha=False,
        challenge_mode=False,
        compounding=False,
        meta_thresh=0.78,
        qual_thresh=0.30
    ):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.risk_scaler = risk_scaler
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.verify_alpha = verify_alpha
        self.challenge_mode = challenge_mode
        self.compounding = compounding
        self.meta_thresh = meta_thresh
        self.qual_thresh = qual_thresh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Challenge Mode Tracking
        self.daily_high_water_mark = initial_equity
        self.daily_trades_count = 0
        self.is_halted_until_next_day = False
        self.current_day = None
        self.disqualified = False
        self.disqualification_reason = ""

        # Environment setup
        self.env = env if env is not None else TradingEnv(data_dir=data_dir, stage=1, is_training=False)
        self.env.equity = initial_equity
        self.equity = initial_equity
        self.peak_equity = initial_equity

        # Risk model constants
        self.MAX_LEVERAGE = 100.0
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000

    def calculate_position_size(self, asset, entry_price, size_out):
        """Calculate position size using Direct Model Allocation"""
        leverage = self.MAX_LEVERAGE
        if self.challenge_mode:
            is_forex = asset in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
            leverage = 100.0 if is_forex else 30.0

        base_equity = self.equity if self.compounding else self.initial_equity
        position_size = base_equity * size_out
        position_value_usd = position_size * leverage

        is_usd_quote = asset in ["EURUSD", "GBPUSD", "XAUUSD"]
        lot_value_usd = (self.CONTRACT_SIZE * entry_price if asset != "XAUUSD" else 100 * entry_price) if is_usd_quote else self.CONTRACT_SIZE

        lots = position_value_usd / (lot_value_usd + 1e-9)
        lots = np.clip(lots, self.MIN_LOTS, 5.0 if self.challenge_mode else 100.0)

        return size_out, lots, position_size

    def _precalculate_signals(self):
        """Batch inference for speed."""
        logger.info("Pre-calculating signals...")
        master_obs = self.env.master_obs_matrix 
        N, _ = master_obs.shape
        num_assets = len(self.env.assets)
        obs_flat = master_obs.reshape(-1, 40)
        batch_size = 4096

        # Alpha Inference
        alpha_dir, alpha_qual, alpha_meta = [], [], []
        for i in tqdm(range(0, len(obs_flat), batch_size), desc="Alpha Batch"):
            batch = torch.from_numpy(obs_flat[i : i + batch_size]).to(self.device)
            with torch.no_grad():
                dir_logits, qual, meta_logits = self.alpha_model(batch)
                alpha_dir.append((torch.argmax(dir_logits, dim=-1) - 1).cpu().numpy())
                alpha_qual.append(qual.squeeze(-1).cpu().numpy())
                alpha_meta.append(torch.sigmoid(meta_logits).squeeze(-1).cpu().numpy())

        self.alpha_direction_matrix = np.concatenate(alpha_dir).reshape(N, num_assets)
        self.alpha_quality_matrix = np.concatenate(alpha_qual).reshape(N, num_assets)
        self.alpha_meta_matrix = np.concatenate(alpha_meta).reshape(N, num_assets)

        # Risk Inference (PPO)
        if not self.verify_alpha:
            sl_l, tp_l, sz_l = [], [], []
            for i in tqdm(range(0, len(obs_flat), batch_size), desc="Risk Batch"):
                batch = obs_flat[i : i + batch_size]
                actions, _ = self.risk_model.predict(batch, deterministic=True)
                # PPO Mapping: -1...1 to real world values
                sl_l.append(0.8 + (actions[:, 0] + 1) / 2 * (3.5 - 0.8))
                tp_l.append(1.2 + (actions[:, 1] + 1) / 2 * (8.0 - 1.2))
                sz_l.append(0.1 + (actions[:, 2] + 1) / 2 * (1.0 - 0.1))
            
            self.sl_matrix = np.concatenate(sl_l).reshape(N, num_assets)
            self.tp_matrix = np.concatenate(tp_l).reshape(N, num_assets)
            self.size_matrix = np.concatenate(sz_l).reshape(N, num_assets)

    def run_backtest(self, episodes=1, max_steps=None):
        self._precalculate_signals()
        metrics_tracker = BacktestMetrics()
        assets = self.env.assets
        close_prices = {a: self.env.close_arrays[a] for a in assets}
        atr_values = {a: self.env.atr_arrays[a] for a in assets}

        for episode in range(episodes):
            self.env.reset()
            start_step = self.env.current_step
            end_step = min(start_step + (max_steps or self.env.max_steps), self.env.max_steps)

            for current_idx in tqdm(range(start_step, end_step), desc=f"Ep {episode+1}"):
                self.env.current_step = current_idx
                current_time = self.env._get_current_timestamp()

                # Day Management & Drawdown Checks
                day_str = current_time.strftime("%Y-%m-%d")
                if day_str != self.current_day:
                    self.current_day = day_str
                    self.daily_trades_count = 0
                    self.is_halted_until_next_day = False
                    self.daily_high_water_mark = self.equity

                daily_loss = self.daily_high_water_mark - self.equity
                if daily_loss >= (self.initial_equity * 0.05):
                    self.disqualified = True; break
                if daily_loss >= (self.initial_equity * 0.045):
                    self.is_halted_until_next_day = True

                # Signal Selection
                combined_actions = {}
                open_pos_count = sum(1 for p in self.env.positions.values() if p is not None)

                for i, asset in enumerate(assets):
                    if self.is_halted_until_next_day or self.daily_trades_count >= 50: break
                    
                    direction = int(self.alpha_direction_matrix[current_idx, i])
                    quality = self.alpha_quality_matrix[current_idx, i]
                    meta = self.alpha_meta_matrix[current_idx, i]

                    if direction == 0 or quality < self.qual_thresh or meta < self.meta_thresh:
                        continue

                    # Risk Params
                    sl_mult = self.sl_matrix[current_idx, i] if not self.verify_alpha else 2.0
                    tp_mult = self.tp_matrix[current_idx, i] if not self.verify_alpha else 4.0
                    size_out = self.size_matrix[current_idx, i] if not self.verify_alpha else 0.25
                    
                    entry_price = close_prices[asset][current_idx]
                    size_pct, lots, _ = self.calculate_position_size(asset, entry_price, size_out)

                    # Compounding adjustment for Env
                    env_size = size_pct if self.compounding else size_pct * (self.initial_equity / max(self.equity, 1e-9))

                    if env_size > 0.0001:
                        combined_actions[asset] = {
                            "direction": direction, "size": env_size,
                            "sl_mult": sl_mult, "tp_mult": tp_mult, "lots": lots
                        }

                # Execution
                self.env.completed_trades = []
                for asset, act in combined_actions.items():
                    # Simplified execution logic (abbreviated for brevity)
                    self.env._open_position(asset, act["direction"], act, close_prices[asset][current_idx], atr_values[asset][current_idx])

                self.env.current_step += 1
                self.env._update_positions()
                self.equity = self.env.equity
                for trade in self.env.completed_trades: metrics_tracker.add_trade(trade)
                metrics_tracker.add_equity_point(current_time, self.equity)

        return metrics_tracker
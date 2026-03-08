"""
Combined Alpha-Risk Model Backtesting Script - OPTIMIZED VERSION

Uses Vectorized Batch Inference and Pre-calculated features for maximum speed.
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
import tempfile
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

from Alpha.src.model import AlphaSLModel
from stable_baselines3 import PPO
import sys

SL_CHOICES = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75, 3.0]

# Absolute imports from project root
from Alpha.src.trading_env import TradingEnv
from backtest.rl_backtest import BacktestMetrics, NumpyEncoder, generate_all_charts
import joblib
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

default_initial_equity = 10000.0


class CombinedBacktest:
    """Combined backtest using Alpha model for direction and SL Risk model for SL/TP/sizing"""

    def __init__(
        self,
        alpha_model,
        risk_model,
        risk_scaler,
        data_dir,
        initial_equity=default_initial_equity,
        env=None,
        verify_alpha=False,
        challenge_mode=False,
        compounding=False,
    ):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.risk_scaler = risk_scaler
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.verify_alpha = verify_alpha
        self.challenge_mode = challenge_mode
        self.compounding = compounding
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Challenge Mode Tracking
        self.daily_high_water_mark = initial_equity
        self.daily_trades_count = 0
        self.is_halted_until_next_day = False
        self.current_day = None
        self.disqualified = False
        self.disqualification_reason = ""
        self.challenge_start_time = None
        self.challenge_end_time = None

        # Create Alpha environment for data access (or reuse existing)
        if env is not None:
            self.env = env
        else:
            self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)

        self.env.equity = initial_equity
        self.env.start_equity = initial_equity
        self.env.peak_equity = initial_equity

        # Risk model constants
        self.MAX_ALLOCATION_PCT = 0.60
        self.MAX_LEVERAGE = 100.0
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000

        # Slippage Configuration
        self.ENABLE_SLIPPAGE = True
        self.SLIPPAGE_MIN_PIPS = 0.5
        self.SLIPPAGE_MAX_PIPS = 1.5

        # Per-asset history tracking
        self.asset_histories = {
            asset: {
                "pnl_history": deque([0.0] * 5, maxlen=5),
                "action_history": deque(
                    [np.zeros(2, dtype=np.float32) for _ in range(5)], maxlen=5
                ),
            }
            for asset in self.env.assets
        }

        # Track current equity and peak
        self.equity = initial_equity
        self.peak_equity = initial_equity

    def calculate_position_size(self, asset, entry_price, size_out):
        """Calculate position size using Direct Model Allocation"""
        # Challenge Mode Leverage
        leverage = self.MAX_LEVERAGE
        if self.challenge_mode:
            is_forex = asset in [
                "EURUSD",
                "GBPUSD",
                "USDJPY",
                "USDCHF",
                "USDCAD",
                "AUDUSD",
                "NZDUSD",
            ]
            leverage = 100.0 if is_forex else 30.0

        # Position Sizing for stability
        # If compounding is False, we use initial_equity for sizing
        base_equity = self.equity if self.compounding else self.initial_equity
        position_size = base_equity * size_out
        position_value_usd = position_size * leverage

        # Calculate Lots
        is_usd_quote = asset in ["EURUSD", "GBPUSD", "XAUUSD"]
        if is_usd_quote:
            lot_value_usd = (
                self.CONTRACT_SIZE * entry_price
                if asset != "XAUUSD"
                else 100 * entry_price
            )
        else:
            lot_value_usd = self.CONTRACT_SIZE

        lots = position_value_usd / (lot_value_usd + 1e-9)

        # Challenge Mode Lot Limits
        if self.challenge_mode:
            is_forex = asset in [
                "EURUSD",
                "GBPUSD",
                "USDJPY",
                "USDCHF",
                "USDCAD",
                "AUDUSD",
                "NZDUSD",
            ]
            max_lots = 5.0 if is_forex else 3.0
            lots = np.clip(lots, self.MIN_LOTS, max_lots)
        else:
            lots = np.clip(lots, self.MIN_LOTS, 100.0)

        return size_out, lots, position_size

    def _precalculate_signals(self):
        """Pre-calculate all model outputs in large batches for the entire dataset."""
        logger.info(
            "Pre-calculating Alpha and Risk signals for all assets and steps..."
        )

        master_obs = self.env.master_obs_matrix  # (N, num_assets * 40)
        N, total_dims = master_obs.shape
        num_assets = len(self.env.assets)

        # Reshape to (N * num_assets, 40)
        obs_flat = master_obs.reshape(-1, 40)

        # 1. Alpha Batch Inference (AlphaSLModel: direction, quality, meta)
        logger.info(f"Running Alpha inference on {len(obs_flat)} observations...")

        alpha_direction_all = []
        alpha_quality_all = []
        alpha_meta_all = []
        batch_size = 4096

        for i in tqdm(range(0, len(obs_flat), batch_size), desc="Alpha Batch"):
            batch = torch.from_numpy(obs_flat[i : i + batch_size]).to(self.device)
            with torch.no_grad():
                dir_logits, quality, meta_logits = self.alpha_model(batch)
                # Map direction from [0,1,2] back to [-1,0,1]
                directions = torch.argmax(dir_logits, dim=-1).cpu().numpy() - 1
                # Quality is linear output (no sigmoid), meta uses sigmoid
                qualities = quality.squeeze(-1).cpu().numpy()
                metas = torch.sigmoid(meta_logits).squeeze(-1).cpu().numpy()
                alpha_direction_all.append(directions)
                alpha_quality_all.append(qualities)
                alpha_meta_all.append(metas)

        self.alpha_direction_matrix = np.concatenate(
            alpha_direction_all, axis=0
        ).reshape(N, num_assets)
        self.alpha_quality_matrix = np.concatenate(alpha_quality_all, axis=0).reshape(
            N, num_assets
        )
        self.alpha_meta_matrix = np.concatenate(alpha_meta_all, axis=0).reshape(
            N, num_assets
        )

        # Count how many would pass thresholds
        quality_passes = np.sum(self.alpha_quality_matrix >= 0.30)
        meta_passes = np.sum(self.alpha_meta_matrix >= 0.78)
        both_passes = np.sum(
            (self.alpha_quality_matrix >= 0.30) & (self.alpha_meta_matrix >= 0.78)
        )
        logger.info(
            f"DEBUG: Signals passing quality>=0.30: {quality_passes}/{self.alpha_quality_matrix.size}"
        )
        logger.info(
            f"DEBUG: Signals passing meta>=0.78: {meta_passes}/{self.alpha_meta_matrix.size}"
        )
        logger.info(
            f"DEBUG: Signals passing BOTH: {both_passes}/{self.alpha_quality_matrix.size}"
        )

        # 2. Risk Batch Inference (PPO model outputs: sl_raw, tp_raw, size_raw)
        if not self.verify_alpha:
            logger.info(f"Running Risk inference on {len(obs_flat)} observations...")

            sl_list, tp_list, size_list = [], [], []
            for i in tqdm(range(0, len(obs_flat), batch_size), desc="Risk Batch"):
                batch = obs_flat[i : i + batch_size]
                with torch.no_grad():
                    actions, _ = self.risk_model.predict(batch, deterministic=True)
                    # Decode PPO output: action[0]=sl_raw, action[1]=tp_raw, action[2]=size_raw
                    sl_raw = actions[:, 0]
                    tp_raw = actions[:, 1]
                    size_raw = actions[:, 2]

                    # Decode to actual values
                    # sl_mult: -1...1 -> 0.8...3.5
                    sl_mults = 0.8 + (sl_raw + 1) / 2 * (3.5 - 0.8)
                    # tp_mult: -1...1 -> 1.2...8.0
                    tp_mults = 1.2 + (tp_raw + 1) / 2 * (8.0 - 1.2)
                    # size_pct: -1...1 -> 0.1...1.0
                    size_pcts = 0.1 + (size_raw + 1) / 2 * (1.0 - 0.1)

                    sl_list.append(sl_mults)
                    tp_list.append(tp_mults)
                    size_list.append(size_pcts)

            self.sl_matrix = np.concatenate(sl_list, axis=0).reshape(N, num_assets)
            self.tp_matrix = np.concatenate(tp_list, axis=0).reshape(N, num_assets)
            self.size_matrix = np.concatenate(size_list, axis=0).reshape(N, num_assets)

        logger.info("Signal pre-calculation complete.")

    def run_backtest(self, episodes=1, max_steps=None):
        """Optimized Backtest using PRE-CALCULATED signals"""
        # Pre-calculate everything first
        self._precalculate_signals()

        metrics_tracker = BacktestMetrics()

        # Pre-cache assets and price arrays
        assets = self.env.assets
        num_assets = len(assets)
        close_prices = {a: self.env.close_arrays[a] for a in assets}
        atr_values = {a: self.env.atr_arrays[a] for a in assets}

        logger.info(
            f"Running {episodes} episodes with max {max_steps or self.env.max_steps} steps..."
        )

        for episode in range(episodes):
            self.env.reset()
            start_step = self.env.current_step
            end_step = min(
                start_step + (max_steps or self.env.max_steps), self.env.max_steps
            )

            # Debug: Print start and end step
            logger.info(f"DEBUG: start_step={start_step}, end_step={end_step}")

            alpha_non_zero = 0
            alpha_actions_sum = 0

            self.equity = self.initial_equity
            self.peak_equity = self.initial_equity
            self.env.equity = self.initial_equity

            if self.verify_alpha:
                logger.info("VERIFY ALPHA MODE: Risk Model Bypassed.")

            for current_idx in tqdm(
                range(start_step, end_step), desc=f"Ep {episode + 1}"
            ):
                self.env.current_step = current_idx
                current_time = self.env._get_current_timestamp()

                # --- CHALLENGE MODE DAY MANAGEMENT ---
                if self.challenge_mode:
                    if self.challenge_start_time is None:
                        self.challenge_start_time = current_time
                        self.challenge_end_time = self.challenge_start_time + timedelta(
                            days=30
                        )
                        logger.info(
                            f"Challenge Started: {self.challenge_start_time}. Ends: {self.challenge_end_time}"
                        )

                    if current_time >= self.challenge_end_time:
                        logger.info(
                            f"Challenge Period Ended (30 days reached). Final Time: {current_time}"
                        )
                        break

                    day_str = current_time.strftime("%Y-%m-%d")
                    if day_str != self.current_day:
                        # New Day Reset
                        if self.current_day is not None:
                            logger.info(
                                f"New Day: {day_str}. Daily trades: {self.daily_trades_count}. Halted: {self.is_halted_until_next_day}"
                            )

                        self.current_day = day_str
                        self.daily_trades_count = 0
                        self.is_halted_until_next_day = False
                        self.daily_high_water_mark = self.equity

                    if self.equity < (self.initial_equity * 0.90):
                        self.disqualified = True
                        self.disqualification_reason = f"Max Overall Loss Breached: Equity ${self.equity:.2f} < ${self.initial_equity * 0.90:.2f}"
                        logger.error(self.disqualification_reason)
                        break

                    daily_loss_amount = self.daily_high_water_mark - self.equity
                    max_daily_loss = self.initial_equity * 0.05

                    if daily_loss_amount >= max_daily_loss:
                        self.disqualified = True
                        self.disqualification_reason = f"Daily Loss Limit Breached: Loss ${daily_loss_amount:.2f} >= ${max_daily_loss:.2f}"
                        logger.error(self.disqualification_reason)
                        break

                    if daily_loss_amount >= (self.initial_equity * 0.045):
                        if not self.is_halted_until_next_day:
                            logger.warning(
                                f"4.5% Daily Drawdown reached. Halting trading until tomorrow. Time: {current_time}"
                            )
                            self.is_halted_until_next_day = True

                # --- FAST SIGNAL LOOKUP ---
                combined_actions = {}

                # Check if we can open new positions
                can_trade = True
                if self.challenge_mode:
                    if self.is_halted_until_next_day or self.daily_trades_count >= 50:
                        can_trade = False

                # Pre-calculate current open positions count
                open_pos_count = sum(
                    1 for p in self.env.positions.values() if p is not None
                )

                for i, asset in enumerate(assets):
                    if not can_trade:
                        break

                    # AlphaSLModel outputs: direction (-1, 0, 1), quality (raw), meta (0-1)
                    direction = int(self.alpha_direction_matrix[current_idx, i])
                    quality = self.alpha_quality_matrix[current_idx, i]
                    meta = self.alpha_meta_matrix[current_idx, i]

                    # Apply thresholds: quality >= 0.30 AND meta >= 0.78
                    if quality < 0.30 or meta < 0.78:
                        continue

                    # Skip if no direction
                    if direction == 0:
                        continue

                    alpha_actions_sum += abs(direction)

                    # Challenge Mode: Max 5 positions
                    if self.challenge_mode:
                        current_pos = self.env.positions.get(asset)
                        if current_pos is None:
                            if open_pos_count >= 5:
                                continue
                        elif current_pos["direction"] == direction:
                            continue

                    alpha_non_zero += 1

                    if self.verify_alpha:
                        sl_mult, tp_mult, size_pct, lots = 2.0, 4.0, 0.25, 0.1
                    else:
                        sl_mult = self.sl_matrix[current_idx, i]
                        tp_mult = self.tp_matrix[current_idx, i]
                        size_out = self.size_matrix[current_idx, i]

                        entry_price = close_prices[asset][current_idx]
                        size_pct, lots, pos_size = self.calculate_position_size(
                            asset, entry_price, max(size_out, 0.1)
                        )

                    # Adjust size for environment based on compounding preference
                    env_size = size_pct
                    if not self.compounding:
                        # To keep size non-compounding, relative to initial_equity:
                        # absolute_size_usd = passed_size * MAX_POS_SIZE_PCT * current_equity
                        # We want absolute_size_usd = size_pct * MAX_POS_SIZE_PCT * initial_equity
                        # So passed_size = size_pct * (initial_equity / current_equity)
                        env_size = size_pct * (self.initial_equity / max(self.equity, 1e-9))

                    if env_size > 0.0001:
                        combined_actions[asset] = {
                            "direction": direction,
                            "size": env_size,
                            "sl_mult": sl_mult,
                            "tp_mult": tp_mult,
                            "lots": lots,
                        }

                        if self.challenge_mode:
                            current_pos = self.env.positions.get(asset)
                            if current_pos is None or current_pos["direction"] != direction:
                                self.daily_trades_count += 1
                                if current_pos is None:
                                    open_pos_count += 1

                # Step 2: Execute Trades (at Close of Candle T)
                self.env.completed_trades = []

                for asset, act in combined_actions.items():
                    current_pos = self.env.positions[asset]
                    price_raw = close_prices[asset][current_idx]
                    atr = atr_values[asset][current_idx]

                    pip_scalar = 0.01 if "JPY" in asset or "XAU" in asset else 0.0001
                    slippage = np.random.uniform(0.5, 1.5) * pip_scalar
                    price = price_raw + (act["direction"] * -1 * slippage)

                    if current_pos is None:
                        self.env._open_position(
                            asset, act["direction"], act, price, atr
                        )
                    elif current_pos["direction"] != act["direction"]:
                        self.env._close_position(asset, price)
                        self.env._open_position(
                            asset, act["direction"], act, price, atr
                        )

                # Step 3: Advance to Candle T+1 and Update Positions
                self.env.current_step += 1
                if self.env.current_step >= self.env.max_steps:
                    break
                self.env._update_positions()

                self.equity = self.env.equity
                if not np.isfinite(self.equity):
                    logger.error(f"Equity is non-finite at step {current_idx}. Stopping.")
                    break
                self.peak_equity = max(self.peak_equity, self.equity)

                if self.env.completed_trades:
                    for trade in self.env.completed_trades:
                        metrics_tracker.add_trade(trade)

                metrics_tracker.add_equity_point(
                    self.env._get_current_timestamp(), self.equity
                )

                if current_idx % 10000 == 0:
                    logger.info(f"Step {current_idx}, Equity: ${self.equity:.2f}")
            logger.info(
                f"Episode {episode + 1} complete. Final Equity: ${self.equity:.2f}"
            )

        return metrics_tracker


def run_combined_backtest(args):
    """Main backtesting function"""
    project_root = Path(__file__).resolve().parent.parent

    # Add RiskLayer/src to path for custom policy loading
    sys.path.insert(0, str(project_root / "RiskLayer" / "src"))

    alpha_model_path = project_root / args.alpha_model
    risk_model_path = project_root / args.risk_model
    data_dir_path = project_root / args.data_dir
    output_dir_path = project_root / args.output_dir

    logger.info("Starting optimized Alpha-Risk backtest")
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if not alpha_model_path.exists():
        logger.error(f"Alpha model file not found: {alpha_model_path}")
        sys.exit(1)
    if not risk_model_path.exists():
        logger.error(f"Risk model file not found: {risk_model_path}")
        sys.exit(1)

    logger.info("Initializing environment...")
    shared_env = TradingEnv(data_dir=data_dir_path, stage=1, is_training=False)

    # Load AlphaSLModel (3 outputs: direction, quality, meta)
    logger.info(f"Loading AlphaSLModel from {alpha_model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    alpha_model = AlphaSLModel(input_dim=40)
    alpha_checkpoint = torch.load(
        alpha_model_path, map_location=device, weights_only=False
    )
    alpha_model.load_state_dict(alpha_checkpoint)
    alpha_model.to(device)
    alpha_model.eval()

    # Load PPO Risk Model
    logger.info(f"Loading PPO Risk Model from {risk_model_path}")
    risk_model = PPO.load(risk_model_path, env=None)

    backtest = CombinedBacktest(
        alpha_model,
        risk_model,
        None,  # No scaler needed for PPO
        data_dir_path,
        initial_equity=args.initial_equity,
        env=shared_env,
        verify_alpha=args.verify_alpha,
        challenge_mode=args.challenge_mode,
        compounding=args.compounding,
    )

    metrics_tracker = backtest.run_backtest(
        episodes=args.episodes, max_steps=args.max_steps
    )

    # Calculate and Print Metrics
    metrics = metrics_tracker.calculate_metrics()
    logger.info(f"\n{'BACKTEST RESULTS':^60}")
    logger.info("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k:<40} {v:.4f}")
        else:
            logger.info(f"{k:<40} {v}")
    logger.info("=" * 60)

    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = output_dir_path / f"metrics_combined_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)

    if metrics_tracker.trades:
        trades_file = output_dir_path / f"trades_combined_{timestamp}.csv"
        pd.DataFrame(metrics_tracker.trades).to_csv(trades_file, index=False)

        per_asset = metrics_tracker.get_per_asset_metrics()
        generate_all_charts(
            metrics_tracker, per_asset, "combined", output_dir_path, timestamp
        )

    logger.info("Backtest complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alpha-model", type=str, default="Alpha/models/alpha_model.pth"
    )
    parser.add_argument(
        "--risk-model",
        type=str,
        default="RiskLayer/models/ppo_risk_model_final_v2_opt.zip",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="backtest/results")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--verify-alpha", action="store_true")
    parser.add_argument(
        "--challenge-mode",
        action="store_true",
        help="Enable prop-firm challenge risk rules",
    )
    parser.add_argument(
        "--compounding",
        action="store_true",
        help="Enable equity compounding",
    )
    run_combined_backtest(parser.parse_args())

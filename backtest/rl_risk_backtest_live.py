"""
Backtest Script with RL Risk Model (PPO) - Realistic Live Execution Environment
Uses the default thresholds from LiveExecution/src/config.py:
- meta_threshold: 0.7071
- qual_threshold: 0.70
- risk_threshold: 0.10
"""

import os
import sys
import argparse
import logging
import json
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# NumPy compatibility shim
if not hasattr(np, "_core"):
    import sys

    sys.modules["numpy._core"] = np.core

# Create module alias for RiskActorCriticPolicy
import sys
from Risklayer.src.risk_model_ppo import RiskActorCriticPolicy

sys.modules["risk_model_ppo"] = sys.modules["Risklayer.src.risk_model_ppo"]

from Alpha.src.model import AlphaSLModel
from Alpha.src.trading_env import TradingEnv
from stable_baselines3 import PPO
from backtest.rl_backtest import BacktestMetrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_INITIAL_EQUITY = 10000.0


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class RealisticLiveBacktest:
    """
    Backtest using RL risk model with live execution thresholds and realistic environment settings.
    """

    def __init__(
        self,
        alpha_model,
        risk_model,
        risk_scaler,
        data_dir,
        initial_equity=DEFAULT_INITIAL_EQUITY,
        compounding=False,
        challenge_mode=False,
        meta_thresh=0.7071,
        qual_thresh=0.70,
        risk_thresh=0.10,
    ):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.risk_scaler = risk_scaler
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.compounding = compounding
        self.challenge_mode = challenge_mode
        self.meta_thresh = meta_thresh
        self.qual_thresh = qual_thresh
        self.risk_thresh = risk_thresh
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Challenge Mode Tracking
        self.daily_high_water_mark = initial_equity
        self.daily_trades_count = 0
        self.is_halted_until_next_day = False
        self.current_day = None
        self.disqualified = False
        self.disqualification_reason = ""

        # Environment setup
        self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
        self.env.equity = initial_equity
        self.equity = initial_equity
        self.peak_equity = initial_equity

        # Risk model constants (matching live execution)
        self.MAX_LEVERAGE = 100.0
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000

    def calculate_position_size(self, asset, entry_price, size_out):
        """Calculate position size using Direct Model Allocation (matching live execution)"""
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

        base_equity = self.equity if self.compounding else self.initial_equity
        position_size = base_equity * size_out
        position_value_usd = position_size * leverage

        is_usd_quote = asset in ["EURUSD", "GBPUSD", "XAUUSD"]
        lot_value_usd = (
            (
                self.CONTRACT_SIZE * entry_price
                if asset != "XAUUSD"
                else 100 * entry_price
            )
            if is_usd_quote
            else self.CONTRACT_SIZE
        )

        lots = position_value_usd / (lot_value_usd + 1e-9)
        lots = np.clip(lots, self.MIN_LOTS, 5.0 if self.challenge_mode else 100.0)

        return size_out, lots, position_size

    def _precalculate_signals(self):
        """Batch inference for speed (optimized)"""
        logger.info("Pre-calculating signals...")
        master_obs = self.env.master_obs_matrix
        N, _ = master_obs.shape
        num_assets = len(self.env.assets)
        obs_flat = master_obs.reshape(-1, 40)
        batch_size = 4096

        # Alpha Inference
        alpha_dir, alpha_qual, alpha_meta = [], [], []
        for i in tqdm(range(0, len(obs_flat), batch_size), desc="Alpha Batch"):
            batch = torch.from_numpy(
                obs_flat[i : i + batch_size].astype(np.float32)
            ).to(self.device)
            with torch.no_grad():
                dir_logits, qual, meta_logits = self.alpha_model(batch)
                alpha_dir.append((torch.argmax(dir_logits, dim=-1) - 1).cpu().numpy())
                alpha_qual.append(qual.squeeze(-1).cpu().numpy())
                alpha_meta.append(torch.sigmoid(meta_logits).squeeze(-1).cpu().numpy())

        self.alpha_direction_matrix = np.concatenate(alpha_dir).reshape(N, num_assets)
        self.alpha_quality_matrix = np.concatenate(alpha_qual).reshape(N, num_assets)
        self.alpha_meta_matrix = np.concatenate(alpha_meta).reshape(N, num_assets)

        # Risk Inference (PPO)
        sl_l, tp_l, sz_l = [], [], []
        for i in tqdm(range(0, len(obs_flat), batch_size), desc="Risk Batch"):
            batch = obs_flat[i : i + batch_size]
            obs_scaled = self.risk_scaler.transform(batch).astype(np.float32)
            actions, _ = self.risk_model.predict(obs_scaled, deterministic=True)

            # PPO Mapping: -1...1 to real world values (Matching RiskPPOEnv training)
            sl_l.append(0.8 + (actions[:, 0] + 1) / 2 * (3.5 - 0.8))
            tp_l.append(1.2 + (actions[:, 1] + 1) / 2 * (8.0 - 1.2))
            sz_l.append(0.01 + (actions[:, 2] + 1) / 2 * (0.30 - 0.01))

        self.sl_matrix = np.concatenate(sl_l).reshape(N, num_assets)
        self.tp_matrix = np.concatenate(tp_l).reshape(N, num_assets)
        self.size_matrix = np.concatenate(sz_l).reshape(N, num_assets)

    def run_backtest(self, episodes=1, max_steps=None):
        """Run backtest with realistic live execution settings"""
        self._precalculate_signals()
        metrics_tracker = BacktestMetrics()
        assets = self.env.assets
        close_prices = {a: self.env.close_arrays[a] for a in assets}
        atr_values = {a: self.env.atr_arrays[a] for a in assets}

        for episode in range(episodes):
            self.env.reset()
            start_step = self.env.current_step
            end_step = min(
                start_step + (max_steps or self.env.max_steps), self.env.max_steps
            )

            for current_idx in tqdm(
                range(start_step, end_step), desc=f"Ep {episode + 1}"
            ):
                self.env.current_step = current_idx
                current_time = self.env._get_current_timestamp()

                # Day Management & Drawdown Checks (matching live execution)
                day_str = current_time.strftime("%Y-%m-%d")
                if day_str != self.current_day:
                    self.current_day = day_str
                    self.daily_trades_count = 0
                    self.is_halted_until_next_day = False
                    self.daily_high_water_mark = self.equity

                daily_loss = self.daily_high_water_mark - self.equity
                if daily_loss >= (self.initial_equity * 0.05):
                    self.is_halted_until_next_day = True

                # Signal Selection
                combined_actions = {}
                open_pos_count = sum(
                    1 for p in self.env.positions.values() if p is not None
                )

                for i, asset in enumerate(assets):
                    if self.is_halted_until_next_day or self.daily_trades_count >= 50:
                        break
                    if self.env.positions[asset] is not None:
                        continue

                    direction = int(self.alpha_direction_matrix[current_idx, i])
                    quality = self.alpha_quality_matrix[current_idx, i]
                    meta = self.alpha_meta_matrix[current_idx, i]

                    if (
                        direction == 0
                        or quality < self.qual_thresh
                        or meta < self.meta_thresh
                    ):
                        continue

                    # Risk Params
                    sl_mult = self.sl_matrix[current_idx, i]
                    tp_mult = self.tp_matrix[current_idx, i]
                    size_out = self.size_matrix[current_idx, i]

                    if size_out < self.risk_thresh:
                        continue

                    entry_price = close_prices[asset][current_idx]
                    size_pct, lots, _ = self.calculate_position_size(
                        asset, entry_price, size_out
                    )

                    # Compounding adjustment for Env
                    env_size = (
                        size_pct
                        if self.compounding
                        else size_pct * (self.initial_equity / max(self.equity, 1e-9))
                    )

                    if env_size > 0.0001:
                        combined_actions[asset] = {
                            "direction": direction,
                            "size": env_size,
                            "sl_mult": sl_mult,
                            "tp_mult": tp_mult,
                            "lots": lots,
                        }

                # Execution
                self.env.completed_trades = []
                for asset, act in combined_actions.items():
                    self.env._open_position(
                        asset,
                        act["direction"],
                        act,
                        close_prices[asset][current_idx],
                        atr_values[asset][current_idx],
                    )
                    self.daily_trades_count += 1

                self.env._update_positions()
                self.equity = self.env.equity
                for trade in self.env.completed_trades:
                    metrics_tracker.add_trade(trade)
                metrics_tracker.add_equity_point(current_time, self.equity)

        return metrics_tracker


def main():
    parser = argparse.ArgumentParser(
        description="RL Risk Model Backtest - Realistic Live Execution Environment"
    )
    parser.add_argument(
        "--alpha-model", type=str, default="Alpha/models/alpha_model.pth"
    )
    parser.add_argument(
        "--risk-model",
        type=str,
        default="Risklayer/models/ppo_risk_model_final_v2_opt.zip",
    )
    parser.add_argument(
        "--risk-scaler", type=str, default="Risklayer/models/rl_risk_scaler.pkl"
    )
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--compounding", action="store_true", default=False)
    parser.add_argument("--challenge-mode", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="backtest/results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Models
    logger.info("Loading models...")
    alpha_model = AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4).to(
        device
    )
    alpha_model.load_state_dict(torch.load(args.alpha_model, map_location=device))
    alpha_model.eval()

    risk_model = PPO.load(args.risk_model, device=device)
    risk_scaler = joblib.load(args.risk_scaler)

    # Initialize Backtest with LIVE EXECUTION THRESHOLDS
    logger.info("Initializing backtest with live execution thresholds:")
    logger.info(f"  Meta threshold: 0.7071")
    logger.info(f"  Quality threshold: 0.70")
    logger.info(f"  Risk threshold: 0.10")

    bt = RealisticLiveBacktest(
        alpha_model=alpha_model,
        risk_model=risk_model,
        risk_scaler=risk_scaler,
        data_dir=args.data_dir,
        initial_equity=args.initial_equity,
        compounding=args.compounding,
        challenge_mode=args.challenge_mode,
        meta_thresh=0.7071,
        qual_thresh=0.70,
        risk_thresh=0.10,
    )

    logger.info("Starting backtest...")
    metrics = bt.run_backtest()
    final_results = metrics.calculate_metrics()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"rl_risk_backtest_live_thresholds_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=4, cls=NumpyEncoder)

    logger.info(f"\nBacktest Completed. Results saved to {results_file}")
    logger.info(f"\n=== Final Results ===")
    logger.info(f"Final Equity: {final_results['final_equity']:.2f}")
    logger.info(f"Total Return: {final_results['total_return']:.2%}")
    logger.info(f"Win Rate: {final_results['win_rate']:.2%}")
    logger.info(f"Max Drawdown: {final_results['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {final_results['sharpe_ratio']:.2f}")
    logger.info(f"Profit Factor: {final_results['profit_factor']:.3f}")
    logger.info(f"Number of Trades: {final_results['total_trades']}")


if __name__ == "__main__":
    main()

"""
Combined Alpha-Risk Model Backtesting Script — RL PPO Risk Model Edition

Uses the SL Alpha model for direction signals (meta/quality filtering)
and the PPO RL Risk Model for SL/TP placement & position sizing.

Execution model matches the RL training environment:
  - Bid/Ask half-spread on entry (Long@Ask, Short@Bid)
  - SL/TP anchored to MID price (not execution price)
  - SL/TP triggers checked against Bid (long) / Ask (short) with gap handling
  - Spread constants identical to training SPREAD_PIPS / PIP_SIZE
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add RiskLayer/src so SB3 can unpickle the custom policy class
_risk_src = os.path.join(project_root, "Risklayer", "src")
if _risk_src not in sys.path:
    sys.path.insert(0, _risk_src)

import numpy as np

# Numpy 1.x/2.x compatibility shim for SB3 model loading
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

import argparse
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure custom policy is importable before PPO.load()
from risk_model_ppo import RiskActorCriticPolicy, RiskFeatureExtractor  # noqa: F401

# Absolute imports from project root
from Alpha.src.trading_env import TradingEnv
from Alpha.src.model import AlphaSLModel
from backtest.rl_backtest import BacktestMetrics, NumpyEncoder, generate_all_charts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Training-Env Constants — MUST match risk_ppo_env.py exactly
# ─────────────────────────────────────────────────────────────
SPREAD_PIPS = {
    "EURUSD": 1.2,
    "GBPUSD": 1.5,
    "USDJPY": 1.0,
    "USDCHF": 1.8,
    "XAUUSD": 45.0,
}

PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "USDCHF": 0.0001,
    "XAUUSD": 0.1,
}

# PPO action-space mapping (from VectorizedRiskEnv.step_wait)
SL_LOW, SL_HIGH = 1.0, 3.5
TP_LOW, TP_HIGH = 1.2, 8.0
SIZE_LOW, SIZE_HIGH = 0.1, 0.3

initial_equity = 10.0


def _map_actions(raw_actions, sl_low=SL_LOW, sl_high=SL_HIGH, tp_low=TP_LOW, tp_high=TP_HIGH, size_low=SIZE_LOW, size_high=SIZE_HIGH):
    """Map PPO [-1,1] actions to SL/TP/Size using training-env formulas."""
    sl = sl_low + (raw_actions[:, 0] + 1) / 2 * (sl_high - sl_low)
    tp = tp_low + (raw_actions[:, 1] + 1) / 2 * (tp_high - tp_low)
    sz = size_low + (raw_actions[:, 2] + 1) / 2 * (size_high - size_low)
    return sl, tp, sz


class CombinedBacktest:
    """Combined backtest: SL Alpha (direction) + PPO RL Risk (SL/TP/sizing)"""

    def __init__(
        self,
        alpha_model,
        risk_model,
        data_dir,
        initial_equity=initial_equity,
        alpha_norm_env=None,
        env=None,
        verify_alpha=False,
        challenge_mode=False,
        meta_thresh=0.78,
        qual_thresh=0.30,
        sl_low=SL_LOW,
        sl_high=SL_HIGH,
        enable_trailing_stop=True,
        breakeven_trigger_r=0.9,
        breakeven_buffer_atr=0.10,
        trailing_trigger_r=1.25,
        trailing_atr_mult=1.0,
    ):
        self.alpha_model = alpha_model
        self.risk_model = risk_model        # SB3 PPO model
        self.alpha_norm_env = alpha_norm_env
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.verify_alpha = verify_alpha
        self.challenge_mode = challenge_mode
        self.meta_thresh = meta_thresh
        self.qual_thresh = qual_thresh
        self.sl_low = sl_low
        self.sl_high = sl_high
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
            # We want to use the consolidated backtest data if available
            self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
            
            # Update internal asset data if backtest file exists (overrides default search)
            for a in self.env.assets:
                fname = os.path.join(data_dir, f"{a}_5m_backtest.parquet")
                if os.path.exists(fname):
                    df = pd.read_parquet(fname)
                    self.env.data[a] = df
                    logger.info(f"TradingEnv: Loaded extended data for {a} from {fname}")
            
            # Re-process and re-cache if we loaded new data
            self.env.raw_data, self.env.processed_data = self.env.feature_engine.preprocess_data(self.env.data)
            self.env._cache_data_arrays()
            self.env.max_steps = len(self.env.processed_data) - 1

        self.env.equity = initial_equity
        self.env.start_equity = initial_equity
        self.env.peak_equity = initial_equity
        self.env.enable_trailing_stop = True
        self.env.breakeven_trigger_r = breakeven_trigger_r
        self.env.breakeven_buffer_atr = breakeven_buffer_atr
        self.env.trailing_trigger_r = trailing_trigger_r
        self.env.trailing_atr_mult = trailing_atr_mult

        # Position sizing constants
        self.MAX_LEVERAGE = 100.0
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000

        # The training env does NOT apply random entry-slippage.
        # To match training faithfully, we use half-spread only.
        self.ENABLE_SLIPPAGE = False

        # Track current equity and peak
        self.equity = initial_equity
        self.peak_equity = initial_equity

    # ────────────────── Execution helpers ──────────────────

    def _spread_price(self, asset):
        """Full spread in price terms (same table as training env)."""
        return SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)

    def _get_execution_price(self, asset, mid_price, direction):
        """
        Entry fill with half-spread — exactly matching training env.
        Long  → ask = mid + half_spread
        Short → bid = mid - half_spread
        """
        half_spread = self._spread_price(asset) / 2.0
        return mid_price + direction * half_spread

    def _open_position_rl(self, asset, direction, act, mid_price, atr):
        """
        Open position with SL/TP anchored to MID price.
        This matches the training env where:
            sl_price = entry_mid - direction * sl_mult * atr
            tp_price = entry_mid + direction * tp_mult * atr
        And actual_entry = mid ± half_spread.
        """
        exec_price = self._get_execution_price(asset, mid_price, direction)
        atr = max(atr, mid_price * self.env.MIN_ATR_MULTIPLIER)

        sl_dist = act["sl_mult"] * atr
        tp_dist = act["tp_mult"] * atr
        # Anchor SL/TP to MID (training fidelity)
        sl = mid_price - (direction * sl_dist)
        tp = mid_price + (direction * tp_dist)

        size = act["size"] * self.env.MAX_POS_SIZE_PCT * self.equity
        size = max(size, 0)

        self.env.positions[asset] = {
            "direction": direction,
            "entry_price": exec_price,
            "size": size,
            "sl": sl,
            "tp": tp,
            "entry_step": self.env.current_step,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "sl_mult": float(act.get("sl_mult", 0.0)),
            "tp_mult": float(act.get("tp_mult", 0.0)),
            "size_pct": float(act.get("size", 0.0)),
        }
        # Entry fee (matches Alpha env)
        self.equity -= size * 0.00002
        self.env.equity = self.equity

    def calculate_position_size(self, asset, entry_price, size_out):
        """Calculate lots for record-keeping."""
        leverage = self.MAX_LEVERAGE
        if self.challenge_mode:
            is_forex = asset in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
            leverage = 100.0 if is_forex else 30.0

        position_value_usd = self.equity * size_out * leverage
        is_usd_quote = asset in ["EURUSD", "GBPUSD", "XAUUSD"]
        if is_usd_quote:
            lot_value_usd = (self.CONTRACT_SIZE * entry_price) if asset != "XAUUSD" else (100 * entry_price)
        else:
            lot_value_usd = self.CONTRACT_SIZE

        lots = position_value_usd / (lot_value_usd + 1e-9)

        if self.challenge_mode:
            is_forex = asset in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]
            max_lots = 5.0 if is_forex else 3.0
            lots = np.clip(lots, self.MIN_LOTS, max_lots)
        else:
            lots = np.clip(lots, self.MIN_LOTS, 100.0)

        return size_out, lots

    # ────────────────── Signal Pre-calculation ──────────────────

    def _precalculate_signals(self):
        """Pre-calculate all Alpha & RL-Risk signals via batched inference."""
        logger.info("Pre-calculating Alpha and PPO Risk signals for all assets and steps...")

        master_obs = self.env.master_obs_matrix  # (N, num_assets * 40)
        N, total_dims = master_obs.shape
        num_assets = len(self.env.assets)

        # Reshape to (N * num_assets, 40)
        obs_flat = master_obs.reshape(-1, 40)

        # ── 1. Alpha Batch Inference ──
        logger.info(f"Running Alpha inference on {len(obs_flat)} observations...")
        if self.alpha_norm_env is not None:
            obs_norm = self.alpha_norm_env.normalize_obs(obs_flat)
        else:
            obs_norm = obs_flat

        dir_probs_list, meta_probs_list, qual_scores_list = [], [], []
        batch_size = 16384
        for i in tqdm(range(0, len(obs_norm), batch_size), desc="Alpha Batch"):
            batch = torch.from_numpy(obs_norm[i : i + batch_size]).to(self.device)
            with torch.no_grad():
                dir_logits, qual_pred, meta_logits = self.alpha_model(batch)
                dir_probs_list.append(torch.softmax(dir_logits, dim=1).cpu().numpy())
                meta_probs_list.append(torch.sigmoid(meta_logits).cpu().numpy())
                qual_scores_list.append(qual_pred.cpu().numpy())

        self.alpha_dir_probs = np.concatenate(dir_probs_list, axis=0).reshape(N, num_assets, 3)
        self.alpha_meta_probs = np.concatenate(meta_probs_list, axis=0).reshape(N, num_assets)
        self.alpha_qual_scores = np.concatenate(qual_scores_list, axis=0).reshape(N, num_assets)

        # ── 2. PPO Risk Batch Inference ──
        if not self.verify_alpha:
            logger.info(f"Running PPO Risk inference on {len(obs_flat)} observations...")
            sl_list, tp_list, size_list = [], [], []

            for i in tqdm(range(0, len(obs_flat), batch_size), desc="PPO Risk Batch"):
                batch_obs = obs_flat[i : i + batch_size]
                actions, _ = self.risk_model.predict(batch_obs, deterministic=True)

                sl_vals, tp_vals, sz_vals = _map_actions(actions, sl_low=self.sl_low, sl_high=self.sl_high)
                sl_list.append(sl_vals)
                tp_list.append(tp_vals)
                size_list.append(sz_vals)

            self.sl_matrix = np.concatenate(sl_list, axis=0).reshape(N, num_assets)
            self.tp_matrix = np.concatenate(tp_list, axis=0).reshape(N, num_assets)
            self.size_matrix = np.concatenate(size_list, axis=0).reshape(N, num_assets)

        logger.info("Signal pre-calculation complete.")

    # ────────────────── Main Loop ──────────────────

    def run_backtest(self, episodes=1, max_steps=None):
        """Run backtest using pre-calculated signals."""
        self._precalculate_signals()

        metrics_tracker = BacktestMetrics()

        assets = self.env.assets
        num_assets = len(assets)
        close_prices = {a: self.env.close_arrays[a] for a in assets}
        atr_values = {a: self.env.atr_arrays[a] for a in assets}

        logger.info(f"Running {episodes} episodes with max {max_steps or self.env.max_steps} steps...")
        logger.info(f"Thresholds: meta >= {self.meta_thresh}, qual >= {self.qual_thresh}")
        logger.info(f"RL action mapping: SL [{self.sl_low},{self.sl_high}], TP [{TP_LOW},{TP_HIGH}], Size [{SIZE_LOW},{SIZE_HIGH}]")

        for episode in range(episodes):
            self.env.reset()
            start_step = self.env.current_step
            end_step = min(start_step + (max_steps or self.env.max_steps), self.env.max_steps)

            self.equity = self.initial_equity
            self.peak_equity = self.initial_equity
            self.env.equity = self.initial_equity

            alpha_signals = 0

            if self.verify_alpha:
                logger.info("VERIFY ALPHA MODE: Risk Model Bypassed — fixed SL/TP/Size.")

            for current_idx in tqdm(range(start_step, end_step), desc=f"Ep {episode+1}"):
                self.env.current_step = current_idx
                current_time = self.env._get_current_timestamp()

                # ── CHALLENGE MODE DAY MANAGEMENT ──
                if self.challenge_mode:
                    if self.challenge_start_time is None:
                        self.challenge_start_time = current_time
                        self.challenge_end_time = self.challenge_start_time + timedelta(days=30)
                        logger.info(f"Challenge Started: {self.challenge_start_time}. Ends: {self.challenge_end_time}")

                    if current_time >= self.challenge_end_time:
                        logger.info(f"Challenge Period Ended. Final Time: {current_time}")
                        break

                    day_str = current_time.strftime("%Y-%m-%d")
                    if day_str != self.current_day:
                        if self.current_day is not None:
                            logger.info(f"New Day: {day_str}. Daily trades: {self.daily_trades_count}.")
                        self.current_day = day_str
                        self.daily_trades_count = 0
                        self.is_halted_until_next_day = False
                        self.daily_high_water_mark = self.equity

                    # Overall Loss (10%)
                    if self.equity < (self.initial_equity * 0.90):
                        self.disqualified = True
                        self.disqualification_reason = (
                            f"Max Overall Loss: Equity ${self.equity:.2f} < ${self.initial_equity*0.90:.2f}"
                        )
                        logger.error(self.disqualification_reason)
                        break

                    # Daily Loss (5%)
                    daily_loss = self.daily_high_water_mark - self.equity
                    max_daily = self.initial_equity * 0.05
                    if daily_loss >= max_daily:
                        self.disqualified = True
                        self.disqualification_reason = f"Daily Loss: ${daily_loss:.2f} >= ${max_daily:.2f}"
                        logger.error(self.disqualification_reason)
                        break

                    # Daily Drawdown Halt (4.5%)
                    if daily_loss >= (self.initial_equity * 0.045):
                        if not self.is_halted_until_next_day:
                            logger.warning(f"4.5% daily DD reached. Halting. Time: {current_time}")
                            self.is_halted_until_next_day = True

                # ── SIGNAL LOOKUP ──
                combined_actions = {}

                can_trade = True
                if self.challenge_mode:
                    if self.is_halted_until_next_day or self.daily_trades_count >= 50:
                        can_trade = False

                open_pos_count = sum(1 for p in self.env.positions.values() if p is not None)

                for i, asset in enumerate(assets):
                    if not can_trade:
                        break

                    dir_probs = self.alpha_dir_probs[current_idx, i]
                    meta_prob = self.alpha_meta_probs[current_idx, i]
                    qual_score = self.alpha_qual_scores[current_idx, i]

                    pred_idx = np.argmax(dir_probs)
                    direction = pred_idx - 1  # {0,1,2} → {-1,0,1}

                    if direction == 0:
                        continue

                    # Alpha quality gates
                    if meta_prob < self.meta_thresh or qual_score < self.qual_thresh:
                        continue

                    # Challenge mode position limits
                    if self.challenge_mode:
                        current_pos = self.env.positions.get(asset)
                        if current_pos is None:
                            if open_pos_count >= 5:
                                continue
                        elif current_pos["direction"] == direction:
                            continue

                    alpha_signals += 1

                    if self.verify_alpha:
                        sl_mult, tp_mult, size_pct, lots = 2.0, 4.0, 0.25, 0.1
                    else:
                        sl_mult = float(self.sl_matrix[current_idx, i])
                        tp_mult = float(self.tp_matrix[current_idx, i])
                        size_out = float(self.size_matrix[current_idx, i])

                        entry_price = close_prices[asset][current_idx]
                        size_pct, lots = self.calculate_position_size(asset, entry_price, size_out)

                    if size_pct > 0.01:
                        combined_actions[asset] = {
                            "direction": direction,
                            "size": size_pct,
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

                # ── EXECUTE TRADES ──
                self.env.completed_trades = []
                for asset, act in combined_actions.items():
                    current_pos = self.env.positions[asset]
                    mid_price = close_prices[asset][current_idx]
                    atr = atr_values[asset][current_idx]

                    if current_pos is None:
                        self._open_position_rl(asset, act["direction"], act, mid_price, atr)
                    elif current_pos["direction"] != act["direction"]:
                        # Close at execution price (opposite side)
                        close_price = self._get_execution_price(asset, mid_price, -current_pos["direction"])
                        self.env._close_position(asset, close_price)
                        self.equity = self.env.equity
                        # Open new
                        self._open_position_rl(asset, act["direction"], act, mid_price, atr)

                # ── ADVANCE & UPDATE ──
                self.env.current_step += 1
                if self.env.current_step >= self.env.max_steps:
                    break
                self.env._update_positions()

                self.equity = self.env.equity
                self.peak_equity = max(self.peak_equity, self.equity)

                if self.env.completed_trades:
                    for trade in self.env.completed_trades:
                        metrics_tracker.add_trade(trade)

                metrics_tracker.add_equity_point(self.env._get_current_timestamp(), self.equity)

                if current_idx % 10000 == 0:
                    logger.info(f"Step {current_idx}, Equity: ${self.equity:.4f}, Signals: {alpha_signals}")

            logger.info(f"Episode {episode+1} complete. Final Equity: ${self.equity:.4f}")
            if self.challenge_mode:
                if self.disqualified:
                    logger.error(f"CHALLENGE FAILED: {self.disqualification_reason}")
                else:
                    profit_pct = (self.equity - self.initial_equity) / self.initial_equity
                    if profit_pct >= 0.10:
                        logger.info(f"CHALLENGE PASSED! Profit: {profit_pct:.2%}")
                    else:
                        logger.warning(f"CHALLENGE ENDED: Profit target not met. Profit: {profit_pct:.2%}")

        return metrics_tracker


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def run_combined_backtest(args):
    """Main backtesting function"""
    project_root = Path(__file__).resolve().parent.parent
    alpha_model_path = project_root / args.alpha_model
    risk_model_path = project_root / args.risk_model
    data_dir_path = project_root / args.data_dir
    output_dir_path = project_root / args.output_dir

    logger.info("=" * 60)
    logger.info("COMBINED BACKTEST: SL Alpha + PPO RL Risk Model")
    logger.info("=" * 60)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if not alpha_model_path.exists():
        logger.error(f"Alpha model not found: {alpha_model_path}")
        sys.exit(1)
    if not risk_model_path.exists():
        logger.error(f"Risk model not found: {risk_model_path}")
        sys.exit(1)

    # ── Environment ──
    logger.info("Initializing environment...")
    shared_env = TradingEnv(data_dir=data_dir_path, stage=1, is_training=False)
    dummy_vec_env = DummyVecEnv([lambda: shared_env])

    # ── Alpha Normalizer ──
    alpha_norm_path = str(alpha_model_path).replace(".pth", "_vecnormalize.pkl")
    if not os.path.exists(alpha_norm_path):
        alpha_norm_path = str(alpha_model_path).replace(".zip", "_vecnormalize.pkl")
    if not os.path.exists(alpha_norm_path):
        alpha_norm_path = str(alpha_model_path).replace("_model.zip", "_vecnormalize.pkl")

    alpha_norm_env = None
    if os.path.exists(alpha_norm_path) and alpha_norm_path.endswith(".pkl"):
        try:
            logger.info(f"Loading Alpha Normalizer from {alpha_norm_path}")
            alpha_norm_env = VecNormalize.load(alpha_norm_path, dummy_vec_env)
            alpha_norm_env.training = False
            alpha_norm_env.norm_reward = False
        except Exception as e:
            logger.error(f"Failed to load normalizer: {e}")
            alpha_norm_env = None

    # ── Alpha Model (SL) ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_model = AlphaSLModel(input_dim=40).to(device)
    alpha_model.load_state_dict(torch.load(alpha_model_path, map_location=device))
    alpha_model.eval()
    logger.info(f"Alpha model loaded from {alpha_model_path}")

    # ── Risk Model (PPO RL) ──
    logger.info(f"Loading PPO RL Risk Model from {risk_model_path}")
    risk_model = PPO.load(risk_model_path, device=device)
    logger.info("PPO Risk Model loaded successfully")
    logger.info(f"  Action space: {risk_model.action_space}")
    logger.info(f"  Observation space: {risk_model.observation_space}")

    # ── Run Backtest ──
    backtest = CombinedBacktest(
        alpha_model,
        risk_model,
        data_dir_path,
        initial_equity=args.initial_equity,
        alpha_norm_env=alpha_norm_env,
        env=shared_env,
        verify_alpha=args.verify_alpha,
        challenge_mode=args.challenge_mode,
        meta_thresh=args.meta_thresh,
        qual_thresh=args.qual_thresh,
        sl_low=args.sl_low,
        sl_high=args.sl_high,
        enable_trailing_stop=not args.disable_trailing_stop,
        breakeven_trigger_r=args.breakeven_trigger_r,
        breakeven_buffer_atr=args.breakeven_buffer_atr,
        trailing_trigger_r=args.trailing_trigger_r,
        trailing_atr_mult=args.trailing_atr_mult,
    )

    metrics_tracker = backtest.run_backtest(episodes=args.episodes, max_steps=args.max_steps)

    # ── Results ──
    metrics = metrics_tracker.calculate_metrics()
    logger.info(f"\n{'BACKTEST RESULTS':^60}")
    logger.info("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k:<40} {v:.4f}")
        else:
            logger.info(f"{k:<40} {v}")
    logger.info("=" * 60)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = output_dir_path / f"metrics_combined_rl_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved metrics to {metrics_file}")

    if metrics_tracker.trades:
        trades_file = output_dir_path / f"trades_combined_rl_{timestamp}.csv"
        pd.DataFrame(metrics_tracker.trades).to_csv(trades_file, index=False)
        logger.info(f"Saved trade log to {trades_file}")

        per_asset = metrics_tracker.get_per_asset_metrics()
        generate_all_charts(metrics_tracker, per_asset, "combined_rl", output_dir_path, timestamp)

    logger.info("Backtest complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined SL-Alpha + RL-Risk Backtest")
    parser.add_argument("--alpha-model", type=str, default="Alpha/models/alpha_model.pth")
    parser.add_argument("--risk-model", type=str, default="Risklayer/models/ppo_risk_model_final_v2_opt.zip")
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--output-dir", type=str, default="backtest/results")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--initial-equity", type=float, default=10.0)
    parser.add_argument("--verify-alpha", action="store_true", help="Bypass risk model for alpha-only verification")
    parser.add_argument("--challenge-mode", action="store_true", help="Enable prop-firm challenge risk rules")
    parser.add_argument("--meta-thresh", type=float, default=0.78)
    parser.add_argument("--qual-thresh", type=float, default=0.30)
    parser.add_argument("--sl-low", type=float, default=1.0, help="Lower bound for SL ATR multiplier action mapping.")
    parser.add_argument("--sl-high", type=float, default=3.5, help="Upper bound for SL ATR multiplier action mapping.")
    parser.add_argument("--disable-trailing-stop", action="store_true", help="Disable BE/trailing-stop logic in backtest.")
    parser.add_argument("--breakeven-trigger-r", type=float, default=0.9, help="Move SL to BE after this many R in unrealized profit.")
    parser.add_argument("--breakeven-buffer-atr", type=float, default=0.10, help="Extra ATR buffer when moving stop to breakeven.")
    parser.add_argument("--trailing-trigger-r", type=float, default=1.25, help="Arm trailing stop after this many R in unrealized profit.")
    parser.add_argument("--trailing-atr-mult", type=float, default=1.0, help="ATR multiplier used for trailing stop distance.")
    run_combined_backtest(parser.parse_args())

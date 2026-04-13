"""
Alpha-Only Backtesting Script — LSTM Alpha Model Edition

Uses the LSTM Alpha model for direction signals (meta/quality filtering)
with fixed SL and TP multipliers as requested.

Execution model:
  - Bid/Ask half-spread on entry (Long@Ask, Short@Bid)
  - SL/TP anchored to MID price (not execution price)
  - SL/TP triggers checked against Bid (long) / Ask (short) with gap handling
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
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

# Absolute imports from project root
from Alpha.src.trading_env import TradingEnv
from Alpha.src.model import AlphaSLModel
from Alpha.src.feature_engine import NUM_FEATURES
from backtest.rl_backtest import BacktestMetrics, NumpyEncoder, generate_all_charts

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants — Matching TradingEnv and Risk Env
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

SEQ_LEN = 30
INITIAL_EQUITY = 10.0


def _build_sequence_windows(features_2d: np.ndarray, seq_len: int = SEQ_LEN):
    """Build LSTM windows and return (X_seq, valid_idx)."""
    n = len(features_2d)
    if n <= seq_len:
        return np.empty((0, seq_len, features_2d.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    valid_idx = np.arange(seq_len, n, dtype=np.int64)
    X_seq = np.stack([features_2d[i - seq_len : i] for i in valid_idx], axis=0).astype(np.float32)
    return X_seq, valid_idx


class AlphaSoloBacktest:
    """Alpha-only backtest: LSTM Alpha (direction) + Fixed Risk (SL/TP/sizing)"""

    def __init__(
        self,
        alpha_model,
        data_dir,
        initial_equity=INITIAL_EQUITY,
        env=None,
        meta_thresh=0.55,
        qual_thresh=0.90,
        fixed_sl=2.0,
        fixed_tp=4.0,
        fixed_size=0.25,
        trade_cooldown_bars=6,  # Minimum bars between trades per asset
        enable_trailing_stop=True,
        breakeven_trigger_r=0.9,
        breakeven_buffer_atr=0.10,
        trailing_trigger_r=1.25,
        trailing_atr_mult=1.0,
    ):
        self.alpha_model = alpha_model
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.meta_thresh = meta_thresh
        self.qual_thresh = qual_thresh
        self.fixed_sl = fixed_sl
        self.fixed_tp = fixed_tp
        self.fixed_size = fixed_size
        self.trade_cooldown_bars = trade_cooldown_bars
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create Alpha environment for data access (or reuse existing)
        if env is not None:
            self.env = env
        else:
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
            self.env._create_master_obs_matrix()
            self.env.max_steps = len(self.env.processed_data) - 1

        self.env.equity = initial_equity
        self.env.start_equity = initial_equity
        self.env.peak_equity = initial_equity
        self.env.enable_trailing_stop = enable_trailing_stop
        self.env.breakeven_trigger_r = breakeven_trigger_r
        self.env.breakeven_buffer_atr = breakeven_buffer_atr
        self.env.trailing_trigger_r = trailing_trigger_r
        self.env.trailing_atr_mult = trailing_atr_mult

        # Position sizing constants
        self.MAX_LEVERAGE = 100.0
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000

        # Track current equity and peak
        self.equity = initial_equity
        self.peak_equity = initial_equity

    def _spread_price(self, asset):
        """Full spread in price terms."""
        return SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)

    def _get_execution_price(self, asset, mid_price, direction):
        """Entry fill with half-spread."""
        half_spread = self._spread_price(asset) / 2.0
        return mid_price + direction * half_spread

    def _open_position_fixed(self, asset, direction, mid_price, atr):
        """Open position with fixed SL/TP anchored to MID price."""
        exec_price = self._get_execution_price(asset, mid_price, direction)
        atr = max(atr, mid_price * self.env.MIN_ATR_MULTIPLIER)

        sl_dist = self.fixed_sl * atr
        tp_dist = self.fixed_tp * atr
        
        sl = mid_price - (direction * sl_dist)
        tp = mid_price + (direction * tp_dist)

        size = self.fixed_size * self.env.MAX_POS_SIZE_PCT * self.equity
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
            "sl_mult": float(self.fixed_sl),
            "tp_mult": float(self.fixed_tp),
            "size_pct": float(self.fixed_size),
            "initial_risk_dist": sl_dist,
            "best_price": exec_price,
            "trailing_active": False,
            "trailing_distance": 0.0,
            "current_stop_loss": sl,
        }
        # Entry fee
        self.equity -= size * 0.00002
        self.env.equity = self.equity

    def _precalculate_signals(self):
        """Pre-calculate Alpha signals via batched inference."""
        logger.info("Pre-calculating Alpha signals for all assets and steps...")

        master_obs = self.env.master_obs_matrix  # (N, num_assets * NUM_FEATURES)
        N, total_dims = master_obs.shape
        num_assets = len(self.env.assets)
        obs_by_asset = master_obs.reshape(N, num_assets, NUM_FEATURES)

        # Alpha Batch Inference
        logger.info(f"Running Alpha sequence inference on {N} timesteps x {num_assets} assets...")
        self.alpha_dir_probs = np.zeros((N, num_assets, 3), dtype=np.float32)
        self.alpha_meta_probs = np.zeros((N, num_assets), dtype=np.float32)
        self.alpha_qual_scores = np.zeros((N, num_assets), dtype=np.float32)
        self.alpha_dir_probs[:, :, 1] = 1.0  # default neutral

        batch_size = 4096
        with torch.no_grad():
            for asset_idx in tqdm(range(num_assets), desc="Alpha Seq Assets"):
                obs_asset = obs_by_asset[:, asset_idx, :]  # (N, 40)
                X_seq, valid_idx = _build_sequence_windows(obs_asset, seq_len=SEQ_LEN)
                if len(valid_idx) == 0:
                    continue
                for i in range(0, len(X_seq), batch_size):
                    batch = torch.from_numpy(X_seq[i : i + batch_size]).to(self.device)
                    dir_logits, qual_pred, meta_logits = self.alpha_model(batch)
                    self.alpha_dir_probs[valid_idx[i : i + batch_size], asset_idx, :] = torch.softmax(dir_logits, dim=1).cpu().numpy()
                    self.alpha_meta_probs[valid_idx[i : i + batch_size], asset_idx] = torch.sigmoid(meta_logits).squeeze(-1).cpu().numpy()
                    self.alpha_qual_scores[valid_idx[i : i + batch_size], asset_idx] = qual_pred.squeeze(-1).cpu().numpy()

        logger.info("Signal pre-calculation complete.")

    def run_backtest(self, episodes=1, max_steps=None):
        """Run backtest using pre-calculated signals."""
        self._precalculate_signals()

        metrics_tracker = BacktestMetrics()

        assets = self.env.assets
        close_prices = {a: self.env.close_arrays[a] for a in assets}
        atr_values = {a: self.env.atr_arrays[a] for a in assets}

        logger.info(f"Running {episodes} episodes with max {max_steps or self.env.max_steps} steps...")
        logger.info(f"Thresholds: meta >= {self.meta_thresh}, qual >= {self.qual_thresh}")
        logger.info(f"Fixed Params: SL={self.fixed_sl}, TP={self.fixed_tp}, Size={self.fixed_size}")

        for episode in range(episodes):
            self.env.reset()
            start_step = self.env.current_step
            end_step = min(start_step + (max_steps or self.env.max_steps), self.env.max_steps)

            self.equity = self.initial_equity
            self.peak_equity = self.initial_equity
            self.env.equity = self.initial_equity

            alpha_signals = 0
            last_trade_step = {a: -999 for a in assets}  # Cooldown tracker

            for current_idx in tqdm(range(start_step, end_step), desc=f"Ep {episode+1}"):
                self.env.current_step = current_idx
                
                # SIGNAL LOOKUP
                combined_actions = {}
                for i, asset in enumerate(assets):
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

                    # Per-asset cooldown to prevent overtrading
                    if (current_idx - last_trade_step[asset]) < self.trade_cooldown_bars:
                        continue

                    alpha_signals += 1
                    combined_actions[asset] = {"direction": direction}

                # EXECUTE TRADES
                self.env.completed_trades = []
                for asset, act in combined_actions.items():
                    current_pos = self.env.positions[asset]
                    mid_price = close_prices[asset][current_idx]
                    atr = atr_values[asset][current_idx]

                    if current_pos is None:
                        self._open_position_fixed(asset, act["direction"], mid_price, atr)
                        last_trade_step[asset] = current_idx
                    elif current_pos["direction"] != act["direction"]:
                        close_price = self._get_execution_price(asset, mid_price, -current_pos["direction"])
                        self.env._close_position(asset, close_price)
                        self.equity = self.env.equity
                        self._open_position_fixed(asset, act["direction"], mid_price, atr)
                        last_trade_step[asset] = current_idx

                # ADVANCE & UPDATE
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

        return metrics_tracker


def main():
    parser = argparse.ArgumentParser(description="Alpha-Only LSTM Backtest")
    parser.add_argument("--alpha-model", type=str, default="Alpha/models/alpha_model.pth")
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--output-dir", type=str, default="backtest/results")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--initial-equity", type=float, default=10.0)
    parser.add_argument("--meta-thresh", type=float, default=0.55)
    parser.add_argument("--qual-thresh", type=float, default=0.90)
    parser.add_argument("--sl", type=float, default=2.0, help="Fixed SL ATR multiplier")
    parser.add_argument("--tp", type=float, default=4.0, help="Fixed TP ATR multiplier")
    parser.add_argument("--cooldown", type=int, default=6, help="Min bars between trades per asset")
    parser.add_argument("--size", type=float, default=0.25, help="Fixed position size multiplier")
    parser.add_argument("--disable-trailing", action="store_true", help="Disable trailing stop")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    alpha_model_path = project_root / args.alpha_model
    data_dir_path = project_root / args.data_dir
    output_dir_path = project_root / args.output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if not alpha_model_path.exists():
        logger.error(f"Alpha model not found: {alpha_model_path}")
        sys.exit(1)

    # Load optimized thresholds if they exist
    thresholds_path = output_dir_path / "optimal_thresholds_alpha_solo.json"
    if thresholds_path.exists():
        try:
            with open(thresholds_path, "r") as f:
                best = json.load(f)
            args.meta_thresh = float(best.get("meta_threshold", args.meta_thresh))
            args.qual_thresh = float(best.get("qual_threshold", args.qual_thresh))
            logger.info(f"Loaded optimized thresholds: meta={args.meta_thresh}, qual={args.qual_thresh}")
        except Exception as e:
            logger.warning(f"Failed to load optimized thresholds: {e}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha_model = AlphaSLModel(input_dim=NUM_FEATURES, hidden_dim=256, num_layers=2).to(device)
    alpha_model.load_state_dict(torch.load(alpha_model_path, map_location=device))
    alpha_model.eval()
    logger.info(f"Alpha model loaded from {alpha_model_path}")

    backtest = AlphaSoloBacktest(
        alpha_model,
        data_dir_path,
        initial_equity=args.initial_equity,
        meta_thresh=args.meta_thresh,
        qual_thresh=args.qual_thresh,
        fixed_sl=args.sl,
        fixed_tp=args.tp,
        fixed_size=args.size,
        trade_cooldown_bars=args.cooldown,
        enable_trailing_stop=not args.disable_trailing,
    )

    metrics_tracker = backtest.run_backtest(episodes=args.episodes, max_steps=args.max_steps)
    metrics = metrics_tracker.calculate_metrics()

    logger.info(f"\n{'ALPHA-ONLY BACKTEST RESULTS':^60}")
    logger.info("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"{k:<40} {v:.4f}")
        else:
            logger.info(f"{k:<40} {v}")
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = output_dir_path / f"metrics_alpha_solo_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    
    if metrics_tracker.trades:
        trades_file = output_dir_path / f"trades_alpha_solo_{timestamp}.csv"
        pd.DataFrame(metrics_tracker.trades).to_csv(trades_file, index=False)
        per_asset = metrics_tracker.get_per_asset_metrics()
        generate_all_charts(metrics_tracker, per_asset, "alpha_solo", output_dir_path, timestamp)

    logger.info("Backtest complete!")


if __name__ == "__main__":
    main()

"""
Combined Alpha-Risk Model Backtesting Script - OPTIMIZED VERSION
Resolved Merge Conflict: Vectorized PPO Inference with Flexible Thresholds
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

# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add numpy 1.x/2.x compatibility shim for SB3 model loading
if not hasattr(np, "_core"):
    import sys

    sys.modules["numpy._core"] = np.core

from Alpha.src.model import AlphaSLModel
from Alpha.src.trading_env import TradingEnv
from RiskLayer.model import MultiHeadRiskLSTM
from backtest.rl_backtest import BacktestMetrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SL_CHOICES = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75, 3.0]
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


class CombinedBacktest:
    """Combined backtest using Alpha model for direction and SL Risk model for SL/TP/sizing"""

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
        risk_thresh=0.10,
        seq_len=50,
    ):
        self.alpha_model = alpha_model
        self.risk_model = risk_model
        self.risk_scaler = risk_scaler
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        self.verify_alpha = verify_alpha
        self.challenge_mode = challenge_mode
        self.compounding = compounding
        self.risk_thresh = risk_thresh
        self.seq_len = seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Challenge Mode Tracking
        self.daily_high_water_mark = initial_equity
        self.daily_trades_count = 0
        self.is_halted_until_next_day = False
        self.current_day = None
        self.disqualified = False
        self.disqualification_reason = ""

        # Environment setup
        self.env = (
            env
            if env is not None
            else TradingEnv(data_dir=data_dir, stage=1, is_training=False)
        )
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

    def _precalculate_signals(self, max_steps=None, start_date=None, end_date=None):
        """Batch inference for speed with windowing."""
        logger.info("Pre-calculating signals...")
        
        df = self.env.processed_data
        n_total = len(df)
        num_assets = len(self.env.assets)
        input_dim = 17 # From new alpha model
        batch_size = 2048

        # Determine range for pre-calculation
        start_idx = self.seq_len - 1
        end_idx = n_total

        if start_date:
            start_ts = pd.to_datetime(start_date).tz_localize(df.index.tz)
            start_indices = np.where(df.index >= start_ts)[0]
            if len(start_indices) > 0:
                start_idx = max(self.seq_len - 1, start_indices[0])
        
        if end_date:
            end_ts = pd.to_datetime(end_date).tz_localize(df.index.tz)
            end_indices = np.where(df.index <= end_ts)[0]
            if len(end_indices) > 0:
                end_idx = min(n_total, end_indices[-1] + 1)

        if max_steps and not (start_date or end_date):
            end_idx = min(n_total, start_idx + max_steps)

        logger.info(f"Pre-calculating from index {start_idx} to {end_idx}")

        # Optimize: Only extract features for the required range + lookback
        feature_start = max(0, start_idx - self.seq_len + 1)
        feature_end = end_idx
        data_slice = self.env.processed_data.iloc[feature_start:feature_end]
        
        logger.info(f"Extracting features for range {feature_start}:{feature_end}...")
        master_obs_17 = np.zeros((feature_end - feature_start, num_assets, input_dim), dtype=np.float32)
        for i, asset in enumerate(self.env.assets):
            master_obs_17[:, i, :] = self.env.feature_engine.get_observation_vectorized(data_slice, asset)
            
        # Helper to map global index to slice index
        def to_slice_idx(global_idx):
            return global_idx - feature_start

        # Initialize matrices
        self.alpha_direction_matrix = np.zeros((n_total, num_assets), dtype=np.int8)
        self.alpha_probs_matrix = np.zeros((n_total, num_assets, 3), dtype=np.float32)
        self.alpha_quality_matrix = np.ones((n_total, num_assets), dtype=np.float32)
        self.alpha_meta_matrix = np.ones((n_total, num_assets), dtype=np.float32)

        # 1. Alpha Inference (Batched)
        logger.info(f"Alpha Inference (Batched)...")
        from numpy.lib.stride_tricks import sliding_window_view
        
        for asset_idx in range(num_assets):
            asset_obs = master_obs_17[:, asset_idx, :]
            # Create sliding windows: (N_windows, seq_len, input_dim)
            windows = sliding_window_view(asset_obs, (self.seq_len, input_dim)).squeeze(1)
            
            # The windows correspond to global indices [start_idx, end_idx)
            # Find which windows in the slice correspond to our target range
            # window[0] is for slice index self.seq_len-1
            # We want windows for global_idx in [start_idx, end_idx)
            # slice_idx = global_idx - feature_start
            # window index = slice_idx - (self.seq_len - 1)

            target_global_indices = np.arange(start_idx, end_idx)
            target_slice_indices = target_global_indices - feature_start
            window_indices = target_slice_indices - (self.seq_len - 1)

            # Filter valid window indices
            valid_mask = (window_indices >= 0) & (window_indices < len(windows))
            window_indices = window_indices[valid_mask]
            target_global_indices = target_global_indices[valid_mask]

            relevant_windows = windows[window_indices]

            for b in tqdm(range(0, len(relevant_windows), batch_size), desc=f"Alpha {self.env.assets[asset_idx]}"):
                batch = torch.from_numpy(relevant_windows[b : b + batch_size].astype(np.float32)).to(self.device)
                with torch.inference_mode():
                    logits = self.alpha_model(batch)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    
                    directions = np.argmax(probs, axis=1) - 1
                    
                    batch_globals = target_global_indices[b : b + batch_size]
                    self.alpha_probs_matrix[batch_globals, asset_idx] = probs
                    self.alpha_direction_matrix[batch_globals, asset_idx] = directions

        # 2. Risk Inference (Batched)
        if not self.verify_alpha:
            logger.info(f"Risk Inference (Batched 21-features)...")
            self.sl_matrix = np.zeros((n_total, num_assets), dtype=np.float32)
            self.tp_matrix = np.zeros((n_total, num_assets), dtype=np.float32)
            self.size_matrix = np.zeros((n_total, num_assets), dtype=np.float32)
            
            for asset_idx in range(num_assets):
                asset_obs = master_obs_17[:, asset_idx, :]
                
                # Scale base observations if scaler exists
                if self.risk_scaler is not None:
                    obs_scaled = self.risk_scaler.transform(asset_obs).astype(np.float32)
                else:
                    obs_scaled = asset_obs.astype(np.float32)
                
                # Construct 21 features
                alpha_decision = np.zeros(len(obs_scaled), dtype=np.float32)
                alpha_probs = np.zeros((len(obs_scaled), 3), dtype=np.float32)

                # Map back global precalculated alpha values to this slice
                global_indices = np.arange(feature_start, feature_end)
                alpha_decision = self.alpha_direction_matrix[global_indices, asset_idx].astype(np.float32)
                alpha_probs = self.alpha_probs_matrix[global_indices, asset_idx]
                
                full_features = np.concatenate([obs_scaled, alpha_probs, alpha_decision[..., np.newaxis]], axis=1)
                
                # Create windows for 21 features
                windows = sliding_window_view(full_features, (self.seq_len, 21)).squeeze(1)

                target_global_indices = np.arange(start_idx, end_idx)
                target_slice_indices = target_global_indices - feature_start
                window_indices = target_slice_indices - (self.seq_len - 1)

                valid_mask = (window_indices >= 0) & (window_indices < len(windows))
                window_indices = window_indices[valid_mask]
                target_global_indices = target_global_indices[valid_mask]

                relevant_windows = windows[window_indices]

                for b in tqdm(range(0, len(relevant_windows), batch_size), desc=f"Risk {self.env.assets[asset_idx]}"):
                    batch = torch.from_numpy(relevant_windows[b : b + batch_size].astype(np.float32)).to(self.device)
                    with torch.inference_mode():
                        sl, tp, quality = self.risk_model(batch)
                        
                        batch_globals = target_global_indices[b : b + batch_size]
                        self.sl_matrix[batch_globals, asset_idx] = sl.cpu().numpy()
                        self.tp_matrix[batch_globals, asset_idx] = tp.cpu().numpy()
                        self.size_matrix[batch_globals, asset_idx] = quality.cpu().numpy()

    def run_backtest(self, episodes=1, max_steps=None, start_date=None, end_date=None):
        self._precalculate_signals(max_steps=max_steps, start_date=start_date, end_date=end_date)
        metrics_tracker = BacktestMetrics()
        assets = self.env.assets
        close_prices = {a: self.env.close_arrays[a] for a in assets}
        atr_values = {a: self.env.atr_arrays[a] for a in assets}

        for episode in range(episodes):
            self.env.reset()
            
            # Date Filtering logic
            if start_date or end_date:
                df = self.env.processed_data
                if start_date:
                    start_ts = pd.to_datetime(start_date).tz_localize(df.index.tz)
                    start_indices = np.where(df.index >= start_ts)[0]
                    actual_start = start_indices[0] if len(start_indices) > 0 else self.env.current_step
                else:
                    actual_start = self.env.current_step
                
                if end_date:
                    end_ts = pd.to_datetime(end_date).tz_localize(df.index.tz)
                    end_indices = np.where(df.index <= end_ts)[0]
                    actual_end = end_indices[-1] if len(end_indices) > 0 else self.env.max_steps
                else:
                    actual_end = self.env.max_steps
                
                start_step = max(actual_start, self.seq_len) # Ensure enough history
                end_step = min(actual_end, self.env.max_steps)
            else:
                start_step = self.env.current_step
                end_step = min(
                    start_step + (max_steps or self.env.max_steps), self.env.max_steps
                )

            logger.info(f"Running backtest from {self.env.processed_data.index[start_step]} to {self.env.processed_data.index[end_step]}")

            for current_idx in tqdm(
                range(start_step, end_step), desc=f"Ep {episode + 1}"
            ):
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
                    self.is_halted_until_next_day = (
                        True  # Changed from disqualified for standard backtest
                    )

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

                    if direction == 0:
                        continue

                    # Risk Params (from Risk LSTM)
                    sl_mult = (
                        self.sl_matrix[current_idx, i] if not self.verify_alpha else 2.0
                    )
                    tp_mult = (
                        self.tp_matrix[current_idx, i] if not self.verify_alpha else 4.0
                    )
                    size_out = (
                        self.size_matrix[current_idx, i]
                        if not self.verify_alpha
                        else 0.25
                    )

                    # Risk Gate: Risk model predicts "Quality" (Confidence) [0, 1]
                    if not self.verify_alpha and size_out < self.risk_thresh:
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
        description="Alpha-Risk Multi-Model Combined Backtest"
    )
    parser.add_argument(
        "--alpha-model", type=str, default="Alpha/models/alpha_model.pth"
    )
    parser.add_argument(
        "--risk-model",
        type=str,
        default="RiskLayer/models/risk_lstm_multitask.pth",
    )
    parser.add_argument(
        "--risk-scaler", type=str, default="RiskLayer/models/sl_risk_scaler.pkl"
    )
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--risk-thresh", type=float, default=0.10, help="Min confidence from risk model")
    parser.add_argument("--compounding", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps for backtest")
    parser.add_argument("--output-dir", type=str, default="backtest/results")
    parser.add_argument("--start-date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Models
    # Alpha Model (17 features)
    alpha_model = AlphaSLModel(input_dim=17, lstm_units=64, dense_units=32).to(device)
    alpha_model.load_state_dict(torch.load(args.alpha_model, map_location=device))
    alpha_model.eval()

    # Risk Model (21 features: 17 market + 3 probs + 1 decision)
    risk_model = MultiHeadRiskLSTM(input_size=21, hidden_size=128, num_layers=2).to(device)
    risk_checkpoint = torch.load(args.risk_model, map_location=device)
    risk_model.load_state_dict(risk_checkpoint["model_state_dict"])
    risk_model.eval()

    # Load Scaler (The one used during training)
    # If the user didn't download a specific one, we assume the old one exists or needs to be found
    if not os.path.exists(args.risk_scaler):
        # Check if it's in the same folder as the risk model
        alt_scaler = os.path.join(os.path.dirname(args.risk_model), "sl_risk_scaler.pkl")
        if os.path.exists(alt_scaler):
            args.risk_scaler = alt_scaler
        else:
            logger.warning(f"Risk scaler not found at {args.risk_scaler}. Inference might be inaccurate.")
    
    risk_scaler = joblib.load(args.risk_scaler) if os.path.exists(args.risk_scaler) else None

    # Initialize Backtest
    bt = CombinedBacktest(
        alpha_model=alpha_model,
        risk_model=risk_model,
        risk_scaler=risk_scaler,
        data_dir=args.data_dir,
        initial_equity=args.initial_equity,
        compounding=args.compounding,
        risk_thresh=args.risk_thresh,
    )

    metrics = bt.run_backtest(max_steps=args.max_steps, start_date=args.start_date, end_date=args.end_date)
    final_results = metrics.calculate_metrics()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"combined_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=4, cls=NumpyEncoder)

    logger.info(f"Backtest Completed. Results saved to {results_file}")
    logger.info(f"Final Equity: {final_results['final_equity']:.2f}")
    logger.info(f"Total Return: {final_results['total_return']:.2%}")
    logger.info(f"Win Rate: {final_results['win_rate']:.2%}")
    logger.info(f"Max Drawdown: {final_results['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()

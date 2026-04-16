
import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.src.model import AlphaSLModel
from Alpha.src.data_loader import DataLoader as AlphaDataLoader
from Alpha.src.feature_engine import FeatureEngine
from backtest.rl_backtest import BacktestMetrics, generate_all_charts, NumpyEncoder
from ta.trend import ADXIndicator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlphaLSTMBacktester:
    def __init__(
        self,
        model: AlphaSLModel,
        aligned_df: pd.DataFrame,
        normalized_df: pd.DataFrame,
        sequence_length: int,
        confidence_thresh: float,
        initial_equity: float,
        position_size: float,
        sl_mult: float,
        tp_mult: float,
        session_col: str = "is_late_session",
    ):
        self.model = model
        self.aligned_df = aligned_df
        self.normalized_df = normalized_df
        self.sequence_length = sequence_length
        self.confidence_thresh = confidence_thresh
        self.initial_equity = initial_equity
        self.position_size = position_size
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.session_col = session_col

        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.feature_engine = FeatureEngine()

        self.positions = {asset: None for asset in self.assets}
        self.completed_trades = []
        self.equity = initial_equity
        self.leverage = 100
        self.max_pos_size_pct = 0.50
        self.min_atr_multiplier = 1e-4
        self._asset_count = len(self.assets)

        self.asset_obs = {
            asset: self.feature_engine.get_observation_vectorized(self.normalized_df, asset)
            for asset in self.assets
        }
        
        # Pre-calculate ADX for regime filtering in backtest parity
        self.adx_data = {}
        for asset in self.assets:
            high = self.aligned_df[f"{asset}_high"]
            low = self.aligned_df[f"{asset}_low"]
            close = self.aligned_df[f"{asset}_close"]
            self.adx_data[asset] = ADXIndicator(high, low, close, window=14).adx().fillna(0)
            
        self.session_mask = self.normalized_df[self.session_col] == 1
        self.session_indices = np.where(self.session_mask.values)[0]

        # Cache commonly accessed market arrays to avoid repeated DataFrame iloc lookups in the main loop.
        self.close_data = {
            asset: self.aligned_df[f"{asset}_close"].to_numpy(dtype=np.float32)
            for asset in self.assets
        }
        self.atr_data = {
            asset: self.aligned_df[f"{asset}_atr"].to_numpy(dtype=np.float32)
            for asset in self.assets
        }
        self.adx_np = {
            asset: self.adx_data[asset].to_numpy(dtype=np.float32)
            for asset in self.assets
        }
        self.timestamps = self.normalized_df.index.to_numpy()

    def _open_position(self, asset, direction, entry_price, atr, timestamp, idx):
        size = self.position_size * self.max_pos_size_pct * self.equity
        atr = max(float(atr), float(entry_price) * self.min_atr_multiplier)
        sl_dist = self.sl_mult * atr
        tp_dist = self.tp_mult * atr
        sl = entry_price - (direction * sl_dist)
        tp = entry_price + (direction * tp_dist)

        equity_before = self.equity
        self.equity -= size * 0.00002  # entry fee

        self.positions[asset] = {
            'direction': int(direction),
            'entry_price': float(entry_price),
            'size': float(size),
            'sl': float(sl),
            'tp': float(tp),
            'entry_timestamp': timestamp,
            'entry_idx': idx,
            'equity_before': equity_before,
        }

    def _close_position(self, asset, exit_price, timestamp):
        pos = self.positions[asset]
        if pos is None:
            return

        price_change_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['direction']
        pnl = price_change_pct * (pos['size'] * self.leverage)

        self.equity += pnl
        self.equity -= pos['size'] * 0.00002  # exit fee
        equity_after = self.equity

        # Convert numpy timedelta64 to seconds
        diff = timestamp - pos['entry_timestamp']
        hold_seconds = diff.astype('timedelta64[s]').astype(float)
        hold_minutes = max(0.0, hold_seconds / 60.0)
        fees = pos['size'] * 0.00004

        self.completed_trades.append({
            'timestamp': timestamp,
            'asset': asset,
            'pnl': float(pnl),
            'fees': float(fees),
            'net_pnl': float(pnl - fees),
            'entry_price': float(pos['entry_price']),
            'exit_price': float(exit_price),
            'size': float(pos['size']),
            'equity_before': float(pos['equity_before']),
            'equity_after': float(equity_after),
            'hold_time': float(hold_minutes),
            'rr_ratio': float(self.tp_mult / max(self.sl_mult, 1e-8)),
        })
        self.positions[asset] = None

    def _precalculate_all_predictions(self, n_steps):
        logger.info(f"Pre-calculating model predictions for {n_steps} steps...")
        all_probs = np.zeros((n_steps, len(self.assets), 3), dtype=np.float32)

        num_windows = n_steps - self.sequence_length + 1
        if num_windows <= 0:
            return all_probs

        for asset_i, asset in enumerate(self.assets):
            obs = self.asset_obs[asset][:n_steps]
            # Create sliding windows: (num_windows, sequence_length, features)
            windows = np.lib.stride_tricks.sliding_window_view(
                obs, (self.sequence_length, obs.shape[1])
            ).squeeze(axis=1)

            # Inference in large batches
            batch_size = 4096
            for i in range(0, num_windows, batch_size):
                end_idx = min(i + batch_size, num_windows)
                # Copy to avoid non-writable tensor warning from sliding_window_view
                batch_windows = torch.as_tensor(windows[i:end_idx].copy(), device=DEVICE)
                with torch.inference_mode():
                    logits = self.model(batch_windows)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_probs[self.sequence_length - 1 + i : self.sequence_length - 1 + end_idx, asset_i, :] = probs

        return all_probs

    def run(self, max_steps: int | None = None) -> BacktestMetrics:
        start_time = time.time()
        metrics = BacktestMetrics()

        n_steps = len(self.normalized_df)
        if n_steps < self.sequence_length + 1:
            logger.warning("Dataset too small for sequence length.")
            return metrics

        total_steps = n_steps
        if max_steps is not None:
            total_steps = min(n_steps, self.sequence_length + max_steps)

        # Pre-calculate all predictions at once
        pre_calc_start = time.time()
        all_probs = self._precalculate_all_predictions(total_steps)
        pre_calc_end = time.time()
        logger.info(f"Pre-calculation took {pre_calc_end - pre_calc_start:.4f} seconds.")

        # Pre-calculate directions based on confidence threshold
        max_probs = all_probs.max(axis=2)
        best_classes = all_probs.argmax(axis=2) - 1 # Map [0,1,2] to [-1,0,1]

        if self.confidence_thresh > 0:
            all_directions = np.where(max_probs >= self.confidence_thresh, best_classes, 0)
        else:
            all_directions = best_classes

        # Pre-fetch data into arrays for much faster access
        asset_names = self.assets
        num_assets = len(asset_names)
        close_data_arr = np.stack([self.close_data[a][:total_steps] for a in asset_names])
        atr_data_arr = np.stack([self.atr_data[a][:total_steps] for a in asset_names])
        adx_arr = np.stack([self.adx_np[a][:total_steps] for a in asset_names])
        timestamps = self.timestamps[:total_steps]
        all_directions_T = all_directions.T  # (num_assets, total_steps)

        # Pre-calculate constant values
        pos_size_base = self.position_size * self.max_pos_size_pct
        leverage = float(self.leverage)
        min_atr_multiplier = self.min_atr_multiplier
        sl_mult = self.sl_mult
        tp_mult = self.tp_mult
        rr_ratio = float(tp_mult / max(sl_mult, 1e-8))

        # Re-initialize positions for the run
        self.positions = {asset: None for asset in asset_names}
        self.completed_trades = []

        # Localize variables for the loop
        equity = float(self.initial_equity)
        positions = [self.positions[a] for a in asset_names]

        # Iterate through all available steps
        loop_start = time.time()
        steps_run = 0
        for idx in range(self.sequence_length - 1, total_steps - 1):
            next_idx = idx + 1
            ts = timestamps[idx]
            next_ts = timestamps[next_idx]
            steps_run += 1

            # Process signals for all assets
            for i in range(num_assets):
                direction = all_directions_T[i, idx]
                pos = positions[i]

                if pos is None:
                    if direction != 0 and adx_arr[i, idx] >= 20.0:
                        # Open position
                        entry_price = float(close_data_arr[i, idx])
                        atr = max(float(atr_data_arr[i, idx]), entry_price * min_atr_multiplier)
                        size = pos_size_base * equity
                        sl = entry_price - (direction * sl_mult * atr)
                        tp = entry_price + (direction * tp_mult * atr)

                        equity_before = equity
                        equity -= size * 0.00002  # entry fee

                        pos = {
                            'direction': int(direction), 'entry_price': entry_price, 'size': size,
                            'sl': sl, 'tp': tp, 'entry_timestamp': ts, 'entry_idx': idx,
                            'equity_before': equity_before
                        }
                        positions[i] = pos
                else:
                    if direction == 0 or (direction ^ pos['direction'] < 0): # Check if flip (sign bit different)
                        # Close position
                        exit_price = float(close_data_arr[i, idx])
                        price_change_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['direction']
                        pnl = price_change_pct * (pos['size'] * leverage)
                        equity += pnl
                        fee = pos['size'] * 0.00002
                        equity -= fee

                        hold_seconds = (ts - pos['entry_timestamp']).astype('timedelta64[s]').astype(float)
                        self.completed_trades.append({
                            'timestamp': ts, 'asset': asset_names[i], 'pnl': pnl, 'fees': fee + (pos['size'] * 0.00002),
                            'net_pnl': pnl - (fee + pos['size'] * 0.00002), 'entry_price': pos['entry_price'],
                            'exit_price': exit_price, 'size': pos['size'], 'equity_before': pos['equity_before'],
                            'equity_after': equity, 'hold_time': max(0.0, hold_seconds / 60.0), 'rr_ratio': rr_ratio
                        })

                        if direction != 0 and adx_arr[i, idx] >= 20.0:
                            # Re-open position
                            entry_price = float(close_data_arr[i, idx])
                            atr = max(float(atr_data_arr[i, idx]), entry_price * min_atr_multiplier)
                            size = pos_size_base * equity
                            sl = entry_price - (direction * sl_mult * atr)
                            tp = entry_price + (direction * tp_mult * atr)
                            equity_before = equity
                            equity -= size * 0.00002  # entry fee
                            pos = {
                                'direction': int(direction), 'entry_price': entry_price, 'size': size,
                                'sl': sl, 'tp': tp, 'entry_timestamp': ts, 'entry_idx': idx,
                                'equity_before': equity_before
                            }
                            positions[i] = pos
                        else:
                            positions[i] = None

            # Evaluate SL/TP/Time barriers for open positions
            for i in range(num_assets):
                pos = positions[i]
                if pos is None:
                    continue
                
                next_price = float(close_data_arr[i, next_idx])
                exit_price = -1.0
                
                if (next_idx - pos['entry_idx']) >= 24:
                    exit_price = next_price
                elif pos['direction'] == 1:
                    if next_price <= pos['sl']: exit_price = pos['sl']
                    elif next_price >= pos['tp']: exit_price = pos['tp']
                else:
                    if next_price >= pos['sl']: exit_price = pos['sl']
                    elif next_price <= pos['tp']: exit_price = pos['tp']

                if exit_price != -1.0:
                    # Close position
                    price_change_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * pos['direction']
                    pnl = price_change_pct * (pos['size'] * leverage)
                    equity += pnl
                    fee = pos['size'] * 0.00002
                    equity -= fee

                    hold_seconds = (next_ts - pos['entry_timestamp']).astype('timedelta64[s]').astype(float)
                    self.completed_trades.append({
                        'timestamp': next_ts, 'asset': asset_names[i], 'pnl': pnl, 'fees': fee + (pos['size'] * 0.00002),
                        'net_pnl': pnl - (fee + pos['size'] * 0.00002), 'entry_price': pos['entry_price'],
                        'exit_price': exit_price, 'size': pos['size'], 'equity_before': pos['equity_before'],
                        'equity_after': equity, 'hold_time': max(0.0, hold_seconds / 60.0), 'rr_ratio': rr_ratio
                    })
                    positions[i] = None

            if self.completed_trades:
                metrics.trades.extend(self.completed_trades)
                self.completed_trades.clear()

            metrics.add_equity_point(next_ts, equity)

        # Update final state
        self.equity = equity
        for i, a in enumerate(asset_names):
            self.positions[a] = positions[i]

        end_time = time.time()
        logger.info(f"Total run() took {end_time - start_time:.4f} seconds.")
        logger.info(f"Main loop (backtest) took {end_time - loop_start:.4f} seconds for {steps_run} steps.")

        # Force-close any remaining open positions at the end.
        final_idx = n_steps - 1
        final_ts = self.timestamps[final_idx]
        for asset, pos_data in list(self.positions.items()):
            if pos_data is None:
                continue
            last_price = float(self.close_data[asset][final_idx])
            self._close_position(asset, last_price, final_ts)

        if self.completed_trades:
            for trade in self.completed_trades:
                metrics.add_trade(trade)
            self.completed_trades.clear()

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Backtest Alpha LSTM model with session-only entries")
    parser.add_argument("--model-path", type=str, default="Alpha/models/alpha_model.pth")
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--output-dir", type=str, default="backtest/results")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument("--confidence-thresh", type=float, default=0.45)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--position-size", type=float, default=0.2)
    parser.add_argument("--sl-mult", type=float, default=2.0)
    parser.add_argument("--tp-mult", type=float, default=4.0)
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model_path
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    loader = AlphaDataLoader(data_dir=str(data_dir))
    aligned_df, normalized_df = loader.get_features()

    input_dim = 17
    model = AlphaSLModel(input_dim=input_dim, lstm_units=64, dense_units=32, dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    bt = AlphaLSTMBacktester(
        model=model,
        aligned_df=aligned_df,
        normalized_df=normalized_df,
        sequence_length=args.sequence_length,
        confidence_thresh=args.confidence_thresh,
        initial_equity=args.initial_equity,
        position_size=args.position_size,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
    )

    metrics = bt.run(max_steps=args.steps)
    results = metrics.calculate_metrics()
    per_asset = metrics.get_per_asset_metrics()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_file = output_dir / f"metrics_alpha_lstm_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    if metrics.trades:
        trades_file = output_dir / f"trades_alpha_lstm_{timestamp}.csv"
        pd.DataFrame(metrics.trades).to_csv(trades_file, index=False)

    if per_asset:
        asset_file = output_dir / f"asset_breakdown_alpha_lstm_{timestamp}.csv"
        pd.DataFrame(per_asset).T.to_csv(asset_file)

    if metrics.equity_curve and metrics.trades:
        generate_all_charts(metrics, per_asset, "AlphaLSTM", output_dir, timestamp)

    logger.info("\nAlpha LSTM backtest complete.")
    for key, value in results.items():
        if isinstance(value, float):
            logger.info(f"{key:<25}: {value:.4f}")
        else:
            logger.info(f"{key:<25}: {value}")


if __name__ == "__main__":
    main()

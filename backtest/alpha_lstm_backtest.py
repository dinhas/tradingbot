
import os
import sys
import json
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

        self.asset_obs = {
            asset: self.feature_engine.get_observation_vectorized(self.normalized_df, asset)
            for asset in self.assets
        }
        self.session_mask = self.normalized_df[self.session_col] == 1
        self.session_indices = np.where(self.session_mask.values)[0]

    def _open_position(self, asset, direction, entry_price, atr, timestamp):
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

        hold_minutes = max(0.0, (timestamp - pos['entry_timestamp']).total_seconds() / 60.0)
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

    def run(self, max_steps: int | None = None) -> BacktestMetrics:
        metrics = BacktestMetrics()

        n_steps = len(self.normalized_df)
        if n_steps < self.sequence_length + 1:
            logger.warning("Dataset too small for sequence length.")
            return metrics

        # Iterate through all available steps
        steps_run = 0
        for idx in range(self.sequence_length - 1, n_steps - 1):
            if max_steps is not None and steps_run >= max_steps:
                break
            
            next_idx = idx + 1
            seq_indices = range(idx - self.sequence_length + 1, idx + 1)
            ts = self.normalized_df.index[idx]
            next_ts = self.normalized_df.index[next_idx]
            steps_run += 1

            for asset in self.assets:
                entry_price = float(self.aligned_df.iloc[idx][f"{asset}_close"])
                atr = float(self.aligned_df.iloc[idx][f"{asset}_atr"])

                seq = self.asset_obs[asset][seq_indices]
                seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = self.model(seq_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                direction = int(np.argmax(probs) - 1)
                
                current_pos = self.positions[asset]

                if current_pos is None:
                    if direction != 0:
                        self._open_position(asset, direction, entry_price, atr, ts)
                else:
                    # Optional explicit close signal if model flips/neutral.
                    if direction == 0 or np.sign(direction) != np.sign(current_pos['direction']):
                        self._close_position(asset, entry_price, ts)
                        if direction != 0:
                            self._open_position(asset, direction, entry_price, atr, ts)

            # Advance one candle and evaluate SL/TP on next close.
            for asset, pos_data in list(self.positions.items()):
                if pos_data is None:
                    continue
                next_price = float(self.aligned_df.iloc[next_idx][f"{asset}_close"])
                if pos_data['direction'] == 1:
                    if next_price <= pos_data['sl']:
                        self._close_position(asset, pos_data['sl'], next_ts)
                    elif next_price >= pos_data['tp']:
                        self._close_position(asset, pos_data['tp'], next_ts)
                else:
                    if next_price >= pos_data['sl']:
                        self._close_position(asset, pos_data['sl'], next_ts)
                    elif next_price <= pos_data['tp']:
                        self._close_position(asset, pos_data['tp'], next_ts)

            while self.completed_trades:
                metrics.add_trade(self.completed_trades.pop(0))

            metrics.add_equity_point(next_ts, float(self.equity))

        # Force-close any remaining open positions at the end.
        final_idx = n_steps - 1
        final_ts = self.normalized_df.index[final_idx]
        for asset, pos_data in list(self.positions.items()):
            if pos_data is None:
                continue
            last_price = float(self.aligned_df.iloc[final_idx][f"{asset}_close"])
            self._close_position(asset, last_price, final_ts)

        while self.completed_trades:
            metrics.add_trade(self.completed_trades.pop(0))

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

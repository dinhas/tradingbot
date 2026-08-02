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
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from joblib import load as joblib_load

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.src.model import AlphaSLModel, AlphaSLModelV7, BUY
from Alpha.src.data_loader import DataLoader as AlphaDataLoader
from Alpha.src.calibration import apply_temperature, load_calibration
from Alpha.src.trade_simulator import TradeConfig, TradeSimulator
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine
from Alpha.src.labeling import Labeler
from Filter.src.feature_engine import FeatureEngine as FilterFeatureEngine
from backtest.rl_backtest import BacktestMetrics, generate_all_charts, NumpyEncoder
from shared_constants import FX_ALPHA_ASSETS, DEFAULT_SPREADS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_lstm_input_dim(state_dict: dict) -> int:
    # V7: VSN projection layer contains input dim
    if "vsn.feat_proj.weight" in state_dict:
        return int(state_dict["vsn.feat_proj.weight"].shape[1])
    key = "lstm.weight_ih_l0"
    if key not in state_dict:
        raise ValueError(f"Cannot infer input dimension from checkpoint.")
    return int(state_dict[key].shape[1])


def _load_state_dict(model_path: Path) -> dict:
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format in {model_path}")
    return checkpoint


class AlphaSLModelV3(nn.Module):
    def __init__(self, input_dim=14, lstm_units=64, dense_units=32,
                 dropout=0.3, num_assets=4, asset_embedding_dim=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_units,
                            num_layers=1, batch_first=True, bidirectional=False)
        self.attention_weights = nn.Linear(lstm_units, 1)
        self.asset_embedding = nn.Embedding(num_assets, asset_embedding_dim)
        self.fc1 = nn.Linear(lstm_units + asset_embedding_dim, dense_units)
        self.dropout = nn.Dropout(dropout)
        self.action_head = nn.Linear(dense_units, 2)

    def forward(self, x, asset_ids=None, return_dict=False):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention_weights(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)
        if asset_ids is None:
            asset_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        asset_context = self.asset_embedding(asset_ids.long())
        x = F.relu(self.fc1(torch.cat([context_vector, asset_context], dim=1)))
        x = self.dropout(x)
        action_logits = self.action_head(x)
        if return_dict:
            return {"action_logits": action_logits}
        return action_logits


class AlphaLSTMVectorizedBacktester:
    def __init__(
        self,
        model,
        aligned_df,
        normalized_df,
        sequence_length,
        confidence_thresh,
        initial_equity,
        position_size_pct,
        sl_mult,
        tp_mult,
        adx_thresh=25.0,
        max_hold_bars=12,
        leverage=100.0,
        batch_size=1024,
        calibration=None,
        format_version=4,
        use_labeler_filter=True,
        labeler_adx_threshold=15.0,
        labeler_min_edge_r=0.12,
        filter_rf_ensemble=None,
        filter_rf_threshold=0.565,
    ):
        self.model = model.to(DEVICE)
        self.aligned_df = aligned_df
        self.normalized_df = normalized_df
        self.sequence_length = sequence_length
        self.confidence_thresh = confidence_thresh
        self.initial_equity = initial_equity
        self.position_size_pct = position_size_pct
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.adx_thresh = adx_thresh
        self.max_hold_bars = max_hold_bars
        self.leverage = leverage
        self.batch_size = batch_size
        self.calibration = calibration or {}
        self.format_version = format_version
        self.use_labeler_filter = use_labeler_filter
        self.filter_rf_ensemble = filter_rf_ensemble
        self.filter_rf_threshold = filter_rf_threshold
        self.simulator = TradeSimulator(TradeConfig(
            tp_mult=tp_mult, sl_mult=sl_mult,
            max_hold_bars=max_hold_bars, leverage=leverage,
        ))

        self.assets = FX_ALPHA_ASSETS
        self.spreads = DEFAULT_SPREADS
        self.alpha_feature_engine = AlphaFeatureEngine()
        self.filter_feature_engine = FilterFeatureEngine() if filter_rf_ensemble is not None else None

        if self.use_labeler_filter:
            self.labeler = Labeler(
                tp_mult=tp_mult, sl_mult=sl_mult,
                max_bars=max_hold_bars,
                adx_threshold=labeler_adx_threshold,
                min_edge_r=labeler_min_edge_r,
            )

    def _precompute_predictions(self):
        logger.info(f"Precomputing predictions for {len(self.assets)} assets...")
        all_action_probs = {}
        all_filter_rf_prob = {}
        self.all_tradeable = {}

        n_steps = len(self.normalized_df)

        for asset_id, asset in enumerate(self.assets):
            logger.info(f"Predicting for {asset}...")
            obs = self.alpha_feature_engine.get_observation_vectorized(self.normalized_df, asset)

            sequences = np.lib.stride_tricks.sliding_window_view(
                obs, (self.sequence_length, obs.shape[1])
            ).squeeze(1)
            breaks = np.zeros(n_steps, dtype=np.int64)
            breaks[1:] = np.diff(self.normalized_df.index.to_numpy(dtype="datetime64[ns]")) != np.timedelta64(5, "m")
            segment_ids = np.cumsum(breaks)
            end_indices = np.arange(self.sequence_length - 1, n_steps)
            valid_sequences = segment_ids[end_indices] == segment_ids[end_indices - self.sequence_length + 1]

            asset_action_probs = np.zeros((n_steps, 3), dtype=np.float32)

            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(sequences), self.batch_size):
                    valid_batch = valid_sequences[i:i + self.batch_size]
                    if not valid_batch.any():
                        continue
                    batch_seq = sequences[i:i + self.batch_size]
                    batch_tensor = torch.from_numpy(batch_seq[valid_batch].copy()).to(DEVICE)
                    batch_assets = torch.full((int(valid_batch.sum()),), asset_id, dtype=torch.long, device=DEVICE)
                    outputs = self.model(batch_tensor, batch_assets, return_dict=True)
                    action_logits = outputs["action_logits"].float().cpu().numpy()
                    if self.format_version == 3:
                        sigmoids = 1.0 / (1.0 + np.exp(-action_logits))
                        short_probs = sigmoids[:, 0]
                        long_probs = sigmoids[:, 1]
                        hold_probs = np.maximum(1.0 - short_probs - long_probs, 0.0)
                        action_probs = np.column_stack([hold_probs, short_probs, long_probs])
                        row_sums = action_probs.sum(axis=1, keepdims=True)
                        row_sums = np.maximum(row_sums, 1e-8)
                        action_probs = action_probs / row_sums
                    elif self.format_version in (5, 7):
                        prob_buy = 1.0 / (1.0 + np.exp(-action_logits[:, 0]))
                        action_probs = np.column_stack([
                            np.zeros_like(prob_buy), 1.0 - prob_buy, prob_buy,
                        ])
                    elif self.calibration:
                        temperature = self.calibration.get("temperature", 1.0)
                        action_probs = apply_temperature(action_logits, temperature)
                    else:
                        action_probs = torch.softmax(outputs["action_logits"].float(), dim=1).cpu().numpy()

                    batch_end_indices = end_indices[i:i + self.batch_size][valid_batch]
                    asset_action_probs[batch_end_indices] = action_probs

            all_action_probs[asset] = asset_action_probs

            if self.filter_rf_ensemble is not None and self.filter_feature_engine is not None:
                logger.info(f"  Computing RF filter predictions for {asset}...")
                filter_obs = self.filter_feature_engine.get_observation_vectorized(self.normalized_df, asset)

                asset_rf_prob = np.zeros(n_steps, dtype=np.float32)
                rf1 = self.filter_rf_ensemble["rf1"]
                rf2 = self.filter_rf_ensemble["rf2"]
                gb = self.filter_rf_ensemble["gb"]
                meta = self.filter_rf_ensemble["meta"]

                for i in range(0, n_steps, self.batch_size):
                    batch_end = min(i + self.batch_size, n_steps)
                    batch_X = filter_obs[i:batch_end]

                    rf1_p = rf1.predict_proba(batch_X)[:, 1]
                    rf2_p = rf2.predict_proba(batch_X)[:, 1]
                    gb_p = gb.predict_proba(batch_X)[:, 1]
                    meta_input = np.column_stack([rf1_p, rf2_p, gb_p])
                    rf_prob = meta.predict_proba(meta_input)[:, 1]

                    asset_rf_prob[i:batch_end] = rf_prob

                all_filter_rf_prob[asset] = asset_rf_prob

            if self.use_labeler_filter:
                labels_df = self.labeler.label_data(self.aligned_df, asset)
                self.all_tradeable[asset] = labels_df['tradeable'].values.astype(np.float32)

        return all_action_probs, all_filter_rf_prob

    def run(self, max_steps=None):
        metrics = BacktestMetrics()

        all_action_probs, all_filter_rf_prob = self._precompute_predictions()

        n_steps = len(self.normalized_df)
        if max_steps:
            n_steps = min(n_steps, max_steps)

        equity = self.initial_equity
        positions = {asset: None for asset in self.assets}

        close_prices = {asset: self.aligned_df[f"{asset}_close"].values for asset in self.assets}
        open_prices = {asset: self.aligned_df[f"{asset}_open"].values for asset in self.assets}
        high_prices = {asset: self.aligned_df[f"{asset}_high"].values for asset in self.assets}
        low_prices = {asset: self.aligned_df[f"{asset}_low"].values for asset in self.assets}
        atrs = {asset: self.aligned_df[f"{asset}_atr"].values for asset in self.assets}
        timestamps = self.normalized_df.index

        use_rf_filter = self.filter_rf_ensemble is not None
        logger.info(f"Starting backtest (Labeler: {self.use_labeler_filter}, RF Filter: {use_rf_filter}, RF threshold: {self.filter_rf_threshold:.3f})...")
        metrics.add_equity_point(timestamps[self.sequence_length - 1], float(equity))

        for idx in tqdm(range(self.sequence_length - 1, n_steps - 1)):
            ts = timestamps[idx]

            for asset in self.assets:
                action_probs = all_action_probs[asset][idx]
                direction = 0

                if use_rf_filter:
                    rf_prob = all_filter_rf_prob[asset][idx]
                    if rf_prob < self.filter_rf_threshold:
                        continue

                if self.use_labeler_filter:
                    is_tradeable = self.all_tradeable[asset][idx] > 0.5
                    if is_tradeable:
                        action_idx = int(np.argmax(action_probs))
                        action_conf = action_probs[action_idx]
                        if action_conf > 0.65:
                            direction = -1 if action_idx == 1 else 1
                else:
                    action_idx = int(np.argmax(action_probs))
                    action_conf = action_probs[action_idx]
                    if action_conf > 0.65:
                        direction = -1 if action_idx == 1 else 1

                current_pos = positions[asset]
                mid_close = close_prices[asset][idx]
                atr = atrs[asset][idx]

                if current_pos is not None:
                    exit_price = None
                    reason = ""
                    p = current_pos
                    exit_price, reason = self.simulator.barrier_exit(
                        high_prices[asset][idx], low_prices[asset][idx], p['direction'],
                        self.spreads.get(asset, 0.0), p['sl'], p['tp']
                    )

                    if exit_price is None and (idx - p['entry_idx']) >= self.max_hold_bars - 1:
                        exit_price = self.simulator.market_exit_price(
                            mid_close, p['direction'], self.spreads.get(asset, 0.0)
                        )
                        reason = "Timeout"

                    if exit_price is not None:
                        gross_return, net_return, net_r, _raw = self.simulator.returns(
                            p['entry_price'], exit_price, p['direction'], p['atr']
                        )
                        pnl = gross_return * p['size']
                        fee = 2.0 * self.simulator.config.commission_rate_per_side * p['size']
                        net_pnl = net_return * p['size']
                        equity += net_pnl

                        metrics.add_trade({
                            'timestamp': ts,
                            'asset': asset,
                            'pnl': float(pnl),
                            'fees': float(fee),
                            'net_pnl': float(net_pnl),
                            'entry_price': float(p['entry_price']),
                            'exit_price': float(exit_price),
                            'size': float(p['size']),
                            'equity_before': float(p['equity_before']),
                            'equity_after': float(equity),
                            'hold_time': (ts - p['entry_timestamp']).total_seconds() / 60.0,
                            'reason': reason,
                            'net_r': float(net_r),
                        })
                        positions[asset] = None
                        current_pos = None

                if positions[asset] is None and direction != 0 and idx + 1 < n_steps:
                    size = self.position_size_pct * equity
                    entry_idx = idx + 1
                    entry_price = self.simulator.entry_price(
                        open_prices[asset][entry_idx], direction, self.spreads.get(asset, 0.0)
                    )
                    sl, tp = self.simulator.barriers(entry_price, direction, atr)

                    positions[asset] = {
                        'direction': direction,
                        'entry_price': entry_price,
                        'size': size,
                        'atr': atr,
                        'sl': sl,
                        'tp': tp,
                        'entry_timestamp': timestamps[entry_idx],
                        'entry_idx': entry_idx,
                        'equity_before': equity
                    }

            metrics.add_equity_point(ts, float(equity))

        final_idx = n_steps - 1
        for asset in self.assets:
            p = positions[asset]
            if p:
                exit_price = self.simulator.market_exit_price(
                    close_prices[asset][final_idx], p['direction'], self.spreads.get(asset, 0.0)
                )
                gross_return, net_return, net_r, _raw = self.simulator.returns(
                    p['entry_price'], exit_price, p['direction'], p['atr']
                )
                pnl = gross_return * p['size']
                fee = 2.0 * self.simulator.config.commission_rate_per_side * p['size']
                net_pnl = net_return * p['size']
                equity += net_pnl
                metrics.add_trade({
                    'timestamp': timestamps[final_idx],
                    'asset': asset,
                    'pnl': float(pnl),
                    'fees': float(fee),
                    'net_pnl': float(net_pnl),
                    'entry_price': float(p['entry_price']),
                    'exit_price': float(exit_price),
                    'size': float(p['size']),
                    'equity_before': float(p['equity_before']),
                    'equity_after': float(equity),
                    'hold_time': (timestamps[final_idx] - p['entry_timestamp']).total_seconds() / 60.0,
                    'reason': "End of Backtest",
                    'net_r': float(net_r),
                })

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Alpha LSTM + RF Filter Backtester")
    parser.add_argument("--model-path", type=str, default="Alpha/models/alpha_model.pth")
    parser.add_argument("--filter-rf-path", type=str, default=None)
    parser.add_argument("--filter-rf-threshold", type=float, default=0.565)
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--output-dir", type=str, default="backtest/results")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--confidence-thresh", type=float, default=0.55)
    parser.add_argument("--calibration-path", type=str, default=None)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--pos-size", type=float, default=0.1)
    parser.add_argument("--sl-mult", type=float, default=2.0)
    parser.add_argument("--tp-mult", type=float, default=4.0)
    parser.add_argument("--adx-thresh", type=float, default=15.0)
    parser.add_argument("--max-hold-bars", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--use-labeler-filter", action="store_true", default=True)
    parser.add_argument("--no-labeler-filter", action="store_true")
    parser.add_argument("--labeler-adx-threshold", type=float, default=15.0)
    parser.add_argument("--labeler-min-edge-r", type=float, default=0.12)
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model_path
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha_engine = AlphaFeatureEngine()
    loader = AlphaDataLoader(data_dir=str(data_dir))
    aligned_df, normalized_df = loader.get_features(engine=alpha_engine)

    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    checkpoint = _load_state_dict(model_path)
    fmt = checkpoint.get("format_version", 3)
    if fmt not in (3, 4, 5, 6, 7):
        logger.error("Expected V3/V4/V5/V6/V7 checkpoint, got version %d", fmt)
        return

    input_dim = len(alpha_engine.feature_names)
    model_config = checkpoint["model_config"]
    state_dict = checkpoint["model_state_dict"]
    checkpoint_input_dim = _infer_lstm_input_dim(state_dict)
    if checkpoint_input_dim != input_dim:
        logger.error("Model/input mismatch: checkpoint=%d, engine=%d.", checkpoint_input_dim, input_dim)
        return
    logger.info("Feature check passed: %d features, format v%d", checkpoint_input_dim, fmt)

    if fmt == 3:
        model = AlphaSLModelV3(**model_config)
        model.load_state_dict(state_dict)
    elif fmt == 7:
        model = AlphaSLModelV7(**model_config)
        model.load_state_dict(state_dict)
    else:
        if fmt >= 6 and checkpoint.get("feature_names") != alpha_engine.feature_names:
            logger.error("Feature names mismatch.")
            return
        model = AlphaSLModel(**model_config)
        model.load_state_dict(state_dict)
    model.eval()

    filter_rf_ensemble = None
    filter_rf_path = args.filter_rf_path
    if filter_rf_path is None:
        default_rf_path = PROJECT_ROOT / "Filter" / "models" / "filter_rf_ensemble.joblib"
        if default_rf_path.exists():
            filter_rf_path = str(default_rf_path)
    if filter_rf_path:
        filter_rf_path = PROJECT_ROOT / filter_rf_path if not Path(filter_rf_path).is_absolute() else Path(filter_rf_path)
        if filter_rf_path.exists():
            logger.info("Loading RF filter from %s (threshold=%.3f)", filter_rf_path, args.filter_rf_threshold)
            filter_rf_ensemble = joblib_load(filter_rf_path)
        else:
            logger.warning("RF filter not found at %s", filter_rf_path)

    use_labeler = not args.no_labeler_filter
    logger.info("Version: %d | Labeler: %s | RF Filter: %s", fmt, use_labeler,
                "ON" if filter_rf_ensemble else "OFF")

    calibration = load_calibration(PROJECT_ROOT / args.calibration_path) if args.calibration_path else None

    bt = AlphaLSTMVectorizedBacktester(
        model=model,
        aligned_df=aligned_df,
        normalized_df=normalized_df,
        sequence_length=25,
        confidence_thresh=args.confidence_thresh,
        initial_equity=args.initial_equity,
        position_size_pct=args.pos_size,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        adx_thresh=args.adx_thresh,
        max_hold_bars=args.max_hold_bars,
        batch_size=args.batch_size,
        calibration=calibration,
        format_version=fmt,
        use_labeler_filter=use_labeler,
        labeler_adx_threshold=args.labeler_adx_threshold,
        labeler_min_edge_r=args.labeler_min_edge_r,
        filter_rf_ensemble=filter_rf_ensemble,
        filter_rf_threshold=args.filter_rf_threshold,
    )

    start_time = datetime.now()
    metrics = bt.run(max_steps=args.steps)
    end_time = datetime.now()

    duration = (end_time - start_time).total_seconds()
    logger.info(f"Backtest completed in {duration:.2f}s")

    results = metrics.calculate_metrics()
    per_asset = metrics.get_per_asset_metrics()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics_file = output_dir / f"metrics_alpha_lstm_vectorized_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    if metrics.trades:
        trades_file = output_dir / f"trades_alpha_lstm_vectorized_{timestamp}.csv"
        pd.DataFrame(metrics.trades).to_csv(trades_file, index=False)

    if per_asset:
        asset_file = output_dir / f"asset_breakdown_alpha_lstm_vectorized_{timestamp}.csv"
        pd.DataFrame(per_asset).T.to_csv(asset_file)

    if metrics.equity_curve and metrics.trades and not args.no_charts:
        generate_all_charts(metrics, per_asset, "AlphaLSTM_Vectorized", output_dir, timestamp)

    logger.info("\n=== RESULTS ===")
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"{k:<25}: {v:.4f}")
        else:
            logger.info(f"{k:<25}: {v}")


if __name__ == "__main__":
    main()

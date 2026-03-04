import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
import logging
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)

from Alpha.src.model import AlphaSLModel
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine
from RiskLayer.src.frozen_alpha_env import TradingEnv

if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Alpha", "models", "alpha_model.pth")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_DATA_DIR_2025 = os.path.join(PROJECT_ROOT, "backtest", "data")
DEFAULT_OUTPUT_FILE = os.path.join(BASE_DIR, "data", "rl_risk_dataset.parquet")

LOOKAHEAD_BARS = 50
BATCH_SIZE = 5000

META_THRESHOLD = 0.78
QUAL_THRESHOLD = 0.30

SL_CHOICES = [0.5, 0.75, 1.0, 1.5, 1.75, 2.0, 2.5, 2.75, 3.0]
TP_MIN, TP_MAX = 1.0, 5.0
SIZE_MIN, SIZE_MAX = 0.10, 1.0

ASSET_BASE_SPREADS = {
    "EURUSD": 1.5,
    "GBPUSD": 1.8,
    "USDJPY": 2.0,
    "USDCHF": 1.5,
    "XAUUSD": 20.0,
}

SESSION_MULTIPLIERS = {"asian": 1.5, "london": 1.0, "ny": 1.0, "overlap": 0.8}


def get_dynamic_spread(asset, hour, atr, atr_ma):
    base_spread = ASSET_BASE_SPREADS.get(asset, 1.5)

    if 0 <= hour < 7:
        session = "asian"
    elif 7 <= hour < 12:
        session = "overlap"
    elif 12 <= hour < 17:
        session = "ny"
    else:
        session = "london" if 17 <= hour < 24 else "london"

    session_mult = SESSION_MULTIPLIERS.get(session, 1.0)

    vol_mult = 1.3 if atr > atr_ma * 1.2 else 1.0

    spread = base_spread * session_mult * vol_mult

    pip_value = 0.0001 if "JPY" not in asset else 0.01
    return spread * pip_value


class RLRiskDatasetGenerator:
    def __init__(self, model_path, data_dir):
        self.assets = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD"]

        logger.info("Loading Alpha model...")
        self.model = AlphaSLModel(input_dim=40)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        logger.info("Loading data via TradingEnv...")
        self.env = TradingEnv(data_dir=data_dir, stage=3, is_training=False)
        self.env.feature_engine = AlphaFeatureEngine()
        self.env.raw_data, self.env.processed_data = (
            self.env.feature_engine.preprocess_data(self.env.data)
        )
        self.env._cache_data_arrays()

        self.close_arrays = self.env.close_arrays
        self.high_arrays = self.env.high_arrays
        self.low_arrays = self.env.low_arrays
        self.atr_arrays = self.env.atr_arrays

        self.processed_data = self.env.processed_data

        self._compute_hour_and_atr_ma()

    def _compute_hour_and_atr_ma(self):
        self.hour_arrays = {}
        self.atr_ma_arrays = {}

        for asset in self.assets:
            hour_idx = self.env.raw_data[asset].index.hour.values
            self.hour_arrays[asset] = hour_idx.astype(np.float32)

            atr = self.atr_arrays[asset]
            atr_ma = np.zeros_like(atr)
            window = 20
            for i in range(window, len(atr)):
                atr_ma[i] = np.mean(atr[i - window : i])
            self.atr_ma_arrays[asset] = atr_ma

    def simulate_trade(self, asset, direction, entry_idx, sl_mult, tp_mult, size_pct):
        close = self.close_arrays[asset]
        high = self.high_arrays[asset]
        low = self.low_arrays[asset]
        atr = self.atr_arrays[asset]
        hour = self.hour_arrays[asset]
        atr_ma = self.atr_ma_arrays[asset]

        entry_price_mid = close[entry_idx]
        spread = get_dynamic_spread(
            asset, hour[entry_idx], atr[entry_idx], atr_ma[entry_idx]
        )

        if direction == 1:
            entry_price = entry_price_mid + spread
            sl_price = entry_price - (sl_mult * atr[entry_idx])
            tp_price = entry_price + (tp_mult * atr[entry_idx])
        else:
            entry_price = entry_price_mid - spread
            sl_price = entry_price + (sl_mult * atr[entry_idx])
            tp_price = entry_price - (tp_mult * atr[entry_idx])

        f_end = min(entry_idx + LOOKAHEAD_BARS, len(high))

        pnl = 0
        exit_reason = "timeout"

        for i in range(entry_idx + 1, f_end):
            if direction == 1:
                if high[i] >= tp_price:
                    pnl = (tp_price - entry_price) * size_pct
                    exit_reason = "tp"
                    break
                elif low[i] <= sl_price:
                    pnl = (sl_price - entry_price) * size_pct
                    exit_reason = "sl"
                    break
            else:
                if low[i] <= tp_price:
                    pnl = (entry_price - tp_price) * size_pct
                    exit_reason = "tp"
                    break
                elif high[i] >= sl_price:
                    pnl = (entry_price - sl_price) * size_pct
                    exit_reason = "sl"
                    break

        if exit_reason == "timeout":
            exit_price = close[f_end - 1]
            if direction == 1:
                pnl = (exit_price - entry_price) * size_pct
            else:
                pnl = (entry_price - exit_price) * size_pct

        return pnl, exit_reason

    def find_optimal_action(self, asset, direction, entry_idx, features):
        high = self.high_arrays[asset]
        low = self.low_arrays[asset]
        close = self.close_arrays[asset]
        atr = self.atr_arrays[asset]
        hour = self.hour_arrays[asset]
        atr_ma = self.atr_ma_arrays[asset]

        entry_price_mid = close[entry_idx]
        spread = get_dynamic_spread(
            asset, hour[entry_idx], atr[entry_idx], atr_ma[entry_idx]
        )

        if direction == 1:
            entry_price = entry_price_mid + spread
        else:
            entry_price = entry_price_mid - spread

        f_end = min(entry_idx + LOOKAHEAD_BARS, len(high))

        future_highs = high[entry_idx + 1 : f_end]
        future_lows = low[entry_idx + 1 : f_end]

        if direction == 1:
            mfe_idx = np.argmax(future_highs)
            mfe_dist = future_highs[mfe_idx] - entry_price
            mae_dist = entry_price - np.min(future_lows[: mfe_idx + 1])
        else:
            mfe_idx = np.argmin(future_lows)
            mfe_dist = entry_price - future_lows[mfe_idx]
            mae_dist = np.max(future_highs[: mfe_idx + 1]) - entry_price

        mfe_atr = mfe_dist / atr[entry_idx] if atr[entry_idx] > 0 else 0
        mae_atr = mae_dist / atr[entry_idx] if atr[entry_idx] > 0 else 0

        target_sl_mult = np.clip(mae_atr + 0.2, 0.2, 3.0)
        target_tp_mult = np.clip(mfe_atr, 1.0, 5.0)

        sl_idx = np.argmin(np.abs(np.array(SL_CHOICES) - target_sl_mult))

        target_size = 1.0 if (target_tp_mult / max(target_sl_mult, 0.1)) >= 1.5 else 0.5

        pnl, exit_reason = self.simulate_trade(
            asset, direction, entry_idx, SL_CHOICES[sl_idx], target_tp_mult, target_size
        )

        return sl_idx, target_tp_mult, target_size, pnl

    def generate_dataset(self, output_file, max_samples=None):
        total_rows = len(self.processed_data)
        start_idx = 500
        end_idx = total_rows - LOOKAHEAD_BARS

        logger.info(
            f"Processing from {start_idx} to {end_idx} ({end_idx - start_idx} rows)"
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        if os.path.exists(output_file):
            os.remove(output_file)

        total_signals = 0
        timestamps = self.processed_data.index[start_idx:end_idx]

        batch_starts = np.arange(0, end_idx - start_idx, BATCH_SIZE)

        for b_start in tqdm(batch_starts, desc="Generating dataset"):
            if max_samples and total_signals >= max_samples:
                break

            b_end = min(b_start + BATCH_SIZE, end_idx - start_idx)
            batch_df = self.processed_data.iloc[start_idx + b_start : start_idx + b_end]

            asset_predictions = {}
            asset_obs = {}

            for asset in self.assets:
                obs_batch = self.feature_engine.get_observation_vectorized(
                    batch_df, asset
                )
                asset_obs[asset] = obs_batch

                obs_tensor = torch.from_numpy(obs_batch)
                with torch.no_grad():
                    dir_logits, quality, meta_logits = self.model(obs_tensor)

                pred_direction = torch.argmax(dir_logits, dim=1).numpy() - 1
                pred_quality = torch.sigmoid(quality).numpy().flatten()
                pred_meta = torch.sigmoid(meta_logits).numpy().flatten()

                mask = (pred_meta > META_THRESHOLD) & (pred_quality >= QUAL_THRESHOLD)
                mask = mask & ((pred_direction > 0) | (pred_direction < 0))

                asset_predictions[asset] = {
                    "direction": pred_direction,
                    "quality": pred_quality,
                    "meta": pred_meta,
                    "mask": mask,
                }

            batch_results = {
                "timestamp": [],
                "asset": [],
                "direction": [],
                "entry_price": [],
                "atr": [],
                "features": [],
                "target_sl_idx": [],
                "target_tp_mult": [],
                "target_size": [],
                "pnl": [],
                "exit_reason": [],
            }

            for asset in self.assets:
                if max_samples and total_signals >= max_samples:
                    break

                preds = asset_predictions[asset]
                obs_batch = asset_obs[asset]
                mask = preds["mask"]

                if not np.any(mask):
                    continue

                active_local_indices = np.where(mask)[0]
                global_indices = start_idx + b_start + active_local_indices

                for local_idx, global_idx in zip(active_local_indices, global_indices):
                    if max_samples and total_signals >= max_samples:
                        break

                    direction = preds["direction"][local_idx]
                    quality = preds["quality"][local_idx]
                    meta = preds["meta"][local_idx]

                    entry_price = self.close_arrays[asset][global_idx]
                    atr = self.atr_arrays[asset][global_idx]
                    if atr == 0:
                        continue

                    features = obs_batch[local_idx]

                    sl_idx, tp_mult, size_pct, pnl = self.find_optimal_action(
                        asset, direction, global_idx, features
                    )

                    _, exit_reason = self.simulate_trade(
                        asset,
                        direction,
                        global_idx,
                        SL_CHOICES[sl_idx],
                        tp_mult,
                        size_pct,
                    )

                    batch_results["timestamp"].append(timestamps[b_start + local_idx])
                    batch_results["asset"].append(asset)
                    batch_results["direction"].append(int(direction))
                    batch_results["entry_price"].append(float(entry_price))
                    batch_results["atr"].append(float(atr))
                    batch_results["features"].append(features.tolist())
                    batch_results["target_sl_idx"].append(int(sl_idx))
                    batch_results["target_tp_mult"].append(float(tp_mult))
                    batch_results["target_size"].append(float(size_pct))
                    batch_results["pnl"].append(float(pnl))
                    batch_results["exit_reason"].append(exit_reason)

                    total_signals += 1

            if batch_results["timestamp"]:
                batch_df = pd.DataFrame(batch_results)
                if os.path.exists(output_file):
                    existing = pd.read_parquet(output_file)
                    combined = pd.concat([existing, batch_df], ignore_index=True)
                    combined.to_parquet(output_file, index=False)
                else:
                    batch_df.to_parquet(output_file, index=False)

            gc.collect()

        logger.info(
            f"Dataset generation complete: {total_signals} samples saved to {output_file}"
        )
        return total_signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    generator = RLRiskDatasetGenerator(args.model, args.data)

    generator.generate_dataset(args.output, args.max_samples)


if __name__ == "__main__":
    main()

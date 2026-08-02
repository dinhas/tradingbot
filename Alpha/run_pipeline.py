import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from datetime import datetime
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Alpha.src.data_loader import DataLoader as MyDataLoader
from Alpha.src.labeling import Labeler
from Alpha.src.model import (
    AlphaSLModel, AlphaSLModelV7, binary_signal_loss,
    HOLD, BUY, NUM_CLASSES,
)
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.diagnostics import (
    DiagnosticsRecorder, NumpyJSONEncoder,
    confidence_histogram, confidence_bucket_table, feature_label_scores, zip_run,
)

# Configure logging (LOG_FILE is bundled into the diagnostics zip at the end of the run)
LOG_FILE = f"alpha_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Enable cuDNN auto-tuner for fixed-size LSTM inputs (10-25% speedup)
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

# Use bfloat16 on Ampere+ GPUs for better numerical stability
AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
USE_AMP = DEVICE.type == "cuda"

# BEST-PRACTICE SETTINGS
SEQUENCE_LENGTH = 25
BATCH_SIZE = 256 if torch.cuda.device_count() >= 2 else 128
LEARNING_RATE = 1e-4
DROPOUT = 0.25        # reduced from 0.4 — 6 dropout sites compounded too aggressively
SESSION_COL = "is_late_session"

GRAD_CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.01
WARMUP_EPOCHS = 5     # reduced from 10 — AdamW adapts fast, long warmup wastes epochs
GRAD_ACCUM_STEPS = 4  # effective batch = BATCH_SIZE * 4

# Label / split hygiene
LABEL_MAX_BARS = 18      # must match Labeler.max_bars (vertical barrier)
BAR_MINUTES = 5
# Purge: drop train/val samples whose label horizon crosses the split boundary.
PURGE_TD = np.timedelta64(LABEL_MAX_BARS * BAR_MINUTES, 'm')
# Embargo: gap after each split cut so no input window or label horizon straddles it.
EMBARGO_TD = np.timedelta64((SEQUENCE_LENGTH + LABEL_MAX_BARS) * BAR_MINUTES, 'm')


class AlphaSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, action_targets: np.ndarray, asset_ids: np.ndarray):
        self.sequences = sequences
        # Pre-convert to tensors to avoid per-sample torch.tensor() calls
        self.action_targets = torch.from_numpy(action_targets.astype(np.int64))
        self.asset_ids = torch.from_numpy(asset_ids.astype(np.int64))

    def __len__(self):
        return len(self.action_targets)

    def __getitem__(self, idx):
        # .copy() needed for mmap-backed arrays
        x_seq = self.sequences[idx].copy()
        return (
            torch.from_numpy(x_seq),
            self.action_targets[idx],
            self.asset_ids[idx],
        )


def _date_based_split_indices(timestamps: np.ndarray, train_ratio: float = 0.70, val_ratio: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits samples by signal timestamp so future periods never leak into training.

    Applies purging (drops samples whose label horizon crosses a split cut) and an
    embargo (gap of SEQUENCE_LENGTH + LABEL_MAX_BARS bars after each cut) so that
    overlapping input windows and forward-looking labels never straddle a boundary.
    """
    if len(timestamps) == 0:
        raise ValueError("Cannot split an empty timestamp array.")
    if not 0 < train_ratio < 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.")

    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    unique_ts = np.unique(ts)
    if len(unique_ts) < 3:
        raise ValueError("Need at least 3 unique timestamps for train/val/test splitting.")

    train_cut = unique_ts[max(0, min(len(unique_ts) - 2, int(len(unique_ts) * train_ratio) - 1))]
    val_cut = unique_ts[max(1, min(len(unique_ts) - 1, int(len(unique_ts) * (train_ratio + val_ratio)) - 1))]

    # Purge the tail of train/val (label horizon would peek past the cut),
    # embargo the head of val/test (input windows would reach back past the cut).
    train_idx = np.flatnonzero(ts <= train_cut - PURGE_TD)
    val_idx = np.flatnonzero((ts > train_cut + EMBARGO_TD) & (ts <= val_cut - PURGE_TD))
    test_idx = np.flatnonzero(ts > val_cut + EMBARGO_TD)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Date split produced an empty train, validation, or test partition.")

    return train_idx, val_idx, test_idx


def _build_sequences_for_asset(features: np.ndarray, action_targets: np.ndarray,
                               action_net_r: np.ndarray, valid_mask: np.ndarray,
                               timestamps: np.ndarray, seq_len: int) -> tuple:
    """Builds CONTIGUOUS rolling windows for one asset.

    Windows are always built over adjacent bars (matching live/backtest inference).
    The valid_mask only selects WHICH windows are kept (by their final bar), it never
    removes bars from inside a window.
    """
    empty = (
        np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        np.empty((0, 2), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )
    if len(features) < seq_len:
        return empty

    end_indices = np.flatnonzero(valid_mask)
    end_indices = end_indices[end_indices >= seq_len - 1].astype(np.int64)
    ts = np.asarray(timestamps, dtype="datetime64[ns]")
    breaks = np.zeros(len(ts), dtype=np.int64)
    breaks[1:] = np.diff(ts) != np.timedelta64(BAR_MINUTES, "m")
    segment_ids = np.cumsum(breaks)
    contiguous = segment_ids[end_indices] == segment_ids[end_indices - seq_len + 1]
    end_indices = end_indices[contiguous]
    if len(end_indices) == 0:
        return empty

    X_seq = np.stack([features[e - seq_len + 1:e + 1] for e in end_indices]).astype(np.float32)
    y_actions = action_targets[end_indices].astype(np.int64)
    y_action_net_r = action_net_r[end_indices].astype(np.float32)

    return X_seq, y_actions, y_action_net_r, end_indices



def generate_dataset(data_dir, output_dir, smoke_test=False, seq_len=SEQUENCE_LENGTH, max_rows=0, exclude_assets=None):
    logger.info(f"Generating session-only dataset from {data_dir}...")
    loader = MyDataLoader(data_dir=data_dir)
    labeler = Labeler()
    engine = FeatureEngine()

    aligned_df, normalized_df = loader.get_features(engine=engine, max_rows=max_rows)

    if exclude_assets:
        exclude_set = set(a.upper() for a in exclude_assets)
        loader.assets = [a for a in loader.assets if a.upper() not in exclude_set]
        logger.info(f"Excluded assets: {exclude_set}. Using: {loader.assets}")

    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "_tmp_sequences")
    os.makedirs(temp_dir, exist_ok=True)

    sequence_parts = []
    all_tradeable = []
    all_directions = []
    all_net_r = []
    all_raw_return = []
    all_action_classes = []
    all_action_net_r = []
    all_timestamps = []
    all_asset_ids = []
    all_local_indices = []
    total_rows = 0
    asset_stats = {}
    input_dim = None

    for asset_id, asset in enumerate(loader.assets):
        logger.info(f"Processing {asset}...")
        labels_df = labeler.label_data(aligned_df, asset)
        if smoke_test:
            labels_df = labels_df.head(5000)

        common_indices = labels_df.index.intersection(normalized_df.index)
        if len(common_indices) == 0:
            logger.warning(f"No overlapping rows for {asset}; skipping.")
            continue

        filtered_norm_df = normalized_df.loc[common_indices]
        filtered_labels_df = labels_df.loc[common_indices]

        if filtered_norm_df.empty:
            logger.warning(f"No rows after filtering for {asset}; skipping.")
            continue

        asset_X = engine.get_observation_vectorized(filtered_norm_df, asset)
        asset_tradeable = filtered_labels_df['tradeable'].values.astype(np.float32)
        asset_direction = filtered_labels_df['direction'].values.astype(np.int64)
        asset_net_r = filtered_labels_df['net_r'].values.astype(np.float32)
        asset_raw_return = filtered_labels_df['raw_return'].values.astype(np.float32)
        asset_action_class = filtered_labels_df['action_class'].values.astype(np.int64)
        asset_action_net_r = filtered_labels_df[['short_net_r', 'long_net_r']].values.astype(np.float32)
        asset_valid = filtered_labels_df['valid'].values.astype(bool)
        asset_timestamps = common_indices.to_numpy(dtype="datetime64[ns]")

        X_seq, y_actions, y_action_net_r, end_indices = _build_sequences_for_asset(
            asset_X, asset_action_class, asset_action_net_r, asset_valid,
            asset_timestamps, seq_len
        )
        if len(y_actions) == 0:
            continue

        if input_dim is None:
            input_dim = int(X_seq.shape[-1])
        part_path = os.path.join(temp_dir, f"{asset}_sequences.npy")
        np.save(part_path, X_seq.astype(np.float32, copy=False))
        sequence_parts.append((asset_id, asset, part_path))
        del X_seq

        y_trade = asset_tradeable[end_indices]
        y_dir = asset_direction[end_indices]
        y_net_r = asset_net_r[end_indices]
        y_raw_r = asset_raw_return[end_indices]
        all_tradeable.append(y_trade)
        all_directions.append(y_dir)
        all_net_r.append(y_net_r)
        all_raw_return.append(y_raw_r)
        all_action_classes.append(y_actions)
        all_action_net_r.append(y_action_net_r)
        all_timestamps.append(asset_timestamps[end_indices])
        all_asset_ids.append(np.full(len(y_actions), asset_id, dtype=np.int8))
        all_local_indices.append(np.arange(len(y_actions), dtype=np.int32))
        asset_total = len(y_actions)

        # Per-asset dataset diagnostics
        cls_cls, cls_cnt = np.unique(y_actions, return_counts=True)
        asset_stats[asset] = {
            "sequences": int(asset_total),
            "valid_ratio": round(float(filtered_labels_df['valid'].mean()), 4),
            "tradeable_count": int(y_trade.sum()),
            "tradeable_rate": round(float(y_trade.mean()), 4),
            "action_class_counts": {str(int(c)): int(n) for c, n in zip(cls_cls, cls_cnt)},
            "net_r_mean": round(float(y_net_r.mean()), 4),
            "net_r_p90": round(float(np.percentile(y_net_r, 90)), 4),
        }

        total_rows += asset_total
        logger.info(f"{asset}: generated {asset_total} sequences.")

    if not sequence_parts:
        raise RuntimeError("No training sequences were generated. Check data quality and session filters.")

    y_trade_np = np.concatenate(all_tradeable, axis=0).astype(np.float32)
    y_dir_np = np.concatenate(all_directions, axis=0).astype(np.int64)
    y_net_r_np = np.concatenate(all_net_r, axis=0).astype(np.float32)
    y_raw_r_np = np.concatenate(all_raw_return, axis=0).astype(np.float32)
    action_classes_np = np.concatenate(all_action_classes, axis=0).astype(np.int64)
    action_net_r_np = np.concatenate(all_action_net_r, axis=0).astype(np.float32)
    timestamps_np = np.concatenate(all_timestamps, axis=0).astype("datetime64[ns]")
    asset_ids_np = np.concatenate(all_asset_ids, axis=0).astype(np.int8)
    local_indices_np = np.concatenate(all_local_indices, axis=0).astype(np.int32)

    order = np.argsort(timestamps_np, kind="mergesort")
    y_trade_np = y_trade_np[order]
    y_dir_np = y_dir_np[order]
    y_net_r_np = y_net_r_np[order]
    y_raw_r_np = y_raw_r_np[order]
    action_classes_np = action_classes_np[order]
    action_net_r_np = action_net_r_np[order]
    timestamps_np = timestamps_np[order]
    asset_ids_np = asset_ids_np[order]
    local_indices_np = local_indices_np[order]

    # From labeler: 0=hold, 1=short(sell), 2=long(buy)
    n_total = len(action_classes_np)
    
    # Filter to only buy and sell trades (exclude hold)
    trade_mask = action_classes_np != 0  # Keep only buy (2) and sell (1)
    trade_indices = np.where(trade_mask)[0]
    
    if len(trade_indices) == 0:
        raise RuntimeError("No buy or sell trades found in dataset. Check labeler configuration.")
    
    # Filter sequences and labels to trade-only
    action_classes_filtered = action_classes_np[trade_mask]
    timestamps_filtered = timestamps_np[trade_mask]
    asset_ids_filtered = asset_ids_np[trade_mask]
    y_trade_filtered = y_trade_np[trade_mask]
    y_dir_filtered = y_dir_np[trade_mask]
    y_net_r_filtered = y_net_r_np[trade_mask]
    y_raw_r_filtered = y_raw_r_np[trade_mask]
    action_net_r_filtered = action_net_r_np[trade_mask]
    local_indices_filtered = local_indices_np[trade_mask]
    
    # Save labels: 1 if long(buy), 0 if short(sell) — binary buy vs sell
    labels = np.where(action_classes_filtered == 2, 1, 0).astype(np.int64)

    logger.info("Trade-only dataset: %d sequences (filtered from %d total)", len(trade_indices), n_total)
    logger.info("Labels: buy=%d, sell=%d", int(labels.sum()), int((labels == 0).sum()))

    sequences_path = os.path.join(output_dir, "sequences.npy")
    labels_path = os.path.join(output_dir, "labels.npz")

    logger.info("Writing disk-backed sequence matrix to %s", sequences_path)
    X_out = np.lib.format.open_memmap(
        sequences_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(labels), seq_len, int(input_dim)),
    )
    # Write filtered sequences (trade-only)
    # For each asset, find which sequences are in the filtered set and copy them
    for asset_id, asset, part_path in sequence_parts:
        src = np.load(part_path, mmap_mode="r")
        # Get all positions for this asset in the original dataset
        original_asset_positions = np.where(asset_ids_np == asset_id)[0]
        # Get positions for this asset in the filtered dataset
        filtered_asset_positions = np.where(asset_ids_filtered == asset_id)[0]
        
        # For each filtered position, find the corresponding original local index
        for filtered_pos in filtered_asset_positions:
            # Get the local index in the original asset's data
            local_idx = local_indices_filtered[filtered_pos]
            # The original position in the full dataset
            orig_pos = original_asset_positions[local_idx]
            # Copy the sequence from the part file to X_out
            X_out[filtered_pos] = src[local_idx]
        
        logger.info("Wrote %d %s sequences.", len(filtered_asset_positions), asset)
    X_out.flush()

    np.savez(
        labels_path,
        tradeable=y_trade_filtered,
        direction=y_dir_filtered,
        net_r=y_net_r_filtered,
        raw_return=y_raw_r_filtered,
        labels=labels,
        action_classes=action_classes_filtered,
        action_net_r=action_net_r_filtered,
        timestamp=timestamps_filtered,
        asset_id=asset_ids_filtered,
        asset_names=np.asarray(loader.assets),
        sequence_length=np.int32(seq_len),
    )

    # --- Dataset diagnostics: is this target learnable? ---
    logger.info("Computing dataset diagnostics...")
    cls_cls, cls_cnt = np.unique(action_classes_filtered, return_counts=True)

    # Label distribution per month: a moving target here explains unstable training
    ts_index = pd.DatetimeIndex(timestamps_filtered)
    monthly = pd.crosstab(ts_index.to_period('M').astype(str), action_classes_filtered)
    monthly_dist = {
        str(month): {str(int(c)): int(v) for c, v in row.items()}
        for month, row in monthly.iterrows()
    }

    # Feature-label ANOVA F-scores on the LAST bar of each sequence.
    # If nothing scores above noise, the features (not the model) are the problem.
    f_scores = feature_label_scores(np.asarray(X_out[:, -1, :]), labels, engine.feature_names)

    dataset_stats = {
        "generated_at": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "sequence_length": int(seq_len),
        "smoke_test": bool(smoke_test),
        "total_sequences": int(len(labels)),
        "total_before_filter": int(n_total),
        "filtered_out_hold": int(n_total - len(labels)),
        "action_class_counts": {str(int(c)): int(n) for c, n in zip(cls_cls, cls_cnt)},
        "net_r_mean": round(float(y_net_r_filtered.mean()), 4),
        "net_r_p90": round(float(np.percentile(y_net_r_filtered, 90)), 4),
        "assets": asset_stats,
        "monthly_class_distribution": monthly_dist,
        "feature_label_f_scores": f_scores,
    }
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(dataset_stats, f, indent=2, cls=NumpyJSONEncoder)
    logger.info(f"Dataset diagnostics saved to {stats_path}")
    logger.info(f"Top-5 feature F-scores: {dict(list(f_scores.items())[:5])}")

    # Feature selection: zero out noise features (F < 1.0)
    MIN_F_SCORE = 1.0
    feature_mask = np.array([f_scores.get(name, 0.0) >= MIN_F_SCORE for name in engine.feature_names])
    n_kept = int(feature_mask.sum())
    n_dropped = int((~feature_mask).sum())
    if n_dropped > 0:
        dropped_names = [engine.feature_names[i] for i in range(len(feature_mask)) if not feature_mask[i]]
        logger.info("Feature selection: keeping %d, dropping %d (F < %.1f): %s",
                     n_kept, n_dropped, MIN_F_SCORE, dropped_names)
        # Apply mask to the saved sequences
        X_out[:, :, ~feature_mask] = 0.0
    else:
        logger.info("Feature selection: all %d features kept (F >= %.1f)", n_kept, MIN_F_SCORE)

    # Save feature mask for inference
    mask_path = os.path.join(output_dir, "feature_mask.npy")
    np.save(mask_path, feature_mask)

    for _, _, part_path in sequence_parts:
        try:
            os.remove(part_path)
        except OSError:
            logger.warning("Failed to remove temporary sequence part: %s", part_path)
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    logger.info(f"Dataset generated. Total sequences: {len(y_trade_np)}")
    return sequences_path, labels_path


def _binary_metrics(targets: np.ndarray, probs: np.ndarray) -> dict:
    """Per-class precision/recall for binary (sell=0, buy=1)."""
    preds = (probs > 0.5).astype(np.int64)
    results = {}
    class_names = ["sell", "buy"]
    for cls_idx, cls_name in enumerate(class_names):
        tp = int(((preds == cls_idx) & (targets == cls_idx)).sum())
        fp = int(((preds == cls_idx) & (targets != cls_idx)).sum())
        fn = int(((preds != cls_idx) & (targets == cls_idx)).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        base_rate = float((targets == cls_idx).mean()) if len(targets) else 0.0
        results[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
            "base_rate": round(base_rate, 4),
            "edge": round(precision - base_rate, 4),
        }
    results["accuracy"] = round(float((preds == targets).mean()), 4)
    return results


def _binary_policy_metrics(targets: np.ndarray, probs: np.ndarray,
                           threshold: float = 0.5) -> dict:
    """Evaluate trading policy: buy if prob_buy >= threshold, else hold."""
    preds = (probs >= threshold).astype(np.int64)
    correct = int((preds == targets).sum())
    return {
        "threshold": threshold,
        "trades": len(targets),
        "correct": correct,
        "accuracy": round(correct / max(1, len(targets)), 4),
        "coverage": 1.0,
    }


def _evaluate_holdout(model, sequences, action_classes, asset_ids, test_idx, batch_size=512) -> dict:
    """Evaluates binary sell/buy on untouched data."""
    model.eval()
    action_probs_list = []
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            idx = test_idx[i:i + batch_size]
            batch = np.asarray(sequences[idx], dtype=np.float32)
            outputs = model(
                torch.from_numpy(batch).to(DEVICE),
                torch.from_numpy(asset_ids[idx].astype(np.int64)).to(DEVICE),
                return_dict=True,
            )
            prob_buy = torch.sigmoid(outputs["action_logits"].float()).squeeze(-1).cpu().numpy()
            action_probs_list.append(prob_buy)

    action_probs = np.concatenate(action_probs_list)
    targets = action_classes[test_idx].astype(np.int64)
    policy = _binary_policy_metrics(targets, action_probs)
    logger.info("Holdout policy accuracy: %s over %d trades",
                policy["accuracy"], policy["trades"])

    return {
        "n_samples": int(len(test_idx)),
        "class_metrics": _binary_metrics(targets, action_probs),
        "selected_action": policy,
        "confidence_histogram": confidence_histogram(np.maximum(action_probs, 1 - action_probs)),
    }


def train_model(sequences_path, labels_path, model_save_path, max_epochs=200,
                model_version=7):
    """Train a single buy-vs-sell model."""
    logger.info("Starting alpha model training...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Diagnostics: one directory per run, zipped automatically at the end
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    diag_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(model_save_path))), "diagnostics", f"run_{run_ts}")
    recorder = DiagnosticsRecorder(diag_dir)

    # Bundle the dataset-generation stats if present
    dataset_stats_path = os.path.join(os.path.dirname(labels_path), "dataset_stats.json")
    if os.path.exists(dataset_stats_path):
        with open(dataset_stats_path) as f:
            recorder.set_dataset_stats(json.load(f))

    sequences = np.load(sequences_path, mmap_mode='r')
    labels_data = np.load(labels_path)
    if 'labels' not in labels_data:
        raise RuntimeError("labels.npz is missing 'labels'. Regenerate the dataset.")
    if 'asset_id' not in labels_data or 'timestamp' not in labels_data:
        raise RuntimeError("labels.npz is missing asset_id or timestamp. Regenerate the dataset.")

    action_classes = labels_data['labels'].astype(np.int64)
    asset_ids = labels_data['asset_id'].astype(np.int64)
    timestamps = labels_data['timestamp']

    cls_cls, cls_cnt = np.unique(action_classes, return_counts=True)
    pos_count = int((action_classes == 1).sum())
    neg_count = int((action_classes == 0).sum())
    logger.info("Labels: buy=%d, sell=%d (%.2f%% buy rate)",
                pos_count, neg_count, 100.0 * pos_count / max(1, len(action_classes)))

    total_samples = len(sequences)
    if total_samples == 0:
        raise RuntimeError("No sequences found for training.")

    train_idx, val_idx, test_idx = _date_based_split_indices(timestamps)
    X_train, X_val = sequences[train_idx], sequences[val_idx]
    y_train, y_val = action_classes[train_idx], action_classes[val_idx]
    asset_train, asset_val = asset_ids[train_idx], asset_ids[val_idx]

    # Class weights for BCE: inverse frequency (0=sell, 1=buy)
    class_counts = np.bincount(y_train, minlength=2).astype(np.float64)
    class_counts = np.maximum(class_counts, 1.0)
    total_train = class_counts.sum()
    # pos_weight = negative_count / positive_count for BCE pos_weight
    pos_weight_val = class_counts[0] / max(class_counts[1], 1.0)
    pos_weight_val = float(np.clip(pos_weight_val, 1.0, 10.0))
    logger.info("Class distribution: sell=%d, buy=%d | pos_weight=%.2f",
                int(class_counts[0]), int(class_counts[1]), pos_weight_val)
    logger.info(
        "Date split: train=%d (%s to %s), val=%d (%s to %s), holdout_test=%d (%s to %s)",
        len(train_idx), np.asarray(timestamps[train_idx[0]]).astype(str), np.asarray(timestamps[train_idx[-1]]).astype(str),
        len(val_idx), np.asarray(timestamps[val_idx[0]]).astype(str), np.asarray(timestamps[val_idx[-1]]).astype(str),
        len(test_idx), np.asarray(timestamps[test_idx[0]]).astype(str), np.asarray(timestamps[test_idx[-1]]).astype(str),
    )

    train_dataset = AlphaSequenceDataset(X_train, y_train, asset_train)
    val_dataset = AlphaSequenceDataset(X_val, y_val, asset_val)

    # Kaggle Linux: use workers for parallel data loading
    num_workers = 4 if os.name != 'nt' else 0
    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    input_dim = X_train.shape[-1]
    if model_version == 7:
        model = AlphaSLModelV7(
            input_dim=input_dim, lstm_units=128, dense_units=128,
            dropout=DROPOUT, num_layers=3, num_heads=4, bidirectional=True,
        ).to(DEVICE)
    else:
        model = AlphaSLModel(
            input_dim=input_dim, lstm_units=128, dense_units=128,
            dropout=DROPOUT, num_layers=3, num_heads=4, bidirectional=True,
        ).to(DEVICE)

    if N_GPUS > 1:
        logger.info("Using DataParallel on %d GPUs: %s", N_GPUS,
                     [torch.cuda.get_device_name(i) for i in range(N_GPUS)])
        model = torch.nn.DataParallel(model)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    # OneCycleLR: handles warmup + cosine decay in one schedule
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE * 3,  # peak LR is 3x base
        epochs=max_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    class_weights_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)

    recorder.set_config(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT,
        input_dim=int(input_dim),
        lstm_units=128,
        dense_units=128,
        num_layers=3,
        num_heads=4,
        bidirectional=True,
        weight_decay=1e-3,
        device=str(DEVICE),
        label_smoothing=LABEL_SMOOTHING,
        pos_weight=pos_weight_val,
        class_counts={"sell": int(class_counts[0]), "buy": int(class_counts[1])},
        num_assets=int(asset_ids.max()) + 1,
        asset_embedding_dim=4,
        grad_clip_norm=GRAD_CLIP_NORM,
        amp_enabled=DEVICE.type == "cuda",
        warmup_epochs=WARMUP_EPOCHS,
        split={
            "train": {"n": int(len(train_idx)),
                      "from": str(np.asarray(timestamps[train_idx[0]]).astype(str)),
                      "to": str(np.asarray(timestamps[train_idx[-1]]).astype(str))},
            "val": {"n": int(len(val_idx)),
                    "from": str(np.asarray(timestamps[val_idx[0]]).astype(str)),
                    "to": str(np.asarray(timestamps[val_idx[-1]]).astype(str))},
            "test": {"n": int(len(test_idx)),
                     "from": str(np.asarray(timestamps[test_idx[0]]).astype(str)),
                     "to": str(np.asarray(timestamps[test_idx[-1]]).astype(str))},
            "purge_minutes": int(LABEL_MAX_BARS * BAR_MINUTES),
            "embargo_minutes": int((SEQUENCE_LENGTH + LABEL_MAX_BARS) * BAR_MINUTES),
        },
        sequences_path=str(sequences_path),
        labels_path=str(labels_path),
        model_save_path=str(model_save_path),
    )

    best_val_loss = float('inf')
    early_stop_patience = 15
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        train_loss_sum = 0.0
        train_batches = 0
        grad_norm_sum = 0.0

        optimizer.zero_grad(set_to_none=True)
        for step, (b_X, b_actions, b_assets) in enumerate(pbar):
            b_X = b_X.to(DEVICE, non_blocking=True)
            b_actions = b_actions.to(DEVICE, non_blocking=True)
            b_assets = b_assets.to(DEVICE, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                outputs = model(b_X, b_assets, return_dict=True)
                loss = binary_signal_loss(outputs, b_actions, label_smoothing=LABEL_SMOOTHING,
                                         class_weights=class_weights_tensor)
            loss = loss / GRAD_ACCUM_STEPS

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}: {loss.item()}")

            scaler.scale(loss).backward()

            # Step optimizer every ACCUM_STEPS batches
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
                if not torch.isfinite(grad_norm):
                    logger.warning(f"Non-finite gradient norm at epoch {epoch + 1}, skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            else:
                grad_norm = torch.tensor(0.0)

            train_loss_sum += loss.item() * GRAD_ACCUM_STEPS
            grad_norm_sum += float(grad_norm)
            train_batches += 1

        avg_train_loss = train_loss_sum / max(1, train_batches)
        avg_grad_norm = grad_norm_sum / max(1, train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_action_probs_list = []
        val_action_targets_list = []
        with torch.no_grad():
            for b_X, b_actions, b_assets in val_loader:
                b_X = b_X.to(DEVICE, non_blocking=True)
                b_actions = b_actions.to(DEVICE, non_blocking=True)
                b_assets = b_assets.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=USE_AMP):
                    outputs = model(b_X, b_assets, return_dict=True)
                    loss = binary_signal_loss(outputs, b_actions, label_smoothing=LABEL_SMOOTHING,
                                              class_weights=class_weights_tensor)
                val_loss += loss.item()
                val_action_probs_list.append(torch.sigmoid(outputs["action_logits"].float()).squeeze(-1).cpu().numpy())
                val_action_targets_list.append(b_actions.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))

        val_action_probs = np.concatenate(val_action_probs_list)
        val_action_targets = np.concatenate(val_action_targets_list)
        val_policy = _binary_policy_metrics(val_action_targets, val_action_probs)
        val_multiclass = _binary_metrics(val_action_targets, val_action_probs)

        recorder.log_epoch(
            epoch=epoch + 1,
            train_loss=round(avg_train_loss, 5),
            val_loss=round(avg_val_loss, 5),
            lr=optimizer.param_groups[0]['lr'],
            grad_norm=round(avg_grad_norm, 5),
            val_class_metrics=val_multiclass,
            val_selected_action=val_policy,
            val_confidence_histogram=confidence_histogram(np.maximum(val_action_probs, 1 - val_action_probs)),
        )
        logger.info(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | "
            f"Val Acc = {val_multiclass['accuracy']} | Policy Trades = {val_policy['trades']} | "
            f"Grad Norm = {avg_grad_norm:.3f}"
        )

        if avg_val_loss < best_val_loss:
            logger.info(f"New best Val Loss ({avg_val_loss:.4f})! Saving model to {model_save_path}")
            best_val_loss = avg_val_loss
            torch.save({
                "format_version": model_version,
                "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                "model_config": {
                    "input_dim": int(input_dim), "lstm_units": 128, "dense_units": 128,
                    "dropout": DROPOUT, "num_assets": int(asset_ids.max()) + 1,
                    "asset_embedding_dim": 4, "num_layers": 3, "num_heads": 4,
                    "bidirectional": True,
                },
                "feature_names": FeatureEngine().feature_names,
                "sequence_length": SEQUENCE_LENGTH,
                "asset_names": labels_data['asset_names'].tolist(),
                "output_semantics": ["sell", "buy"],
                "trade_only": True,
            }, model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info("Early stopping triggered.")
                break

    # Final report card on the untouched holdout period
    if os.path.exists(model_save_path):
        checkpoint = torch.load(model_save_path, map_location=DEVICE)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Evaluating best model on holdout test period...")
    holdout_report = _evaluate_holdout(model, sequences, action_classes, asset_ids, test_idx)
    recorder.set_holdout(holdout_report)
    recorder.set_config(best_val_loss=round(best_val_loss, 5), epochs_trained=len(recorder.report["training"]["epochs"]))

    # Persist diagnostics and zip everything (report + curves + run log) into one archive < 10 MB
    recorder.save()
    zip_path, size_mb = zip_run(diag_dir, extra_files=[LOG_FILE])
    logger.info(f"Diagnostics bundle ready: {zip_path} ({size_mb:.2f} MB)")

    logger.info("Training complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--gen-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max raw rows per asset before feature computation (0=all)")
    parser.add_argument("--exclude-assets", type=str, nargs="*", default=None,
                        help="Assets to exclude from training (e.g. --exclude-assets USDCHF)")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Max training epochs (default: 200)")
    parser.add_argument("--version", type=int, default=7, choices=[6, 7],
                        help="Model version: 6=current LSTM+attention, 7=V7 with GRN+VSN+aux (default: 7)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, args.data_dir))
    dataset_dir = os.path.join(base_dir, "data", "training_set")

    if not args.skip_gen:
        generate_dataset(data_dir, dataset_dir, smoke_test=args.smoke_test,
                         max_rows=args.max_samples, exclude_assets=args.exclude_assets)

    if args.gen_only:
        return

    seq_path = os.path.join(dataset_dir, "sequences.npy")
    lbl_path = os.path.join(dataset_dir, "labels.npz")

    model_path = os.path.join(base_dir, "models", "alpha_model.pth")
    logger.info("=" * 60)
    logger.info("TRAINING ALPHA MODEL")
    logger.info("=" * 60)
    train_model(
        seq_path, lbl_path, model_path,
        max_epochs=args.max_epochs,
        model_version=args.version,
    )

    logger.info("Model trained: %s", model_path)


if __name__ == "__main__":
    main()

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
from Alpha.src.model import AlphaSLModel, trade_direction_loss
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

# BEST-PRACTICE SETTINGS
SEQUENCE_LENGTH = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
DROPOUT = 0.3
SESSION_COL = "is_late_session"
TRADE_LOSS_WEIGHT = 0.65
DIRECTION_LOSS_WEIGHT = 0.35
GRAD_CLIP_NORM = 1.0

# Label / split hygiene
LABEL_MAX_BARS = 6      # must match Labeler.max_bars (vertical barrier)
BAR_MINUTES = 5
# Purge: drop train/val samples whose label horizon crosses the split boundary.
PURGE_TD = np.timedelta64(LABEL_MAX_BARS * BAR_MINUTES, 'm')
# Embargo: gap after each split cut so no input window or label horizon straddles it.
EMBARGO_TD = np.timedelta64((SEQUENCE_LENGTH + LABEL_MAX_BARS) * BAR_MINUTES, 'm')


class AlphaSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, tradeable: np.ndarray, directions: np.ndarray):
        self.sequences = sequences
        self.tradeable = tradeable
        self.directions = directions

    def __len__(self):
        return len(self.tradeable)

    def __getitem__(self, idx):
        x_seq = self.sequences[idx].copy()
        y_trade = self.tradeable[idx]
        y_dir = self.directions[idx]
        return torch.from_numpy(x_seq), torch.tensor(y_trade, dtype=torch.float32), torch.tensor(y_dir, dtype=torch.long)


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


def _build_sequences_for_asset(features: np.ndarray, tradeable: np.ndarray, directions: np.ndarray,
                               net_r: np.ndarray, valid_mask: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Builds CONTIGUOUS rolling windows for one asset.

    Windows are always built over adjacent bars (matching live/backtest inference).
    The valid_mask only selects WHICH windows are kept (by their final bar), it never
    removes bars from inside a window.
    """
    empty = (
        np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
        np.empty((0,), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )
    if len(features) < seq_len:
        return empty

    end_indices = np.flatnonzero(valid_mask)
    end_indices = end_indices[end_indices >= seq_len - 1].astype(np.int64)
    if len(end_indices) == 0:
        return empty

    X_seq = np.stack([features[e - seq_len + 1:e + 1] for e in end_indices]).astype(np.float32)
    y_trade = tradeable[end_indices].astype(np.float32)
    y_dir = directions[end_indices].astype(np.int64)
    y_net_r = net_r[end_indices].astype(np.float32)

    return X_seq, y_trade, y_dir, y_net_r, end_indices


def _capped_pos_weight(positives: int, negatives: int, cap: float = 5.0) -> float:
    if positives <= 0 or negatives <= 0:
        return 1.0
    return float(min(np.sqrt(negatives / positives), cap))


def generate_dataset(data_dir, output_dir, smoke_test=False, seq_len=SEQUENCE_LENGTH):
    logger.info(f"Generating session-only dataset from {data_dir}...")
    loader = MyDataLoader(data_dir=data_dir)
    labeler = Labeler()
    engine = FeatureEngine()

    aligned_df, normalized_df = loader.get_features()

    all_sequences = []
    all_tradeable = []
    all_directions = []
    all_net_r = []
    all_timestamps = []
    all_asset_ids = []
    total_rows = 0
    asset_stats = {}

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
        asset_valid = filtered_labels_df['valid'].values.astype(bool)

        X_seq, y_trade, y_dir, y_net_r, end_indices = _build_sequences_for_asset(
            asset_X, asset_tradeable, asset_direction, asset_net_r, asset_valid, seq_len
        )
        if len(y_trade) == 0:
            continue

        all_sequences.append(X_seq)
        all_tradeable.append(y_trade)
        all_directions.append(y_dir)
        all_net_r.append(y_net_r)
        all_timestamps.append(common_indices.to_numpy(dtype="datetime64[ns]")[end_indices])
        all_asset_ids.append(np.full(len(y_trade), asset_id, dtype=np.int8))
        asset_total = len(y_trade)

        # Per-asset dataset diagnostics
        dir_cls, dir_cnt = np.unique(y_dir[y_trade > 0.5], return_counts=True)
        asset_stats[asset] = {
            "sequences": int(asset_total),
            "valid_ratio": round(float(filtered_labels_df['valid'].mean()), 4),
            "tradeable_count": int(y_trade.sum()),
            "tradeable_rate": round(float(y_trade.mean()), 4),
            "direction_counts_tradeable": {str(int(c)): int(n) for c, n in zip(dir_cls, dir_cnt)},
            "net_r_mean": round(float(y_net_r.mean()), 4),
            "net_r_p90": round(float(np.percentile(y_net_r, 90)), 4),
        }

        total_rows += asset_total
        logger.info(f"{asset}: generated {asset_total} sequences.")

    if not all_sequences:
        raise RuntimeError("No training sequences were generated. Check data quality and session filters.")

    X_np = np.concatenate(all_sequences, axis=0).astype(np.float32)
    y_trade_np = np.concatenate(all_tradeable, axis=0).astype(np.float32)
    y_dir_np = np.concatenate(all_directions, axis=0).astype(np.int64)
    y_net_r_np = np.concatenate(all_net_r, axis=0).astype(np.float32)
    timestamps_np = np.concatenate(all_timestamps, axis=0).astype("datetime64[ns]")
    asset_ids_np = np.concatenate(all_asset_ids, axis=0).astype(np.int8)

    order = np.argsort(timestamps_np, kind="mergesort")
    X_np = X_np[order]
    y_trade_np = y_trade_np[order]
    y_dir_np = y_dir_np[order]
    y_net_r_np = y_net_r_np[order]
    timestamps_np = timestamps_np[order]
    asset_ids_np = asset_ids_np[order]

    os.makedirs(output_dir, exist_ok=True)
    sequences_path = os.path.join(output_dir, "sequences.npy")
    labels_path = os.path.join(output_dir, "labels.npz")

    np.save(sequences_path, X_np)
    np.savez(
        labels_path,
        tradeable=y_trade_np,
        direction=y_dir_np,
        net_r=y_net_r_np,
        timestamp=timestamps_np,
        asset_id=asset_ids_np,
        asset_names=np.asarray(loader.assets),
        sequence_length=np.int32(seq_len),
    )

    # --- Dataset diagnostics: is this target learnable? ---
    logger.info("Computing dataset diagnostics...")
    trade_count = int(y_trade_np.sum())
    no_trade_count = int(len(y_trade_np) - trade_count)
    dir_cls, dir_cnt = np.unique(y_dir_np[y_trade_np > 0.5], return_counts=True)

    # Label distribution per month: a moving target here explains unstable training
    ts_index = pd.DatetimeIndex(timestamps_np)
    monthly = pd.crosstab(ts_index.to_period('M').astype(str), y_trade_np)
    monthly_dist = {
        str(month): {str(int(c)): int(v) for c, v in row.items()}
        for month, row in monthly.iterrows()
    }

    # Feature-label ANOVA F-scores on the LAST bar of each sequence.
    # If nothing scores above noise, the features (not the model) are the problem.
    f_scores = feature_label_scores(X_np[:, -1, :], y_trade_np, engine.feature_names)

    dataset_stats = {
        "generated_at": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "sequence_length": int(seq_len),
        "smoke_test": bool(smoke_test),
        "total_sequences": int(len(X_np)),
        "tradeable_counts_total": {"0": no_trade_count, "1": trade_count},
        "tradeable_rate": round(float(y_trade_np.mean()), 4),
        "direction_counts_tradeable": {str(int(c)): int(n) for c, n in zip(dir_cls, dir_cnt)},
        "net_r_mean": round(float(y_net_r_np.mean()), 4),
        "net_r_p90": round(float(np.percentile(y_net_r_np, 90)), 4),
        "assets": asset_stats,
        "monthly_class_distribution": monthly_dist,
        "feature_label_f_scores": f_scores,
    }
    stats_path = os.path.join(output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(dataset_stats, f, indent=2, cls=NumpyJSONEncoder)
    logger.info(f"Dataset diagnostics saved to {stats_path}")
    logger.info(f"Top-5 feature F-scores: {dict(list(f_scores.items())[:5])}")

    logger.info(f"Dataset generated. Total sequences: {len(X_np)}")
    return sequences_path, labels_path


def _binary_metrics(targets: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    targets = targets.astype(np.int64)
    tp = int(((preds == 1) & (targets == 1)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    total = max(1, len(targets))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    base_rate = float(targets.mean()) if len(targets) else 0.0
    return {
        "threshold": threshold,
        "accuracy": round((tp + tn) / total, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
        "base_rate": round(base_rate, 4),
        "edge": round(precision - base_rate, 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _evaluate_holdout(model, sequences, tradeable, directions, test_idx, batch_size=512) -> dict:
    """Evaluates the best two-head model on the untouched holdout set."""
    model.eval()
    trade_probs_list = []
    dir_probs_list = []
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            batch = np.asarray(sequences[test_idx[i:i + batch_size]], dtype=np.float32)
            outputs = model(torch.from_numpy(batch).to(DEVICE), return_dict=True)
            trade_probs_list.append(torch.sigmoid(outputs["trade_logit"].float()).cpu().numpy())
            dir_probs_list.append(torch.softmax(outputs["direction_logits"].float(), dim=1).cpu().numpy())

    trade_probs = np.concatenate(trade_probs_list)
    dir_probs = np.concatenate(dir_probs_list)
    trade_targets = tradeable[test_idx].astype(np.float32)
    dir_targets = directions[test_idx].astype(np.int64)

    trade_metrics = _binary_metrics(trade_targets, trade_probs)
    trade_preds = (trade_probs >= 0.5).astype(np.int64)
    dir_preds = np.argmax(dir_probs, axis=1)
    trade_mask = trade_targets > 0.5
    direction_accuracy = float((dir_preds[trade_mask] == dir_targets[trade_mask]).mean()) if trade_mask.any() else 0.0
    correct_trade = (trade_preds == trade_targets.astype(np.int64)).astype(np.float32)

    logger.info("Holdout tradeability confusion matrix (rows=true, cols=pred) [NoTrade, Trade]:\n%s",
                np.asarray(trade_metrics["confusion_matrix"]))
    logger.info(
        "Holdout tradeability: precision=%.3f recall=%.3f base_rate=%.3f edge=%+.3f",
        trade_metrics["precision"], trade_metrics["recall"], trade_metrics["base_rate"], trade_metrics["edge"]
    )
    logger.info("Holdout direction accuracy on tradeable samples: %.3f", direction_accuracy)

    return {
        "n_samples": int(len(test_idx)),
        "tradeability": trade_metrics,
        "direction_accuracy_tradeable": round(direction_accuracy, 4),
        "trade_confidence_histogram": confidence_histogram(trade_probs),
        "trade_confidence_calibration": confidence_bucket_table(trade_probs, correct_trade),
    }


def train_model(sequences_path, labels_path, model_save_path):
    logger.info("Starting optimized LSTM training...")
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
    if 'tradeable' not in labels_data or 'net_r' not in labels_data:
        raise RuntimeError("labels.npz is missing tradeability targets. Regenerate the dataset before training.")
    tradeable = labels_data['tradeable'].astype(np.float32)
    directions = labels_data['direction'].astype(np.int64)
    if 'timestamp' not in labels_data:
        raise RuntimeError("labels.npz is missing timestamp metadata. Regenerate the dataset before training.")
    timestamps = labels_data['timestamp']

    total_samples = len(sequences)
    if total_samples == 0:
        raise RuntimeError("No sequences found for training.")

    positives = int(tradeable.sum())
    negatives = int(total_samples - positives)
    trade_pos_weight = _capped_pos_weight(positives, negatives)
    trade_dirs = directions[tradeable > 0.5]
    direction_counts = {"sell": int((trade_dirs == 0).sum()), "buy": int((trade_dirs == 1).sum())}
    logger.info(f"Tradeability Distribution: NoTrade={negatives} | Trade={positives} | rate={positives / max(1, total_samples):.4f}")
    logger.info(f"Direction Distribution on tradeable samples: Sell={direction_counts['sell']} | Buy={direction_counts['buy']}")
    logger.info(f"Applied capped trade pos_weight={trade_pos_weight:.4f}")

    train_idx, val_idx, test_idx = _date_based_split_indices(timestamps)
    X_train, X_val = sequences[train_idx], sequences[val_idx]
    y_trade_train, y_trade_val = tradeable[train_idx], tradeable[val_idx]
    y_dir_train, y_dir_val = directions[train_idx], directions[val_idx]
    logger.info(
        "Date split: train=%d (%s to %s), val=%d (%s to %s), holdout_test=%d (%s to %s)",
        len(train_idx), np.asarray(timestamps[train_idx[0]]).astype(str), np.asarray(timestamps[train_idx[-1]]).astype(str),
        len(val_idx), np.asarray(timestamps[val_idx[0]]).astype(str), np.asarray(timestamps[val_idx[-1]]).astype(str),
        len(test_idx), np.asarray(timestamps[test_idx[0]]).astype(str), np.asarray(timestamps[test_idx[-1]]).astype(str),
    )

    train_dataset = AlphaSequenceDataset(X_train, y_trade_train, y_dir_train)
    val_dataset = AlphaSequenceDataset(X_val, y_trade_val, y_dir_val)

    num_workers = max(0, min(4, os.cpu_count() or 1))
    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[-1]
    model = AlphaSLModel(input_dim=input_dim, lstm_units=64, dense_units=32, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler(enabled=False)

    recorder.set_config(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT,
        input_dim=int(input_dim),
        lstm_units=64,
        dense_units=32,
        weight_decay=1e-4,
        device=str(DEVICE),
        trade_loss_weight=TRADE_LOSS_WEIGHT,
        direction_loss_weight=DIRECTION_LOSS_WEIGHT,
        trade_pos_weight=round(float(trade_pos_weight), 4),
        tradeable_counts={"0": negatives, "1": positives},
        direction_counts_tradeable=direction_counts,
        grad_clip_norm=GRAD_CLIP_NORM,
        amp_enabled=False,
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
    early_stop_patience = 10
    epochs_no_improve = 0

    for epoch in range(100):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        train_loss_sum = 0.0
        train_batches = 0
        grad_norm_sum = 0.0

        for b_X, b_trade, b_dir in pbar:
            b_X = b_X.to(DEVICE, non_blocking=True)
            b_trade = b_trade.to(DEVICE, non_blocking=True)
            b_dir = b_dir.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(b_X, return_dict=True)
            loss = trade_direction_loss(
                outputs,
                b_trade,
                b_dir,
                trade_weight=TRADE_LOSS_WEIGHT,
                direction_weight=DIRECTION_LOSS_WEIGHT,
                trade_pos_weight=trade_pos_weight,
            )

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}: {loss.item()}")

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Non-finite gradient norm at epoch {epoch + 1}: {grad_norm}")
            optimizer.step()

            train_loss_sum += loss.item()
            grad_norm_sum += float(grad_norm)
            train_batches += 1

        avg_train_loss = train_loss_sum / max(1, train_batches)
        avg_grad_norm = grad_norm_sum / max(1, train_batches)

        # Validation: loss + tradeability metrics + direction accuracy on tradeable samples
        model.eval()
        val_loss = 0.0
        val_trade_probs_list = []
        val_trade_targets_list = []
        val_dir_probs_list = []
        val_dir_targets_list = []
        with torch.no_grad():
            for b_X, b_trade, b_dir in val_loader:
                b_X = b_X.to(DEVICE, non_blocking=True)
                b_trade = b_trade.to(DEVICE, non_blocking=True)
                b_dir = b_dir.to(DEVICE, non_blocking=True)
                outputs = model(b_X, return_dict=True)
                loss = trade_direction_loss(
                    outputs,
                    b_trade,
                    b_dir,
                    trade_weight=TRADE_LOSS_WEIGHT,
                    direction_weight=DIRECTION_LOSS_WEIGHT,
                    trade_pos_weight=trade_pos_weight,
                )
                val_loss += loss.item()
                val_trade_probs_list.append(torch.sigmoid(outputs["trade_logit"].float()).cpu().numpy())
                val_trade_targets_list.append(b_trade.cpu().numpy())
                val_dir_probs_list.append(torch.softmax(outputs["direction_logits"].float(), dim=1).cpu().numpy())
                val_dir_targets_list.append(b_dir.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)

        val_trade_probs = np.concatenate(val_trade_probs_list)
        val_trade_targets = np.concatenate(val_trade_targets_list)
        val_dir_probs = np.concatenate(val_dir_probs_list)
        val_dir_targets = np.concatenate(val_dir_targets_list)
        val_trade_metrics = _binary_metrics(val_trade_targets, val_trade_probs)
        val_trade_mask = val_trade_targets > 0.5
        val_dir_preds = np.argmax(val_dir_probs, axis=1)
        val_dir_accuracy = float((val_dir_preds[val_trade_mask] == val_dir_targets[val_trade_mask]).mean()) if val_trade_mask.any() else 0.0

        recorder.log_epoch(
            epoch=epoch + 1,
            train_loss=round(avg_train_loss, 5),
            val_loss=round(avg_val_loss, 5),
            lr=optimizer.param_groups[0]['lr'],
            grad_norm=round(avg_grad_norm, 5),
            val_tradeability=val_trade_metrics,
            val_direction_accuracy_tradeable=round(val_dir_accuracy, 4),
            val_trade_confidence=confidence_histogram(val_trade_probs),
        )
        logger.info(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | "
            f"Trade Prec = {val_trade_metrics['precision']:.3f} | Trade Rec = {val_trade_metrics['recall']:.3f} | "
            f"Dir Acc = {val_dir_accuracy:.3f} | Grad Norm = {avg_grad_norm:.3f}"
        )

        if avg_val_loss < best_val_loss:
            logger.info(f"New best Val Loss ({avg_val_loss:.4f})! Saving model to {model_save_path}")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info("Early stopping triggered.")
                break

    # Final report card: per-class precision/recall on the untouched holdout period
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    logger.info("Evaluating best model on holdout test period...")
    holdout_report = _evaluate_holdout(model, sequences, tradeable, directions, test_idx)
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
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, args.data_dir))
    dataset_dir = os.path.join(base_dir, "data", "training_set")
    model_path = os.path.join(base_dir, "models", "alpha_model.pth")

    if not args.skip_gen:
        generate_dataset(data_dir, dataset_dir, smoke_test=args.smoke_test)

    train_model(
        os.path.join(dataset_dir, "sequences.npy"),
        os.path.join(dataset_dir, "labels.npz"),
        model_path,
    )


if __name__ == "__main__":
    main()

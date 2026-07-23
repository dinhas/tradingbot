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
from Alpha.src.model import AlphaSLModel, direction_loss
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.diagnostics import (
    DiagnosticsRecorder, NumpyJSONEncoder, compute_class_metrics,
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
LEARNING_RATE = 1e-3
DROPOUT = 0.3
SESSION_COL = "is_late_session"

# Label / split hygiene
LABEL_MAX_BARS = 6      # must match Labeler.max_bars (vertical barrier)
BAR_MINUTES = 5
# Purge: drop train/val samples whose label horizon crosses the split boundary.
PURGE_TD = np.timedelta64(LABEL_MAX_BARS * BAR_MINUTES, 'm')
# Embargo: gap after each split cut so no input window or label horizon straddles it.
EMBARGO_TD = np.timedelta64((SEQUENCE_LENGTH + LABEL_MAX_BARS) * BAR_MINUTES, 'm')


class AlphaSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_seq = self.sequences[idx].copy()
        y_dir = self.labels[idx]
        return torch.from_numpy(x_seq), torch.tensor(y_dir, dtype=torch.float32)


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


def _build_sequences_for_asset(features: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds CONTIGUOUS rolling windows for one asset.

    Windows are always built over adjacent bars (matching live/backtest inference).
    The valid_mask only selects WHICH windows are kept (by their final bar), it never
    removes bars from inside a window.
    """
    empty = (
        np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
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
    y_seq = labels[end_indices].astype(np.float32)

    return X_seq, y_seq, end_indices


def generate_dataset(data_dir, output_dir, smoke_test=False, seq_len=SEQUENCE_LENGTH):
    logger.info(f"Generating session-only dataset from {data_dir}...")
    loader = MyDataLoader(data_dir=data_dir)
    labeler = Labeler()
    engine = FeatureEngine()

    aligned_df, normalized_df = loader.get_features()

    all_sequences = []
    all_labels = []
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
        asset_y = filtered_labels_df['direction'].values.astype(np.float32)
        asset_valid = filtered_labels_df['valid'].values.astype(bool)

        X_seq, y_seq, end_indices = _build_sequences_for_asset(asset_X, asset_y, asset_valid, seq_len)
        if len(y_seq) == 0:
            continue

        all_sequences.append(X_seq)
        all_labels.append(y_seq)
        all_timestamps.append(common_indices.to_numpy(dtype="datetime64[ns]")[end_indices])
        all_asset_ids.append(np.full(len(y_seq), asset_id, dtype=np.int8))
        asset_total = len(y_seq)

        # Per-asset dataset diagnostics
        cls, cnt = np.unique(y_seq, return_counts=True)
        asset_stats[asset] = {
            "sequences": int(asset_total),
            "valid_ratio": round(float(filtered_labels_df['valid'].mean()), 4),
            "class_counts": {str(int(c)): int(n) for c, n in zip(cls, cnt)},
        }

        total_rows += asset_total
        logger.info(f"{asset}: generated {asset_total} sequences.")

    if not all_sequences:
        raise RuntimeError("No training sequences were generated. Check data quality and session filters.")

    X_np = np.concatenate(all_sequences, axis=0).astype(np.float32)
    y_dir_np = np.concatenate(all_labels, axis=0).astype(np.float32)
    timestamps_np = np.concatenate(all_timestamps, axis=0).astype("datetime64[ns]")
    asset_ids_np = np.concatenate(all_asset_ids, axis=0).astype(np.int8)

    order = np.argsort(timestamps_np, kind="mergesort")
    X_np = X_np[order]
    y_dir_np = y_dir_np[order]
    timestamps_np = timestamps_np[order]
    asset_ids_np = asset_ids_np[order]

    os.makedirs(output_dir, exist_ok=True)
    sequences_path = os.path.join(output_dir, "sequences.npy")
    labels_path = os.path.join(output_dir, "labels.npz")

    np.save(sequences_path, X_np)
    np.savez(
        labels_path,
        direction=y_dir_np,
        timestamp=timestamps_np,
        asset_id=asset_ids_np,
        asset_names=np.asarray(loader.assets),
        sequence_length=np.int32(seq_len),
    )

    # --- Dataset diagnostics: is this target learnable? ---
    logger.info("Computing dataset diagnostics...")
    cls, cnt = np.unique(y_dir_np, return_counts=True)

    # Label distribution per month: a moving target here explains unstable training
    ts_index = pd.DatetimeIndex(timestamps_np)
    monthly = pd.crosstab(ts_index.to_period('M').astype(str), y_dir_np)
    monthly_dist = {
        str(month): {str(int(c)): int(v) for c, v in row.items()}
        for month, row in monthly.iterrows()
    }

    # Feature-label ANOVA F-scores on the LAST bar of each sequence.
    # If nothing scores above noise, the features (not the model) are the problem.
    f_scores = feature_label_scores(X_np[:, -1, :], y_dir_np, engine.feature_names)

    dataset_stats = {
        "generated_at": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "sequence_length": int(seq_len),
        "smoke_test": bool(smoke_test),
        "total_sequences": int(len(X_np)),
        "class_counts_total": {str(int(c)): int(n) for c, n in zip(cls, cnt)},
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


def _evaluate_holdout(model, sequences, directions, test_idx, batch_size=512) -> dict:
    """Evaluates the best model on the untouched holdout set.

    Logs a confusion matrix + per-class precision/recall and returns a full
    diagnostics dict (metrics, confidence histogram, calibration buckets).
    """
    model.eval()
    preds = []
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            batch = np.asarray(sequences[test_idx[i:i + batch_size]], dtype=np.float32)
            logits = model(torch.from_numpy(batch).to(DEVICE))
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            probs_list.append(probs)
            preds.append(np.argmax(probs, axis=1))
    preds = np.concatenate(preds)
    probs = np.concatenate(probs_list)
    max_probs = probs.max(axis=1)
    targets = (directions[test_idx] + 1).astype(np.int64)  # [-1,0,1] -> [0,1,2]

    metrics = compute_class_metrics(targets, preds)
    correct = (preds == targets).astype(np.float32)

    class_names = ["Sell(-1)", "Neutral(0)", "Buy(+1)"]
    logger.info("Holdout confusion matrix (rows=true, cols=pred) [Sell, Neutral, Buy]:\n%s",
                np.asarray(metrics["confusion_matrix"]))
    for c, key in enumerate(["sell", "neutral", "buy"]):
        m = metrics["per_class"][key]
        logger.info(
            f"{class_names[c]:<11}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
            f"base_rate={m['base_rate']:.3f} (edge={m['edge']:+.3f})"
        )
    logger.info(f"Holdout accuracy: {metrics['accuracy']:.3f}")

    return {
        "n_samples": int(len(test_idx)),
        **metrics,
        "confidence_histogram": confidence_histogram(max_probs),
        "confidence_calibration": confidence_bucket_table(max_probs, correct),
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
    directions = labels_data['direction']
    if 'timestamp' not in labels_data:
        raise RuntimeError("labels.npz is missing timestamp metadata. Regenerate the dataset before training.")
    timestamps = labels_data['timestamp']

    total_samples = len(sequences)
    if total_samples == 0:
        raise RuntimeError("No sequences found for training.")

    from collections import Counter
    counts = Counter(directions.tolist())
    
    # Calculate Inverse Frequency Weights 
    # Formula: Total_Samples / (Num_Classes * Class_Count)
    # The targets are mapped to indices [0, 1, 2] corresponding to [-1, 0, 1]
    weight_minus_1 = total_samples / (3 * max(1, counts.get(-1.0, 1)))
    weight_0       = total_samples / (3 * max(1, counts.get(0.0, 1)))
    weight_plus_1  = total_samples / (3 * max(1, counts.get(1.0, 1)))
    
    class_weights = torch.tensor([weight_minus_1, weight_0, weight_plus_1], dtype=torch.float32).to(DEVICE)
    logger.info(f"Class Distribution [-1, 0, 1]: Sell={counts.get(-1.0,0)} | Neutral={counts.get(0.0,0)} | Buy={counts.get(1.0,0)}")
    logger.info(f"Applied Focal Loss Weights: {class_weights.cpu().numpy()}")

    train_idx, val_idx, test_idx = _date_based_split_indices(timestamps)
    X_train, X_val = sequences[train_idx], sequences[val_idx]
    y_train, y_val = directions[train_idx], directions[val_idx]
    logger.info(
        "Date split: train=%d (%s to %s), val=%d (%s to %s), holdout_test=%d (%s to %s)",
        len(train_idx), np.asarray(timestamps[train_idx[0]]).astype(str), np.asarray(timestamps[train_idx[-1]]).astype(str),
        len(val_idx), np.asarray(timestamps[val_idx[0]]).astype(str), np.asarray(timestamps[val_idx[-1]]).astype(str),
        len(test_idx), np.asarray(timestamps[test_idx[0]]).astype(str), np.asarray(timestamps[test_idx[-1]]).astype(str),
    )

    train_dataset = AlphaSequenceDataset(X_train, y_train)
    val_dataset = AlphaSequenceDataset(X_val, y_val)

    num_workers = max(0, min(4, os.cpu_count() or 1))
    pin_memory = DEVICE.type == "cuda"

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[-1]
    model = AlphaSLModel(input_dim=input_dim, lstm_units=64, dense_units=32, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

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
        class_weights=class_weights.cpu().numpy().round(4).tolist(),
        class_counts={"-1": counts.get(-1.0, 0), "0": counts.get(0.0, 0), "1": counts.get(1.0, 0)},
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

        for b_X, b_dir in pbar:
            b_X = b_X.to(DEVICE, non_blocking=True)
            b_dir = b_dir.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                logits = model(b_X)
                loss = direction_loss(logits, b_dir, alpha_dir=class_weights)

            scaler.scale(loss).backward()
            # Unscale so the recorded gradient norm is the true norm (also detects vanishing grads)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += loss.item()
            grad_norm_sum += float(grad_norm)
            train_batches += 1

        avg_train_loss = train_loss_sum / max(1, train_batches)
        avg_grad_norm = grad_norm_sum / max(1, train_batches)

        # Validation: loss + per-class metrics + confidence distribution
        model.eval()
        val_loss = 0.0
        val_probs_list = []
        val_targets_list = []
        with torch.no_grad():
            for b_X, b_dir in val_loader:
                b_X = b_X.to(DEVICE, non_blocking=True)
                b_dir = b_dir.to(DEVICE, non_blocking=True)
                logits = model(b_X)
                loss = direction_loss(logits, b_dir, alpha_dir=class_weights)
                val_loss += loss.item()
                val_probs_list.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
                val_targets_list.append((b_dir + 1).long().cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)

        val_probs = np.concatenate(val_probs_list)
        val_targets = np.concatenate(val_targets_list)
        val_preds = np.argmax(val_probs, axis=1)
        val_metrics = compute_class_metrics(val_targets, val_preds)

        recorder.log_epoch(
            epoch=epoch + 1,
            train_loss=round(avg_train_loss, 5),
            val_loss=round(avg_val_loss, 5),
            lr=optimizer.param_groups[0]['lr'],
            grad_norm=round(avg_grad_norm, 5),
            val_accuracy=val_metrics["accuracy"],
            val_per_class=val_metrics["per_class"],
            val_confidence=confidence_histogram(val_probs.max(axis=1)),
        )
        logger.info(
            f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f} | "
            f"Val Acc = {val_metrics['accuracy']:.3f} | Grad Norm = {avg_grad_norm:.3f} | "
            f"Buy edge = {val_metrics['per_class']['buy']['edge']:+.3f} | "
            f"Sell edge = {val_metrics['per_class']['sell']['edge']:+.3f}"
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
    holdout_report = _evaluate_holdout(model, sequences, directions, test_idx)
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

import os
import sys
import argparse
import numpy as np
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"alpha_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
    """Splits samples by signal timestamp so future periods never leak into training."""
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

    train_idx = np.flatnonzero(ts <= train_cut)
    val_idx = np.flatnonzero((ts > train_cut) & (ts <= val_cut))
    test_idx = np.flatnonzero(ts > val_cut)

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Date split produced an empty train, validation, or test partition.")

    return train_idx, val_idx, test_idx


def _build_sequences_for_asset(features: np.ndarray, labels: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Builds in-session rolling windows for one asset."""
    if len(features) < seq_len:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_seq = []
    y_seq = []

    for end_idx in range(seq_len - 1, len(features)):
        start_idx = end_idx - seq_len + 1
        X_seq.append(features[start_idx:end_idx + 1])
        y_seq.append(labels[end_idx])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


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

        X_seq, y_seq = _build_sequences_for_asset(asset_X, asset_y, seq_len)
        if len(y_seq) == 0:
            continue

        all_sequences.append(X_seq)
        all_labels.append(y_seq)
        all_timestamps.append(common_indices[seq_len - 1:].to_numpy(dtype="datetime64[ns]"))
        all_asset_ids.append(np.full(len(y_seq), asset_id, dtype=np.int8))
        asset_total = len(y_seq)

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

    logger.info(f"Dataset generated. Total sequences: {len(X_np)}")
    return sequences_path, labels_path


def train_model(sequences_path, labels_path, model_save_path):
    logger.info("Starting optimized LSTM training...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

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

    best_val_loss = float('inf')
    early_stop_patience = 10
    epochs_no_improve = 0

    for epoch in range(100):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for b_X, b_dir in pbar:
            b_X = b_X.to(DEVICE, non_blocking=True)
            b_dir = b_dir.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                logits = model(b_X)
                loss = direction_loss(logits, b_dir, alpha_dir=class_weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_X, b_dir in val_loader:
                b_X = b_X.to(DEVICE, non_blocking=True)
                b_dir = b_dir.to(DEVICE, non_blocking=True)
                logits = model(b_X)
                loss = direction_loss(logits, b_dir, alpha_dir=class_weights)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)
        logger.info(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            logger.info(f"✨ New best Val Loss ({avg_val_loss:.4f})! Saving model to {model_save_path}")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info("Early stopping triggered.")
                break

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

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
    total_rows = 0

    for asset in loader.assets:
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

        # Session-only training: keep only the last candles of the session.
        if SESSION_COL not in filtered_norm_df.columns:
            raise ValueError(f"Missing required session filter column: {SESSION_COL}")

        session_mask = filtered_norm_df[SESSION_COL] == 1
        filtered_norm_df = filtered_norm_df.loc[session_mask]
        filtered_labels_df = filtered_labels_df.loc[session_mask]

        if filtered_norm_df.empty:
            logger.warning(f"No late-session rows after filtering for {asset}; skipping.")
            continue

        # Keep sequences within each trading day/session to prevent cross-session leakage.
        session_groups = filtered_norm_df.groupby(filtered_norm_df.index.date)

        asset_total = 0
        for session_date, session_features_df in session_groups:
            session_labels_df = filtered_labels_df.loc[session_features_df.index]
            if len(session_features_df) < seq_len:
                continue

            session_X = engine.get_observation_vectorized(session_features_df, asset)
            session_y = session_labels_df['direction'].values.astype(np.float32)

            X_seq, y_seq = _build_sequences_for_asset(session_X, session_y, seq_len)
            if len(y_seq) == 0:
                continue

            all_sequences.append(X_seq)
            all_labels.append(y_seq)
            asset_total += len(y_seq)

        total_rows += asset_total
        logger.info(f"{asset}: generated {asset_total} sequences.")

    if not all_sequences:
        raise RuntimeError("No training sequences were generated. Check data quality and session filters.")

    X_np = np.concatenate(all_sequences, axis=0).astype(np.float32)
    y_dir_np = np.concatenate(all_labels, axis=0).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    sequences_path = os.path.join(output_dir, "sequences.npy")
    labels_path = os.path.join(output_dir, "labels.npz")

    np.save(sequences_path, X_np)
    np.savez(labels_path, direction=y_dir_np)

    logger.info(f"Dataset generated. Total sequences: {len(X_np)}")
    return sequences_path, labels_path


def train_model(sequences_path, labels_path, model_save_path):
    logger.info("Starting optimized LSTM training...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    sequences = np.load(sequences_path, mmap_mode='r')
    labels_data = np.load(labels_path)
    directions = labels_data['direction']

    total_samples = len(sequences)
    if total_samples == 0:
        raise RuntimeError("No sequences found for training.")

    # Time-series-aware split.
    split = int(0.90 * total_samples)
    X_train, X_val = sequences[:split], sequences[split:]
    y_train, y_val = directions[:split], directions[split:]

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
                loss = direction_loss(logits, b_dir)

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
                loss = direction_loss(logits, b_dir)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        scheduler.step(avg_val_loss)
        logger.info(f"Epoch {epoch + 1}: Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
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

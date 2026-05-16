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

from Alpha.src.model import AlphaSLModel, direction_loss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BEST-PRACTICE SETTINGS
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
DROPOUT = 0.3

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


def train_model(sequences_path, labels_path, model_save_path):
    logger.info("Starting optimized LSTM training...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    sequences = np.load(sequences_path, mmap_mode='r')
    labels_data = np.load(labels_path)
    directions = labels_data['direction']

    total_samples = len(sequences)
    if total_samples == 0:
        raise RuntimeError("No sequences found for training.")

    from collections import Counter
    counts = Counter(directions.tolist())

    # Calculate Inverse Frequency Weights
    weight_minus_1 = total_samples / (3 * max(1, counts.get(-1.0, 1)))
    weight_0       = total_samples / (3 * max(1, counts.get(0.0, 1)))
    weight_plus_1  = total_samples / (3 * max(1, counts.get(1.0, 1)))

    class_weights = torch.tensor([weight_minus_1, weight_0, weight_plus_1], dtype=torch.float32).to(DEVICE)
    logger.info(f"Class Distribution [-1, 0, 1]: Sell={counts.get(-1.0,0)} | Neutral={counts.get(0.0,0)} | Buy={counts.get(1.0,0)}")
    logger.info(f"Applied Focal Loss Weights: {class_weights.cpu().numpy()}")

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
    parser.add_argument("--sequences-path", type=str, default="Alpha/data/training_set/sequences.npy")
    parser.add_argument("--labels-path", type=str, default="Alpha/data/training_set/labels.npz")
    parser.add_argument("--model-save-path", type=str, default="Alpha/models/alpha_model.pth")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sequences_path = os.path.abspath(os.path.join(base_dir, args.sequences_path))
    labels_path = os.path.abspath(os.path.join(base_dir, args.labels_path))
    model_save_path = os.path.abspath(os.path.join(base_dir, args.model_save_path))

    train_model(sequences_path, labels_path, model_save_path)


if __name__ == "__main__":
    main()

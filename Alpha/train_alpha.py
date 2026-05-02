import os
import sys
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
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DROPOUT = 0.3
EPOCHS = 5 # Further reduced for speed on large dataset

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

def train_alpha():
    dataset_dir = os.path.join(PROJECT_ROOT, "Alpha", "data", "training_set")
    model_save_path = os.path.join(PROJECT_ROOT, "Alpha", "models", "alpha_model.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    sequences_path = os.path.join(dataset_dir, "sequences.npy")
    labels_path = os.path.join(dataset_dir, "labels.npz")

    logger.info("Starting optimized LSTM training on CPU...")

    sequences = np.load(sequences_path, mmap_mode='r')
    labels_data = np.load(labels_path)
    directions = labels_data['direction']

    total_samples = len(sequences)
    if total_samples == 0:
        raise RuntimeError("No sequences found for training.")

    from collections import Counter
    counts = Counter(directions.tolist())
    logger.info(f"Class Distribution: {counts}")

    weight_minus_1 = total_samples / (3 * max(1, counts.get(-1.0, 1)))
    weight_0       = total_samples / (3 * max(1, counts.get(0.0, 1)))
    weight_plus_1  = total_samples / (3 * max(1, counts.get(1.0, 1)))

    class_weights = torch.tensor([weight_minus_1, weight_0, weight_plus_1], dtype=torch.float32).to(DEVICE)

    split = int(0.90 * total_samples)
    X_train, X_val = sequences[:split], sequences[split:]
    y_train, y_val = directions[:split], directions[split:]

    train_dataset = AlphaSequenceDataset(X_train, y_train)
    val_dataset = AlphaSequenceDataset(X_val, y_val)

    num_workers = 4 # Optimized for 4-core CPU
    pin_memory = False # CPU only

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    input_dim = X_train.shape[-1]
    model = AlphaSLModel(input_dim=input_dim, lstm_units=64, dense_units=32, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for b_X, b_dir in pbar:
            b_X = b_X.to(DEVICE, non_blocking=True); b_dir = b_dir.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                logits = model(b_X)
                loss = direction_loss(logits, b_dir, alpha_dir=class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_X, b_dir in val_loader:
                b_X = b_X.to(DEVICE, non_blocking=True); b_dir = b_dir.to(DEVICE, non_blocking=True)
                logits = model(b_X)
                loss = direction_loss(logits, b_dir, alpha_dir=class_weights)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / max(1, len(val_loader))
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model saved to {model_save_path}")

    logger.info("Training complete.")

if __name__ == "__main__":
    train_alpha()

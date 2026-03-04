import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import logging
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from risk_model_rl import RiskModelRL, NUM_SL_CHOICES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATASET_PATH = os.environ.get(
    "RL_DATASET_PATH",
    os.path.join(os.path.dirname(__file__), "data", "rl_risk_dataset.parquet"),
)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

BATCH_SIZE = 2048
LEARNING_RATE = 3e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1

W_SL = 1.2
W_TP = 0.8
W_SIZE = 1.0

USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.05
USE_HUBER_LOSS = True
HUBER_DELTA = 1.0

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class RLRiskDataset(Dataset):
    def __init__(self, features, sl_targets, tp_targets, size_targets):
        self.features = features
        self.sl_targets = sl_targets
        self.tp_targets = tp_targets
        self.size_targets = size_targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.sl_targets[idx], dtype=torch.long),
            torch.tensor(self.tp_targets[idx], dtype=torch.float32),
            torch.tensor(self.size_targets[idx], dtype=torch.float32),
        )


def train():
    logger.info(f"Starting RL Risk Model Training on {DEVICE} with {NUM_GPUS} GPUs")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    logger.info(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)

    logger.info(f"Dataset loaded: {len(df)} samples")

    X = np.stack(df["features"].values).astype(np.float32)
    y_sl = df["target_sl_idx"].values.astype(np.int64)
    y_tp = df["target_tp_mult"].values.astype(np.float32)
    y_size = df["target_size"].values.astype(np.float32)

    (
        X_train,
        X_val,
        y_sl_train,
        y_sl_val,
        y_tp_train,
        y_tp_val,
        y_size_train,
        y_size_val,
    ) = train_test_split(X, y_sl, y_tp, y_size, test_size=0.10, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    scaler_path = os.path.join(MODELS_DIR, "rl_risk_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    train_dataset = RLRiskDataset(X_train_scaled, y_sl_train, y_tp_train, y_size_train)
    val_dataset = RLRiskDataset(X_val_scaled, y_sl_val, y_tp_val, y_size_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
    )

    model = RiskModelRL(input_dim=40).to(DEVICE)
    if NUM_GPUS > 1:
        logger.info(f"Using {NUM_GPUS} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    scaler_amp = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    ce_loss = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    if USE_FOCAL_LOSS:
        from torch.nn.modules.loss import CrossEntropyLoss

        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.0):
                super().__init__()
                self.gamma = gamma
                self.alpha = alpha
                self.label_smoothing = label_smoothing

            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(
                    inputs,
                    targets,
                    label_smoothing=self.label_smoothing,
                    reduction="none",
                )
                pt = torch.exp(-ce_loss)
                focal_loss = ((1 - pt) ** self.gamma) * ce_loss
                return focal_loss.mean()

        sl_loss_fn = FocalLoss(gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    else:
        sl_loss_fn = ce_loss

    if USE_HUBER_LOSS:
        tp_loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
        size_loss_fn = nn.HuberLoss(delta=HUBER_DELTA)
    else:
        tp_loss_fn = nn.MSELoss()
        size_loss_fn = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_sl_loss = 0
        train_tp_loss = 0
        train_size_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            features, sl_target, tp_target, size_target = [
                b.to(DEVICE, non_blocking=True) for b in batch
            ]

            optimizer.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    preds = model(features)
                    loss_sl = sl_loss_fn(preds["sl_logits"], sl_target)
                    loss_tp = tp_loss_fn(preds["tp"], tp_target)
                    loss_size = size_loss_fn(preds["size"], size_target)
                    total_loss = (
                        (W_SL * loss_sl) + (W_TP * loss_tp) + (W_SIZE * loss_size)
                    )

                scaler_amp.scale(total_loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                preds = model(features)
                loss_sl = sl_loss_fn(preds["sl_logits"], sl_target)
                loss_tp = tp_loss_fn(preds["tp"], tp_target)
                loss_size = size_loss_fn(preds["size"], size_target)
                total_loss = (W_SL * loss_sl) + (W_TP * loss_tp) + (W_SIZE * loss_size)
                total_loss.backward()
                optimizer.step()

            train_loss += total_loss.item()
            train_sl_loss += loss_sl.item()
            train_tp_loss += loss_tp.item()
            train_size_loss += loss_size.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features, sl_target, tp_target, size_target = [
                    b.to(DEVICE, non_blocking=True) for b in batch
                ]

                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        preds = model(features)
                        v_sl = sl_loss_fn(preds["sl_logits"], sl_target)
                        v_tp = tp_loss_fn(preds["tp"], tp_target)
                        v_size = size_loss_fn(preds["size"], size_target)
                        v_total = (W_SL * v_sl) + (W_TP * v_tp) + (W_SIZE * v_size)
                else:
                    preds = model(features)
                    v_sl = sl_loss_fn(preds["sl_logits"], sl_target)
                    v_tp = tp_loss_fn(preds["tp"], tp_target)
                    v_size = size_loss_fn(preds["size"], size_target)
                    v_total = (W_SL * v_sl) + (W_TP * v_tp) + (W_SIZE * v_size)

                val_loss += v_total.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model = model.module if NUM_GPUS > 1 else model
            torch.save(
                {"model_state_dict": save_model.state_dict(), "scaler": scaler},
                os.path.join(MODELS_DIR, "risk_model_rl_best.pth"),
            )
            logger.info(f"New best model saved with val_loss: {avg_val_loss:.6f}")

    logger.info("Training Complete.")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    train()

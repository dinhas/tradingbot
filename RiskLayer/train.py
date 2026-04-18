"""Training loop for the multi-head LSTM RiskLayer model."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RiskLayer.model import MultiHeadRiskLSTM

LOGGER = logging.getLogger(__name__)


class RiskDataset(Dataset):
    def __init__(self, sequences: np.ndarray, sl: np.ndarray, tp: np.ndarray, quality: np.ndarray) -> None:
        self.x = torch.from_numpy(sequences).float()
        self.sl = torch.from_numpy(sl).float()
        self.tp = torch.from_numpy(tp).float()
        self.q = torch.from_numpy(quality).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.sl[idx], self.tp[idx], self.q[idx]


@dataclass
class TrainConfig:
    seq_path: str
    targets_path: str
    save_path: str
    batch_size: int = 256
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    val_split: float = 0.2
    seed: int = 42


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_loaders(config: TrainConfig) -> tuple[DataLoader, DataLoader, int]:
    X = np.load(config.seq_path).astype(np.float32)
    y = np.load(config.targets_path)

    sl = y["sl_atr"].astype(np.float32)
    tp = y["tp_atr"].astype(np.float32)
    quality = y["quality"].astype(np.float32)

    if len(X) != len(sl):
        raise ValueError("Sequence and target lengths do not match")

    n_total = len(X)
    n_val = max(1, int(n_total * config.val_split))
    n_train = n_total - n_val

    # Chronological split to avoid leakage.
    train_ds = RiskDataset(X[:n_train], sl[:n_train], tp[:n_train], quality[:n_train])
    val_ds = RiskDataset(X[n_train:], sl[n_train:], tp[n_train:], quality[n_train:])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False)

    input_size = X.shape[-1]
    return train_loader, val_loader, input_size


def _run_epoch(
    model: MultiHeadRiskLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_sl = total_tp = total_q = 0.0
    n = 0

    for x, y_sl, y_tp, y_q in loader:
        x = x.to(device)
        y_sl = y_sl.to(device)
        y_tp = y_tp.to(device)
        y_q = y_q.to(device)

        with torch.set_grad_enabled(is_train):
            p_sl, p_tp, p_q = model(x)
            l_sl = criterion(p_sl, y_sl)
            l_tp = criterion(p_tp, y_tp)
            l_q = criterion(p_q, y_q)
            loss = l_sl + l_tp + l_q

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = x.size(0)
        n += batch_size
        total_loss += float(loss.item()) * batch_size
        total_sl += float(l_sl.item()) * batch_size
        total_tp += float(l_tp.item()) * batch_size
        total_q += float(l_q.item()) * batch_size

    return total_loss / n, total_sl / n, total_tp / n, total_q / n


def train(config: TrainConfig) -> str:
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, input_size = _build_loaders(config)

    model = MultiHeadRiskLSTM(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    os.makedirs(os.path.dirname(config.save_path) or ".", exist_ok=True)

    best_val = float("inf")
    patience = 12
    epochs_no_improve = 0

    for epoch in range(1, config.epochs + 1):
        tr = _run_epoch(model, train_loader, optimizer, criterion, device)
        va = _run_epoch(model, val_loader, None, criterion, device)

        LOGGER.info(
            "Epoch %d/%d | train loss=%.5f (sl=%.5f tp=%.5f q=%.5f) | val loss=%.5f (sl=%.5f tp=%.5f q=%.5f)",
            epoch,
            config.epochs,
            tr[0],
            tr[1],
            tr[2],
            tr[3],
            va[0],
            va[1],
            va[2],
            va[3],
        )

        if va[0] < best_val:
            best_val = va[0]
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "config": vars(config),
                    "best_val_loss": best_val,
                },
                config.save_path,
            )
            LOGGER.info("Saved best checkpoint to %s (Val Loss improved)", config.save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                LOGGER.info("Early stopping triggered after %d epochs without improvement.", patience)
                break

    return config.save_path


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(description="Train multi-head LSTM risk model")
    parser.add_argument("--sequences", default="RiskLayer/data/training_set/sequences.npy")
    parser.add_argument("--targets", default="RiskLayer/data/training_set/risk_targets.npz")
    parser.add_argument("--save-path", default="RiskLayer/models/risk_lstm_multitask.pth")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainConfig(
        seq_path=args.sequences,
        targets_path=args.targets,
        save_path=args.save_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        val_split=args.val_split,
        seed=args.seed,
    )
    train(config)


if __name__ == "__main__":
    main()

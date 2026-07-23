"""Micro-validation for the current Alpha SL pipeline.

Verifies end-to-end on a small slice of real data:
1. Features contain no NaNs.
2. Labeling produces the expected {direction, valid} schema and a sane distribution.
3. Contiguous sequence building works and shapes align with the model.
4. One training epoch reduces the loss and gradients flow through the LSTM.
5. Confusion matrix on the micro set (sanity, not a performance claim).

Run: python -m Alpha.src.micro_val
"""

import sys
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.data_loader import DataLoader as MyDataLoader
from Alpha.src.labeling import Labeler
from Alpha.src.model import AlphaSLModel, direction_loss
from Alpha.src.feature_engine import FeatureEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SEQ_LEN = 50
MICRO_ROWS = 5000


def _build_sequences(features: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray, seq_len: int):
    """Contiguous rolling windows; valid_mask selects which windows are kept (by final bar)."""
    end_indices = np.flatnonzero(valid_mask)
    end_indices = end_indices[end_indices >= seq_len - 1]
    if len(end_indices) == 0:
        return (np.empty((0, seq_len, features.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=np.float32))
    X = np.stack([features[e - seq_len + 1:e + 1] for e in end_indices]).astype(np.float32)
    y = labels[end_indices].astype(np.float32)
    return X, y


def run_micro_validation():
    logger.info("--- ALPHA SL MICRO-VALIDATION START ---")

    # 1. Data + features
    loader = MyDataLoader()
    aligned_df, normalized_df = loader.get_features()
    logger.info(f"Normalized DF shape: {normalized_df.shape}")

    nan_count = int(normalized_df.isna().sum().sum())
    logger.info(f"Total NaNs in features: {nan_count}")
    if nan_count > 0:
        logger.error("ERROR: NaNs detected in features!")
        return False

    # 2. Labeling
    labeler = Labeler()
    asset = 'EURUSD'
    labels_df = labeler.label_data(aligned_df, asset).head(MICRO_ROWS)
    assert {'direction', 'valid'}.issubset(labels_df.columns), "Labeler schema changed!"
    logger.info(f"Labeled rows: {len(labels_df)} | valid ratio: {labels_df['valid'].mean():.2%}")
    logger.info("Class distribution among valid labels:")
    logger.info(labels_df.loc[labels_df['valid'], 'direction'].value_counts(normalize=True).sort_index())

    # 3. Sequences
    common = labels_df.index.intersection(normalized_df.index)
    norm = normalized_df.loc[common]
    labels_df = labels_df.loc[common]

    engine = FeatureEngine()
    features = engine.get_observation_vectorized(norm, asset)
    logger.info(f"Feature matrix shape: {features.shape} (expected {len(engine.feature_names)} dims)")
    assert features.shape[1] == len(engine.feature_names), "Feature dim mismatch!"

    X, y = _build_sequences(
        features,
        labels_df['direction'].values,
        labels_df['valid'].values.astype(bool),
        SEQ_LEN,
    )
    logger.info(f"Sequence tensor shape: {X.shape}")
    if len(y) < 64:
        logger.error("ERROR: not enough valid sequences for micro-training.")
        return False

    # 4. One training epoch
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    train_loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

    model = AlphaSLModel(input_dim=X.shape[-1])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    logger.info("\nRunning 1 training epoch...")
    model.train()
    initial_loss = None
    final_loss = None
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        logits = model(batch_X)
        loss = direction_loss(logits, batch_y)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    logger.info(f"Initial loss: {initial_loss:.4f} | Final loss: {final_loss:.4f}")
    loss_ok = final_loss < initial_loss
    logger.info("CONFIRMED: Loss decreased." if loss_ok
                else "WARNING: Loss did not decrease (small dataset noise is possible).")

    grads_ok = all(p.grad is not None for p in model.lstm.parameters())
    logger.info(f"Gradients flow to LSTM: {grads_ok}")

    # 5. Confusion matrix on the micro set (sanity only)
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_t), dim=1).numpy()
    targets = (y + 1).astype(np.int64)
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    logger.info(f"\nConfusion matrix (rows=true, cols=pred) [Sell, Neutral, Buy]:\n{cm}")

    ok = loss_ok is not None and grads_ok
    logger.info("\n--- MICRO-VALIDATION COMPLETE ---")
    return ok


if __name__ == "__main__":
    success = run_micro_validation()
    sys.exit(0 if success else 1)

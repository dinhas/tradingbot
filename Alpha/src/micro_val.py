import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.data_loader import DataLoader as MyDataLoader
from Alpha.src.labeling import Labeler
from Alpha.src.model import AlphaSLModel, multi_head_loss
from Alpha.src.feature_engine import FeatureEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_micro_validation():
    logger.info("--- UNIFIED MICRO-VALIDATION START ---")

    # 1. Data Loading (Phase 1)
    loader = MyDataLoader()
    aligned_df, normalized_df = loader.get_features()
    logger.info(f"Normalized DF shape: {normalized_df.shape}")

    nan_count = normalized_df.isna().sum().sum()
    logger.info(f"Total NaNs in features: {nan_count}")
    if nan_count > 0:
        logger.error("ERROR: NaNs detected in features!")
        return

    # 2. Labeling (Phases 2-4)
    labeler = Labeler()
    asset = 'EURUSD'
    labels_df = labeler.label_data(aligned_df, asset)
    labels_500 = labels_df.head(500)
    logger.info(f"Generated {len(labels_500)} labeled samples.")

    # Class distribution
    logger.info("\nClass distribution (Direction):")
    logger.info(labels_500['direction'].value_counts(normalize=True))

    # First 5 labels
    logger.info("\nFirst 5 labels (Direction, Quality, Meta):")
    logger.info(labels_500[['direction', 'quality', 'meta']].head(5))

    # 3. Feature Prep
    engine = FeatureEngine()
    X = []
    y_dir = []
    y_qual = []
    y_meta = []

    logger.info("\nPreparing features for micro-training...")
    SEQ_LEN = 50
    valid_indices = [i for i, idx in enumerate(labels_500.index) if normalized_df.index.get_loc(idx) >= SEQ_LEN]
    labels_500 = labels_500.iloc[valid_indices]

    for idx in labels_500.index:
        loc = normalized_df.index.get_loc(idx)
        window_df = normalized_df.iloc[loc - SEQ_LEN : loc]
        obs_seq = engine.get_observation_vectorized(window_df, asset)  # (50, 40)
        X.append(obs_seq)
        y_dir.append(labels_500.loc[idx, 'direction'])
        y_qual.append(labels_500.loc[idx, 'quality'])
        y_meta.append(labels_500.loc[idx, 'meta'])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y_dir = torch.tensor(np.array(y_dir), dtype=torch.float32)
    y_qual = torch.tensor(np.array(y_qual), dtype=torch.float32)
    y_meta = torch.tensor(np.array(y_meta), dtype=torch.float32)

    logger.info(f"Feature tensor shape: {X.shape}")  # expect (N, 50, 40)
    logger.info(f"Sample snapshot (last bar, first 5 dims): {X[0, -1, :5].tolist()}")

    # 4. Training (Phase 5)
    dataset = TensorDataset(X, y_dir, y_qual, y_meta)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = AlphaSLModel(input_dim=40, hidden_dim=128, num_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Higher LR for 1 epoch test

    logger.info("\nRunning 1 training epoch...")
    model.train()

    initial_loss = None
    final_loss = None

    for batch_idx, (batch_X, batch_dir, batch_qual, batch_meta) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss, (l_dir, l_qual, l_meta) = multi_head_loss(outputs, (batch_dir, batch_qual, batch_meta))

        if batch_idx == 0:
            initial_loss = loss.item()
            logger.info(f"Initial loss: {initial_loss:.4f} (Dir: {l_dir.item():.4f}, Qual: {l_qual.item():.4f}, Meta: {l_meta.item():.4f})")

        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    logger.info(f"Final loss: {final_loss:.4f}")

    if final_loss < initial_loss:
        logger.info("CONFIRMED: Loss decreased.")
    else:
        logger.warning("WARNING: Loss did not decrease in 1 epoch (might be due to small dataset/LR).")

    # Gradients check
    has_grad = all(p.grad is not None for p in model.trunk.parameters())
    logger.info(f"Gradients flow to trunk: {has_grad}")

    logger.info("Note: stride=10 in Labeler eliminates overlapping windows by design.")

    logger.info("\n--- MICRO-VALIDATION COMPLETE ---")

if __name__ == "__main__":
    run_micro_validation()

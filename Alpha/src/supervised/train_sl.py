import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import sys
import os
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.supervised.data_loader import DataLoader as SLDataLoader
from Alpha.src.supervised.labeler import Labeler
from Alpha.src.supervised.model import MultiHeadModel, FocalLoss

class SLTrainer:
    def __init__(self, input_dim=40, hidden_dim=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiHeadModel(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.direction_criterion = FocalLoss()
        self.quality_criterion = torch.nn.HuberLoss()
        self.meta_criterion = torch.nn.BCELoss()

    def prepare_data(self, sample_size=500):
        print("Preparing data for training...")
        loader = SLDataLoader()
        raw_df, feat_df = loader.get_features()
        labeler = Labeler(time_barrier=20)

        all_features = []
        all_dir_labels = []
        all_qual_labels = []
        all_meta_labels = []

        assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']

        for asset in assets:
            labels = labeler.label_data(raw_df, asset)

            # Get features for this asset
            # Per-asset features: {asset}_{name}
            # Global features: {name}
            asset_feat_names = []
            # Existing (13) + New (11) + Cross-Asset (5)
            for f in loader.engine.feature_names[:29]:
                asset_feat_names.append(f"{asset}_{f}")
            # Global / Time (11)
            for f in loader.engine.feature_names[29:]:
                asset_feat_names.append(f)

            # Filter feat_df for these columns
            asset_feats = feat_df[asset_feat_names].copy()

            # Align labels with features using indices
            common_idx = labels.index.intersection(asset_feats.index)

            all_features.append(asset_feats.loc[common_idx].values)
            all_dir_labels.append(labels.loc[common_idx, 'direction'].values)
            all_qual_labels.append(labels.loc[common_idx, 'quality_score'].values)
            all_meta_labels.append(labels.loc[common_idx, 'meta_label'].values)

        X = np.concatenate(all_features, axis=0)
        y_dir = np.concatenate(all_dir_labels, axis=0)
        y_qual = np.concatenate(all_qual_labels, axis=0)
        y_meta = np.concatenate(all_meta_labels, axis=0)

        # Shuffle and sample
        indices = np.random.permutation(len(X))
        X = X[indices[:sample_size]]
        y_dir = y_dir[indices[:sample_size]]
        y_qual = y_qual[indices[:sample_size]]
        y_meta = y_meta[indices[:sample_size]]

        # Convert to tensors
        # Direction: map [-1, 0, 1] to [0, 0.5, 1] for Tanh output?
        # Wait, Head A is Tanh, outputs [-1, 1]. Labels are [-1, 0, 1].
        # Focal loss needs probabilities.
        # If we use binary focal loss on Head A, we only train it on meta=1 samples?
        # "Direction -> Focal Loss (if imbalance detected)"
        # Let's map Direction [-1, 1] to binary [0, 1] and only use samples where meta=1 for direction loss.

        X_tensor = torch.FloatTensor(X)
        y_dir_tensor = torch.FloatTensor(y_dir).view(-1, 1)
        y_qual_tensor = torch.FloatTensor(y_qual).view(-1, 1)
        y_meta_tensor = torch.FloatTensor(y_meta).view(-1, 1)

        dataset = TensorDataset(X_tensor, y_dir_tensor, y_qual_tensor, y_meta_tensor)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_dir_loss = 0
        total_qual_loss = 0
        total_meta_loss = 0

        for batch in dataloader:
            X, y_dir, y_qual, y_meta = [b.to(self.device) for b in batch]

            self.optimizer.zero_grad()

            pred_dir, pred_qual, pred_meta = self.model(X)

            # Meta Loss
            loss_meta = self.meta_criterion(pred_meta, y_meta)

            # Direction Loss (Only on samples where a barrier was hit)
            # Or use all samples but map 0 to 0.5 probability?
            # User said "meta = 1 if Direction != 0".
            # Let's use Focal Loss for Direction on all samples?
            # If we use binary focal loss, we need to map labels to [0, 1].
            # y_dir is in [-1, 0, 1].
            # If we map to [0, 0.5, 1], BCE might not be ideal for Focal Loss if it's meant for binary.

            # Assuming Head A distinguishes between Long (1) and Short (-1) when Meta=1.
            mask = (y_meta == 1).squeeze()
            if mask.any():
                # Map [-1, 1] to [0, 1]
                y_dir_binary = (y_dir[mask] + 1) / 2
                # Map Tanh output [-1, 1] to [0, 1]
                pred_dir_binary = (pred_dir[mask] + 1) / 2
                loss_dir = self.direction_criterion(pred_dir_binary, y_dir_binary)
            else:
                loss_dir = torch.tensor(0.0).to(self.device)

            # Quality Loss
            loss_qual = self.quality_criterion(pred_qual, y_qual)

            # Weighted Total Loss
            loss = loss_meta + loss_dir + loss_qual

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_dir_loss += loss_dir.item()
            total_qual_loss += loss_qual.item()
            total_meta_loss += loss_meta.item()

        return total_loss / len(dataloader), total_dir_loss / len(dataloader), total_qual_loss / len(dataloader), total_meta_loss / len(dataloader)

def micro_test_phase_5():
    print("\n--- Phase 5 Micro-test (Part 2: Training) ---")
    trainer = SLTrainer()
    dl = trainer.prepare_data(sample_size=500)

    print("Running 1 epoch...")
    loss, dir_loss, qual_loss, meta_loss = trainer.train_epoch(dl)

    print(f"Epoch 1 Loss: {loss:.4f}")
    print(f"  - Direction Loss: {dir_loss:.4f}")
    print(f"  - Quality Loss: {qual_loss:.4f}")
    print(f"  - Meta Loss: {meta_loss:.4f}")

    assert np.isfinite(loss), "Loss is NaN or Inf!"
    print("Phase 5 Micro-test PASSED.")

if __name__ == "__main__":
    micro_test_phase_5()

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from datetime import datetime
from tqdm import tqdm
import gc

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Alpha.src.data_loader import DataLoader as MyDataLoader
from Alpha.src.labeling import LabelingEngine
from Alpha.src.model import AlphaSLModel, multi_head_loss
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

class AlphaDataset(Dataset):
    def __init__(self, features_path, labels_path, indices=None):
        self.features = np.load(features_path, mmap_mode='r')
        labels_data = np.load(labels_path)
        self.directions = labels_data['direction']
        self.qualities = labels_data['quality']
        self.metas = labels_data['meta']

        self.indices = indices if indices is not None else np.arange(len(self.features))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            torch.from_numpy(self.features[real_idx].copy()),
            torch.tensor(self.directions[real_idx], dtype=torch.float32),
            torch.tensor(self.qualities[real_idx], dtype=torch.float32),
            torch.tensor(self.metas[real_idx], dtype=torch.float32)
        )

def generate_dataset(data_dir, output_dir, smoke_test=False, micro_test=False):
    """Generates labeled dataset for all assets efficiently."""
    logger.info(f"Generating dataset from {data_dir}...")
    loader = MyDataLoader(data_dir=data_dir)
    labeler = LabelingEngine()
    engine = FeatureEngine()

    aligned_df, normalized_df = loader.get_features()

    all_X_list = []
    all_y_dir_list = []
    all_y_qual_list = []
    all_y_meta_list = []

    total_samples_collected = 0
    max_samples = 500 if micro_test else (1000 if smoke_test else None)

    for asset in loader.assets:
        if max_samples and total_samples_collected >= max_samples:
            break

        logger.info(f"Processing labels for {asset}...")
        labels_df = labeler.label_data(aligned_df, asset, enforce_non_overlap=True)

        if max_samples:
            labels_df = labels_df.head(max_samples - total_samples_collected)

        common_indices = labels_df.index.intersection(normalized_df.index)
        if len(common_indices) == 0:
            continue

        filtered_norm_df = normalized_df.loc[common_indices]
        filtered_labels_df = labels_df.loc[common_indices]

        asset_X = engine.get_observation_vectorized(filtered_norm_df, asset)

        all_X_list.append(asset_X)
        all_y_dir_list.append(filtered_labels_df['direction'].values)
        all_y_qual_list.append(filtered_labels_df['quality'].values)
        all_y_meta_list.append(filtered_labels_df['meta'].values)

        total_samples_collected += len(asset_X)

    X_np = np.concatenate(all_X_list, axis=0).astype(np.float32)
    y_dir_np = np.concatenate(all_y_dir_list).astype(np.float32)
    y_qual_np = np.concatenate(all_y_qual_list).astype(np.float32)
    y_meta_np = np.concatenate(all_y_meta_list).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, "features.npy")
    labels_path = os.path.join(output_dir, "labels.npz")

    np.save(features_path, X_np)
    np.savez(labels_path, direction=y_dir_np, quality=y_qual_np, meta=y_meta_np)

    logger.info(f"Dataset saved to {output_dir}. Total samples: {len(X_np)}")

    if micro_test:
        logger.info(f"MICRO-TEST: Feature shape: {X_np.shape}")
        logger.info(f"MICRO-TEST: Direction distribution: {pd.Series(y_dir_np).value_counts(normalize=True).to_dict()}")
        logger.info(f"MICRO-TEST: Quality stats: min={y_qual_np.min():.4f}, max={y_qual_np.max():.4f}, mean={y_qual_np.mean():.4f}")
        logger.info(f"MICRO-TEST: Meta distribution: {pd.Series(y_meta_np).value_counts(normalize=True).to_dict()}")

    return features_path, labels_path

def train_model(features_path, labels_path, model_save_path, micro_test=False):
    """Trains the AlphaSLModel."""
    logger.info(f"Starting training...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    BATCH_SIZE = 16384 if not micro_test else 32
    LEARNING_RATE = 1e-3
    EPOCHS = 50 if not micro_test else 1

    total_samples = len(np.load(features_path, mmap_mode='r'))
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    split = int(0.95 * total_samples) if not micro_test else int(0.8 * total_samples)
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = AlphaDataset(features_path, labels_path, indices=train_indices)
    val_dataset = AlphaDataset(features_path, labels_path, indices=val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AlphaSLModel(input_dim=40, head_a_type='tanh').to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Check for imbalance for weight calculation if needed
    labels_data = np.load(labels_path)
    y_dir = labels_data['direction']
    counts = pd.Series(y_dir).value_counts(normalize=True)
    logger.info(f"Direction distribution: {counts.to_dict()}")

    # Requirement: If class imbalance > 80%: Use Focal Loss (but we are using Tanh/MSE now)
    # For Tanh/MSE, we can use sample weights, but let's stick to weighted loss if needed.
    # weights=(dir, qual, meta)
    task_weights = (2.0, 0.5, 1.0)

    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        epoch_dir_loss = 0
        epoch_qual_loss = 0
        epoch_meta_loss = 0

        for batch in pbar:
            b_X, b_dir, b_qual, b_meta = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(b_X)
                    loss, (l_dir, l_qual, l_meta) = multi_head_loss(outputs, (b_dir, b_qual, b_meta), weights=task_weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(b_X)
                loss, (l_dir, l_qual, l_meta) = multi_head_loss(outputs, (b_dir, b_qual, b_meta), weights=task_weights)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            epoch_dir_loss += l_dir.item()
            epoch_qual_loss += l_qual.item()
            epoch_meta_loss += l_meta.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        if micro_test:
            logger.info(f"MICRO-TEST: Per-head losses: Direction={epoch_dir_loss/len(train_loader):.4f}, Quality={epoch_qual_loss/len(train_loader):.4f}, Meta={epoch_meta_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                b_X, b_dir, b_qual, b_meta = [t.to(DEVICE) for t in batch]
                outputs = model(b_X)
                loss, _ = multi_head_loss(outputs, (b_dir, b_qual, b_meta), weights=task_weights)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        logger.info(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model")

    logger.info("Training complete.")

def main():
    parser = argparse.ArgumentParser(description="Alpha Layer Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory containing OHLCV parquet files")
    parser.add_argument("--skip-gen", action="store_true", help="Skip dataset generation")
    parser.add_argument("--smoke-test", action="store_true", help="Run with limited samples")
    parser.add_argument("--micro-test", action="store_true", help="Run mandatory micro-test (500 samples, 1 epoch)")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(base_dir, args.data_dir))

    dataset_dir = os.path.join(base_dir, "data", "training_set")
    if args.micro_test:
        dataset_dir = os.path.join(base_dir, "data", "micro_test_set")

    model_path = os.path.join(base_dir, "models", "alpha_model.pth")

    if not args.skip_gen:
        generate_dataset(args.data_dir, dataset_dir, smoke_test=args.smoke_test, micro_test=args.micro_test)

    features_path = os.path.join(dataset_dir, "features.npy")
    labels_path = os.path.join(dataset_dir, "labels.npz")

    if os.path.exists(features_path) and os.path.exists(labels_path):
        train_model(features_path, labels_path, model_path, micro_test=args.micro_test)
    else:
        logger.error(f"Dataset not found in {dataset_dir}. Cannot train.")

if __name__ == "__main__":
    main()

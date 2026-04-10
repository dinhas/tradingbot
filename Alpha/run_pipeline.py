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
import subprocess

# Add project root to sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(base_dir, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Alpha.src.data_loader import DataLoader as MyDataLoader
from Alpha.src.labeling import Labeler
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
SEQ_LEN = 50

class AlphaDataset(Dataset):
    def __init__(self, features_path, labels_path, indices=None):
        # Use memory mapping to avoid loading large datasets fully into RAM
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
            torch.from_numpy(self.features[real_idx].copy()), # .copy() to make it writable for torch
            torch.tensor(self.directions[real_idx], dtype=torch.float32),
            torch.tensor(self.qualities[real_idx], dtype=torch.float32),
            torch.tensor(self.metas[real_idx], dtype=torch.float32)
        )

def generate_dataset(data_dir, output_dir, smoke_test=False):
    """Generates labeled dataset for all assets efficiently using vectorization."""
    logger.info(f"Generating dataset from {data_dir}...")
    loader = MyDataLoader(data_dir=data_dir)
    labeler = Labeler()
    engine = FeatureEngine()

    # 1. Get raw and normalized features
    aligned_df, normalized_df = loader.get_features()

    all_X_list = []
    all_y_dir_list = []
    all_y_qual_list = []
    all_y_meta_list = []

    for asset in loader.assets:
        logger.info(f"Processing labels for {asset}...")
        try:
            labels_df = labeler.label_data(aligned_df, asset)
        except Exception as e:
            logger.warning(f"Failed to label data for {asset}: {e}")
            continue

        if smoke_test:
            labels_df = labels_df.head(1000)

        # Efficiently filter normalized_df to match labels_df indices
        common_indices = labels_df.index.intersection(normalized_df.index)
        if len(common_indices) == 0:
            logger.warning(f"No common indices found for {asset}")
            continue

        filtered_labels_df = labels_df.loc[common_indices]

        logger.info(f"Extracting vectorized features for {len(filtered_labels_df)} samples of {asset}...")

        # VECTORIZED EXTRACTION
        asset_features = engine.get_observation_vectorized(normalized_df, asset)

        # Build windows
        indices_in_norm = [normalized_df.index.get_loc(idx) for idx in common_indices]

        asset_X_seqs = []
        valid_mask = []
        for idx_in_norm in indices_in_norm:
            if idx_in_norm < SEQ_LEN:
                valid_mask.append(False)
                continue
            window = asset_features[idx_in_norm - SEQ_LEN : idx_in_norm]
            asset_X_seqs.append(window)
            valid_mask.append(True)

        if not asset_X_seqs:
            logger.warning(f"No valid windows for {asset} (all indices < SEQ_LEN)")
            continue

        all_X_list.append(np.stack(asset_X_seqs))
        all_y_dir_list.append(filtered_labels_df['direction'].values[valid_mask])
        all_y_qual_list.append(filtered_labels_df['quality'].values[valid_mask])
        all_y_meta_list.append(filtered_labels_df['meta'].values[valid_mask])

    if not all_X_list:
        logger.error("No data generated for any asset.")
        return None, None

    # 2. Convert to numpy efficiently
    X_np = np.concatenate(all_X_list, axis=0).astype(np.float32)
    y_dir_np = np.concatenate(all_y_dir_list).astype(np.float32)
    y_qual_np = np.concatenate(all_y_qual_list).astype(np.float32)
    y_meta_np = np.concatenate(all_y_meta_list).astype(np.float32)

    # Clear lists to free RAM
    del all_X_list, all_y_dir_list, all_y_qual_list, all_y_meta_list
    gc.collect()

    # 3. Save as binary numpy files
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, "features.npy")
    labels_path = os.path.join(output_dir, "labels.npz")

    np.save(features_path, X_np)
    np.savez(labels_path, direction=y_dir_np, quality=y_qual_np, meta=y_meta_np)

    logger.info(f"Dataset saved to {output_dir}. Total samples: {len(X_np)}")

    return features_path, labels_path

def train_model(features_path, labels_path, model_save_path):
    """Trains the AlphaSLModel with optimizations."""
    logger.info(f"Starting training...")

    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Configuration
    BATCH_SIZE = 2048
    LEARNING_RATE = 1e-3
    EPOCHS = 10

    # 1. Dataset Split
    # Peek at features to get total length
    total_samples = len(np.load(features_path, mmap_mode='r'))
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    split = int(0.95 * total_samples)
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = AlphaDataset(features_path, labels_path, indices=train_indices)
    val_dataset = AlphaDataset(features_path, labels_path, indices=val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 2. Initialize Model
    model = AlphaSLModel(input_dim=40, hidden_dim=256, num_layers=2).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS
    )

    # Check for CUDA availability for GradScaler
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    best_val_loss = float('inf')
    early_stop_patience = 5
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            b_X, b_dir, b_qual, b_meta = [t.to(DEVICE, non_blocking=True) for t in batch]

            optimizer.zero_grad(set_to_none=True)

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(b_X)
                    loss, _ = multi_head_loss(outputs, (b_dir, b_qual, b_meta))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(b_X)
                loss, _ = multi_head_loss(outputs, (b_dir, b_qual, b_meta))
                loss.backward()
                optimizer.step()

            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # 3. Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                b_X, b_dir, b_qual, b_meta = [t.to(DEVICE, non_blocking=True) for t in batch]
                if scaler:
                    with torch.amp.autocast('cuda'):
                        outputs = model(b_X)
                        loss, _ = multi_head_loss(outputs, (b_dir, b_qual, b_meta))
                else:
                    outputs = model(b_X)
                    loss, _ = multi_head_loss(outputs, (b_dir, b_qual, b_meta))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model with Val Loss: {avg_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    logger.info("Training complete.")

def main():
    parser = argparse.ArgumentParser(description="Alpha Layer Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory containing OHLCV parquet files")
    parser.add_argument("--skip-gen", action="store_true", help="Skip dataset generation")
    parser.add_argument("--smoke-test", action="store_true", help="Run with limited samples")
    
    # Resolved Logic: Keep both to maintain backward compatibility but default to running
    parser.add_argument("--run-post-backtest", action="store_true", help="Run threshold optimizer and combined backtest after training")
    parser.add_argument("--skip-post-backtest", action="store_true", help="Skip threshold optimizer and combined backtest after training")
    
    parser.add_argument("--risk-model", type=str, default="Risklayer/models/ppo_risk_model_final_v2_opt.zip", help="Path to RL risk model")
    parser.add_argument("--risk-scaler", type=str, default="Risklayer/models/rl_risk_scaler.pkl", help="Path to RL risk scaler")
    parser.add_argument("--backtest-data-dir", type=str, default="backtest/data", help="Backtest data directory")
    parser.add_argument("--backtest-output-dir", type=str, default="backtest/results", help="Backtest results output directory")
    parser.add_argument("--backtest-max-steps", type=int, default=2000, help="Max steps for post-training backtest")
    
    args = parser.parse_args()
    
    # Logic: Run backtest UNLESS explicitly skipped, or if run-post-backtest is toggled
    run_post_backtest = (not args.skip_post_backtest) or args.run_post_backtest
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(base_dir, args.data_dir))

    dataset_dir = os.path.join(base_dir, "data", "training_set")
    model_path = os.path.join(base_dir, "models", "alpha_model.pth")
    
    if not args.skip_gen:
        if not os.path.exists(args.data_dir):
            logger.error(f"Data directory not found at {args.data_dir}")
            return
        res = generate_dataset(args.data_dir, dataset_dir, smoke_test=args.smoke_test)
        if res == (None, None):
            return
    
    features_path = os.path.join(dataset_dir, "features.npy")
    labels_path = os.path.join(dataset_dir, "labels.npz")
    
    if os.path.exists(features_path) and os.path.exists(labels_path):
        train_model(features_path, labels_path, model_path)

        if run_post_backtest:
            alpha_model_relpath = os.path.relpath(model_path, os.path.dirname(base_dir))
            logger.info("Starting post-training threshold optimization...")
            optimizer_cmd = [
                sys.executable, "-m", "backtest.optimize_thresholds",
                "--alpha-model", alpha_model_relpath,
                "--risk-model", args.risk_model,
                "--risk-scaler", args.risk_scaler,
                "--data-dir", args.backtest_data_dir,
            ]
            logger.info(f"Running: {' '.join(optimizer_cmd)}")
            try:
                subprocess.run(optimizer_cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Threshold optimization failed: {e}")

            logger.info("Starting post-training combined backtest...")
            backtest_cmd = [
                sys.executable, "-m", "backtest.backtest_combined",
                "--alpha-model", alpha_model_relpath,
                "--risk-model", args.risk_model,
                "--risk-scaler", args.risk_scaler,
                "--data-dir", args.backtest_data_dir,
                "--output-dir", args.backtest_output_dir,
                "--max-steps", str(args.backtest_max_steps),
                "--episodes", "1",
            ]
            logger.info(f"Running: {' '.join(backtest_cmd)}")
            try:
                subprocess.run(backtest_cmd, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Combined backtest failed: {e}")
        else:
            logger.info("Skipping post-training optimizer/backtest.")
    else:
        logger.error(f"Dataset not found in {dataset_dir}. Cannot train.")

if __name__ == "__main__":
    main()

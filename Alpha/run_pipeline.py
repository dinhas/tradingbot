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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    def __init__(self, features_path, labels_path, indices=None, seq_len=SEQ_LEN):
        self.features = np.load(features_path, mmap_mode='r')
        labels_data = np.load(labels_path)
        self.directions = labels_data['direction']
        self.qualities = labels_data['quality']
        self.metas = labels_data['meta']
        self.seq_len = seq_len

        if indices is not None:
            # Filter out any index that doesn't have a full lookback window
            self.indices = indices[indices >= seq_len]
        else:
            self.indices = np.arange(seq_len, len(self.features))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # Return a sequence window of shape (seq_len, 40)
        window = self.features[real_idx - self.seq_len : real_idx].copy()  # .copy() for writability
        return (
            torch.from_numpy(window),                                                    # (seq_len, 40)
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
        labels_df = labeler.label_data(aligned_df, asset)
        
        if smoke_test:
            labels_df = labels_df.head(1000)
            
        # Efficiently filter normalized_df to match labels_df indices
        common_indices = labels_df.index.intersection(normalized_df.index)
        if len(common_indices) == 0:
            logger.warning(f"No common indices found for {asset}")
            continue
            
        filtered_norm_df = normalized_df.loc[common_indices]
        filtered_labels_df = labels_df.loc[common_indices]
        
        logger.info(f"Extracting vectorized features for {len(filtered_labels_df)} samples of {asset}...")
        
        # VECTORIZED EXTRACTION
        asset_X = engine.get_observation_vectorized(filtered_norm_df, asset)
        
        all_X_list.append(asset_X)
        all_y_dir_list.append(filtered_labels_df['direction'].values)
        all_y_qual_list.append(filtered_labels_df['quality'].values)
        all_y_meta_list.append(filtered_labels_df['meta'].values)
            
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
    """Trains the AlphaSLModel with optimizations for large datasets (2.5M rows)."""
    logger.info(f"Starting optimized large-scale training...")
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Configuration for large scale training
    BATCH_SIZE = 8192  # Sequences are (50, 40)
    LEARNING_RATE = 1e-3
    EPOCHS = 100 
    NUM_GPUS = torch.cuda.device_count()
    
    # 1. Dataset Split
    total_samples = len(np.load(features_path, mmap_mode='r'))
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    split = int(0.95 * total_samples) 
    train_indices, val_indices = indices[:split], indices[split:]

    # Compute inverse-frequency class weights for direction labels
    all_dir_labels = np.load(labels_path)['direction']
    train_dir_labels = all_dir_labels[train_indices]
    class_counts = np.bincount((train_dir_labels + 1).astype(int), minlength=3).astype(np.float32)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 3
    alpha_dir_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    logger.info(f"Direction class weights: {class_weights}")

    train_dataset = AlphaDataset(features_path, labels_path, indices=train_indices)
    val_dataset = AlphaDataset(features_path, labels_path, indices=val_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # 2. Initialize Model and Multi-GPU
    model = AlphaSLModel(input_dim=40, hidden_dim=256, num_layers=2).to(DEVICE)
    if NUM_GPUS > 1:
        logger.info(f"Using {NUM_GPUS} GPUs for training.")
        model = torch.nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS,
        pct_start=0.3
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
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
            
            with torch.amp.autocast('cuda'):
                outputs = model(b_X)
                loss, _ = multi_head_loss(
                    outputs, (b_dir, b_qual, b_meta), alpha_dir=alpha_dir_tensor
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # 3. Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                b_X, b_dir, b_qual, b_meta = [t.to(DEVICE, non_blocking=True) for t in batch]
                with torch.amp.autocast('cuda'):
                    outputs = model(b_X)
                    loss, _ = multi_head_loss(outputs, (b_dir, b_qual, b_meta), alpha_dir=alpha_dir_tensor)
                    val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            unwrapped_model = model.module if NUM_GPUS > 1 else model
            torch.save(unwrapped_model.state_dict(), model_save_path)
            logger.info(f"Saved best model with Val Loss: {avg_val_loss:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    logger.info("Optimized large-scale training complete.")

def main():
    parser = argparse.ArgumentParser(description="Alpha Layer Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory containing OHLCV parquet files")
    parser.add_argument("--skip-gen", action="store_true", help="Skip dataset generation")
    parser.add_argument("--smoke-test", action="store_true", help="Run with limited samples")
    parser.add_argument("--run-post-backtest", action="store_true", help="Run threshold optimizer and combined backtest after training (legacy flag; now enabled by default)")
    parser.add_argument("--skip-post-backtest", action="store_true", help="Skip threshold optimizer and combined backtest after training")
    parser.add_argument("--risk-model", type=str, default="Risklayer/models/ppo_risk_model_final_v2_opt.zip", help="Path to RL risk model")
    parser.add_argument("--risk-scaler", type=str, default="Risklayer/models/rl_risk_scaler.pkl", help="Path to RL risk scaler")
    parser.add_argument("--backtest-data-dir", type=str, default="backtest/data", help="Backtest data directory")
    parser.add_argument("--backtest-output-dir", type=str, default="backtest/results", help="Backtest results output directory")
    parser.add_argument("--backtest-max-steps", type=int, default=2000, help="Max steps for post-training backtest")
    
    args = parser.parse_args()
    
    # Logic: Run backtest unless explicitly skipped
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
        generate_dataset(args.data_dir, dataset_dir, smoke_test=args.smoke_test)
    
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
            subprocess.run(optimizer_cmd, check=True)

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
            subprocess.run(backtest_cmd, check=True)
        else:
            logger.info("Skipping post-training optimizer/backtest (--skip-post-backtest enabled).")
    else:
        logger.error(f"Dataset not found in {dataset_dir}. Cannot train.")

if __name__ == "__main__":
    main()
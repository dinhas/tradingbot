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

class AlphaDataset(Dataset):
    def __init__(self, features_path, labels_path, indices=None):
        # Use memory mapping to avoid loading 2.5M rows into RAM
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
    BATCH_SIZE = 16384 # Increased for 2.5M rows and multi-GPU throughput
    LEARNING_RATE = 1e-3
    EPOCHS = 100 # Increased to 100 epochs
    NUM_GPUS = torch.cuda.device_count()
    
    # 1. Dataset Split
    # Peek at features to get total length
    total_samples = len(np.load(features_path, mmap_mode='r'))
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    split = int(0.95 * total_samples) # 5% val is plenty for 2.5M (125k samples)
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_dataset = AlphaDataset(features_path, labels_path, indices=train_indices)
    val_dataset = AlphaDataset(features_path, labels_path, indices=val_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, # Reduced to match system suggestion
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, # Reduced to match system suggestion
        pin_memory=True
    )
    
    # 2. Initialize Model and Multi-GPU
    model = AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4).to(DEVICE)
    if NUM_GPUS > 1:
        logger.info(f"Using {NUM_GPUS} GPUs for training.")
        model = torch.nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # OneCycleLR with optimized pct_start for large data
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS,
        pct_start=0.3 # Longer warm-up
    )
    
    # Modern FP16 Mixed Precision API
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
            
            # Use modern autocast API
            with torch.amp.autocast('cuda'):
                outputs = model(b_X)
                # Task weighting: Direction and Meta are critical
                loss, _ = multi_head_loss(
                    outputs, (b_dir, b_qual, b_meta)
                )
            
            scaler.scale(loss).backward()
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
                    loss, _ = multi_head_loss(outputs, (b_dir, b_qual, b_meta))
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

    logger.info("Optimized training complete.")

def main():
    parser = argparse.ArgumentParser(description="Alpha Layer Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory containing OHLCV parquet files")
    parser.add_argument("--skip-gen", action="store_true", help="Skip dataset generation")
    parser.add_argument("--smoke-test", action="store_true", help="Run with limited samples")
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Ensure data_dir is absolute
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
    else:
        logger.error(f"Dataset not found in {dataset_dir}. Cannot train.")

if __name__ == "__main__":
    main()

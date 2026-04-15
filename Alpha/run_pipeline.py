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
from Alpha.src.model import AlphaSLModel, direction_loss
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

# BEST-PRACTICE SETTINGS
SEQUENCE_LENGTH = 50 
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DROPOUT = 0.3

class AlphaDataset(Dataset):
    def __init__(self, features_path, labels_path, indices=None, seq_len=SEQUENCE_LENGTH):
        self.features = np.load(features_path, mmap_mode='r')
        labels_data = np.load(labels_path)
        self.directions = labels_data['direction']
        self.seq_len = seq_len
        
        if indices is not None:
            self.indices = [idx for idx in indices if idx >= seq_len - 1]
        else:
            self.indices = np.arange(seq_len - 1, len(self.features))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_seq = self.features[real_idx - self.seq_len + 1 : real_idx + 1].copy()
        y_dir = self.directions[real_idx]
        return (torch.from_numpy(x_seq), torch.tensor(y_dir, dtype=torch.float32))

def generate_dataset(data_dir, output_dir, smoke_test=False):
    logger.info(f"Generating dataset from {data_dir}...")
    loader = MyDataLoader(data_dir=data_dir)
    labeler = Labeler()
    engine = FeatureEngine()

    aligned_df, normalized_df = loader.get_features()
    
    all_X_list = []
    all_y_dir_list = []

    for asset in loader.assets:
        logger.info(f"Processing labels for {asset}...")
        labels_df = labeler.label_data(aligned_df, asset)
        if smoke_test: labels_df = labels_df.head(1000)
            
        common_indices = labels_df.index.intersection(normalized_df.index)
        if len(common_indices) == 0: continue
            
        filtered_norm_df = normalized_df.loc[common_indices]
        filtered_labels_df = labels_df.loc[common_indices]
        
        asset_X = engine.get_observation_vectorized(filtered_norm_df, asset)
        all_X_list.append(asset_X)
        all_y_dir_list.append(filtered_labels_df['direction'].values)
            
    X_np = np.concatenate(all_X_list, axis=0).astype(np.float32)
    y_dir_np = np.concatenate(all_y_dir_list).astype(np.float32)
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "features.npy"), X_np)
    np.savez(os.path.join(output_dir, "labels.npz"), direction=y_dir_np)
    
    logger.info(f"Dataset generated. Total rows: {len(X_np)}")
    return os.path.join(output_dir, "features.npy"), os.path.join(output_dir, "labels.npz")

def train_model(features_path, labels_path, model_save_path):
    logger.info("Starting optimized LSTM training...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # 1. TIME-SERIES AWARE SPLIT (No random shuffle for validation)
    total_samples = len(np.load(features_path, mmap_mode='r'))
    all_indices = np.arange(total_samples)
    valid_indices = all_indices[SEQUENCE_LENGTH - 1:]
    
    split = int(0.90 * len(valid_indices))
    train_indices = valid_indices[:split]
    val_indices = valid_indices[split:] # Last 10% is validation (forward-walk style)
    
    # We shuffle training, but NOT validation
    np.random.shuffle(train_indices)
    
    train_dataset = AlphaDataset(features_path, labels_path, indices=train_indices)
    val_dataset = AlphaDataset(features_path, labels_path, indices=val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = AlphaSLModel(input_dim=17, lstm_units=64, dense_units=32, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    early_stop_patience = 10
    epochs_no_improve = 0
    
    for epoch in range(100): # Max 100 epochs
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            b_X, b_dir = [t.to(DEVICE, non_blocking=True) for t in batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits = model(b_X)
                loss = direction_loss(logits, b_dir)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                b_X, b_dir = [t.to(DEVICE, non_blocking=True) for t in batch]
                logits = model(b_X)
                loss = direction_loss(logits, b_dir)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        logger.info(f"Epoch {epoch+1}: Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="../data")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(base_dir, args.data_dir))
    dataset_dir = os.path.join(base_dir, "data", "training_set")
    model_path = os.path.join(base_dir, "models", "alpha_model.pth")
    
    if not args.skip_gen:
        generate_dataset(data_dir, dataset_dir, smoke_test=args.smoke_test)
    
    train_model(os.path.join(dataset_dir, "features.npy"), 
                os.path.join(dataset_dir, "labels.npz"), 
                model_path)

if __name__ == "__main__":
    main()

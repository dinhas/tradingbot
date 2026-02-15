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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from risk_model_sl import RiskModelSL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Use environment variable if set by pipeline, otherwise use default
DATASET_PATH = os.environ.get("SL_DATASET_PATH", os.path.join(os.path.dirname(__file__), "data", "sl_risk_dataset.parquet"))
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Optimizing for 2x T4 GPUs
BATCH_SIZE = 4096 # High batch size for multi-GPU throughput
LEARNING_RATE = 2e-4 # Slightly higher LR for larger batch
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()

# Loss Weights
W_SL = 1.0
W_TP = 1.0
W_SIZE = 5.0 

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class RiskDataset(Dataset):
    def __init__(self, features, sl_targets, tp_targets, size_targets):
        # Keep on CPU until batching
        self.features = features
        self.sl_targets = sl_targets.reshape(-1, 1)
        self.tp_targets = tp_targets.reshape(-1, 1)
        self.size_targets = size_targets.reshape(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]), 
            torch.from_numpy(self.sl_targets[idx]), 
            torch.from_numpy(self.tp_targets[idx]), 
            torch.from_numpy(self.size_targets[idx])
        )

def train():
    logger.info(f"Starting Supervised Learning Training on {DEVICE} with {NUM_GPUS} GPUs")
    
    if not os.path.exists(DATASET_PATH):
        alt_path = os.path.join(os.path.dirname(__file__), "data", "test_sl_risk_dataset.parquet")
        if os.path.exists(alt_path):
            DATASET_PATH_TO_USE = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    else:
        DATASET_PATH_TO_USE = DATASET_PATH

    # 1. Load Data
    logger.info(f"Loading dataset from {DATASET_PATH_TO_USE}...")
    df = pd.read_parquet(DATASET_PATH_TO_USE)
    
    X = np.stack(df['features'].values).astype(np.float32)
    y_sl = df['target_sl_mult'].values.astype(np.float32)
    y_tp = df['target_tp_mult'].values.astype(np.float32)
    y_size = df['target_size'].values.astype(np.float32)
    
    # 2. Split and Scale
    X_train, X_val, y_sl_train, y_sl_val, y_tp_train, y_tp_val, y_size_train, y_size_val = train_test_split(
        X, y_sl, y_tp, y_size, test_size=0.10, random_state=42 # 10% val is enough for 1.8M
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    scaler_path = os.path.join(MODELS_DIR, "sl_risk_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 3. Create DataLoaders with Multi-Process Loading
    train_dataset = RiskDataset(X_train_scaled, y_sl_train, y_tp_train, y_size_train)
    val_dataset = RiskDataset(X_val_scaled, y_sl_val, y_tp_val, y_size_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 4. Initialize Model and Multi-GPU Wrapper
    model = RiskModelSL(input_dim=40).to(DEVICE)
    if NUM_GPUS > 1:
        logger.info(f"Using {NUM_GPUS} GPUs with DataParallel")
        model = nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # FP16 Mixed Precision Scaler
    scaler_amp = torch.cuda.amp.GradScaler()
    
    mse_loss = nn.MSELoss()
    best_val_loss = float('inf')
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            features, sl_target, tp_target, size_target = [b.to(DEVICE, non_blocking=True) for b in batch]
            
            optimizer.zero_grad(set_to_none=True)
            
            # Autocast for Mixed Precision
            with torch.cuda.amp.autocast():
                preds = model(features)
                loss_sl = mse_loss(preds['sl'], sl_target)
                loss_tp = mse_loss(preds['tp'], tp_target)
                loss_size = mse_loss(preds['size'], size_target)
                total_loss = (W_SL * loss_sl) + (W_TP * loss_tp) + (W_SIZE * loss_size)
            
            # Backprop with AMP
            scaler_amp.scale(total_loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            
            train_loss += total_loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features, sl_target, tp_target, size_target = [b.to(DEVICE, non_blocking=True) for b in batch]
                with torch.cuda.amp.autocast():
                    preds = model(features)
                    v_total = (W_SL * mse_loss(preds['sl'], sl_target)) + \
                              (W_TP * mse_loss(preds['tp'], tp_target)) + \
                              (W_SIZE * mse_loss(preds['size'], size_target))
                    val_loss += v_total.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the underlying model (unwrapped from DataParallel)
            save_model = model.module if NUM_GPUS > 1 else model
            torch.save(save_model.state_dict(), os.path.join(MODELS_DIR, "risk_model_sl_best.pth"))
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features, sl_target, tp_target, size_target = [b.to(DEVICE) for b in batch]
                preds = model(features)
                
                v_loss_sl = mse_loss(preds['sl'], sl_target)
                v_loss_tp = mse_loss(preds['tp'], tp_target)
                v_loss_size = mse_loss(preds['size'], size_target)
                
                v_total = (W_SL * v_loss_sl) + (W_TP * v_loss_tp) + (W_SIZE * v_loss_size)
                val_loss += v_total.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "risk_model_sl_best.pth"))
            logger.info(f"New best model saved with val_loss: {avg_val_loss:.6f}")

    # Save Final Model
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "risk_model_sl_final.pth"))
    logger.info("Training Complete.")

if __name__ == "__main__":
    train()

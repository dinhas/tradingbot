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
DATASET_PATH = os.environ.get("SL_DATASET_PATH", os.path.join(os.path.dirname(__file__), "data", "sl_risk_dataset.parquet"))
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Optimizing for T4 GPUs
BATCH_SIZE = 8192 
LEARNING_RATE = 1e-3 
EPOCHS = int(os.environ.get("EPOCHS", 100))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()

# Loss Weights
W_SL = 1.0
W_TP = 1.0
W_SIZE = 2.0 
W_PROB = 1.0

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class RiskDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets # Dict of numpy arrays

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]), 
            torch.tensor(self.targets['sl_mult'][idx], dtype=torch.float32), 
            torch.tensor(self.targets['tp_mult'][idx], dtype=torch.float32), 
            torch.tensor(self.targets['size_factor'][idx], dtype=torch.float32),
            torch.tensor(self.targets['prob_tp_first'][idx], dtype=torch.float32)
        )

def train():
    logger.info(f"Starting Multi-Task Risk Model Training on {DEVICE}")
    
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}")
        return

    # 1. Load Data
    logger.info(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    
    X = np.stack(df['features'].values).astype(np.float32)
    targets = {
        'sl_mult': df['target_sl_mult'].values.astype(np.float32),
        'tp_mult': df['target_tp_mult'].values.astype(np.float32),
        'size_factor': df['target_size_factor'].values.astype(np.float32),
        'prob_tp_first': df['target_prob_tp_first'].values.astype(np.float32)
    }
    
    # 2. Split and Scale
    X_train, X_val, y_sl_train, y_sl_val, y_tp_train, y_tp_val, y_size_train, y_size_val, y_prob_train, y_prob_val = train_test_split(
        X, targets['sl_mult'], targets['tp_mult'], targets['size_factor'], targets['prob_tp_first'], 
        test_size=0.10, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    scaler_path = os.path.join(MODELS_DIR, "sl_risk_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 3. Create DataLoaders
    train_targets = {'sl_mult': y_sl_train, 'tp_mult': y_tp_train, 'size_factor': y_size_train, 'prob_tp_first': y_prob_train}
    val_targets = {'sl_mult': y_sl_val, 'tp_mult': y_tp_val, 'size_factor': y_size_val, 'prob_tp_first': y_prob_val}
    
    train_dataset = RiskDataset(X_train_scaled, train_targets)
    val_dataset = RiskDataset(X_val_scaled, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    
    # 4. Initialize Model
    model = RiskModelSL(input_dim=48).to(DEVICE)
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    best_val_loss = float('inf')
    scaler_amp = torch.cuda.amp.GradScaler()
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            features, sl_target, tp_target, size_target, prob_target = [b.to(DEVICE, non_blocking=True) for b in batch]
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                preds = model(features)
                
                # Loss Breakdown
                loss_sl = l1_loss(preds['sl_mult'].squeeze(), sl_target)
                loss_tp = l1_loss(preds['tp_mult'].squeeze(), tp_target)
                loss_size = mse_loss(preds['size_factor'].squeeze(), size_target)
                loss_prob = bce_loss(preds['prob_tp_first'].squeeze(), prob_target)
                
                total_loss = (W_SL * loss_sl) + (W_TP * loss_tp) + (W_SIZE * loss_size) + (W_PROB * loss_prob)
            
            scaler_amp.scale(total_loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
            
            train_loss += total_loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features, sl_target, tp_target, size_target, prob_target = [b.to(DEVICE, non_blocking=True) for b in batch]
                with torch.cuda.amp.autocast():
                    preds = model(features)
                    v_sl = l1_loss(preds['sl_mult'].squeeze(), sl_target)
                    v_tp = l1_loss(preds['tp_mult'].squeeze(), tp_target)
                    v_size = mse_loss(preds['size_factor'].squeeze(), size_target)
                    v_prob = bce_loss(preds['prob_tp_first'].squeeze(), prob_target)
                    v_total = (W_SL * v_sl) + (W_TP * v_tp) + (W_SIZE * v_size) + (W_PROB * v_prob)
                    val_loss += v_total.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            unwrapped = model.module if NUM_GPUS > 1 else model
            torch.save(unwrapped.state_dict(), os.path.join(MODELS_DIR, "risk_model_sl_best.pth"))
            logger.info("Saved best model.")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "risk_model_sl_final.pth"))
    logger.info("Training Complete.")

if __name__ == "__main__":
    train()

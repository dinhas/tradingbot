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
W_EXEC = 0.5
W_EV = 1.5

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class RiskDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.from_numpy(features)
        self.targets = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], {k: v[idx] for k, v in self.targets.items()}

def train():
    logger.info(f"Starting Multi-Task Risk Model Training on {DEVICE}")
    
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}")
        return

    # 1. Load Data
    logger.info(f"Loading dataset from {DATASET_PATH}...")
    df = pd.read_parquet(DATASET_PATH)
    
    X = np.stack(df['features'].values).astype(np.float32)
    target_names = [
        'target_sl_mult', 'target_tp_mult', 'target_size',
        'target_execution_buffer_mult', 'target_expected_value', 'target_tp_first'
    ]
    
    # Check for missing targets
    for t in target_names:
        if t not in df.columns:
            logger.error(f"Missing target column: {t}")
            return
            
    y = {t: df[t].values.astype(np.float32) for t in target_names}
    
    # 2. Split and Scale
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.10, random_state=42)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = {k: v[train_idx] for k, v in y.items()}
    y_val = {k: v[val_idx] for k, v in y.items()}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    scaler_path = os.path.join(MODELS_DIR, "sl_risk_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 3. Create DataLoaders
    train_dataset = RiskDataset(X_train_scaled, y_train)
    val_dataset = RiskDataset(X_val_scaled, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
    
    # 4. Initialize Model
    model = RiskModelSL(input_dim=48).to(DEVICE)
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    mse_loss = nn.MSELoss()
    huber_loss = nn.HuberLoss()
    # Fix: Use BCEWithLogitsLoss for numerical stability with AMP
    bce_logits_loss = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    # Fix: Use torch.amp.GradScaler for newer PyTorch versions
    scaler_amp = torch.amp.GradScaler('cuda')
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            features = features.to(DEVICE, non_blocking=True)
            targets = {k: v.to(DEVICE, non_blocking=True) for k, v in targets.items()}
            
            optimizer.zero_grad(set_to_none=True)
            
            # Fix: Use torch.amp.autocast
            with torch.amp.autocast('cuda'):
                preds = model(features)
                
                loss_sl = huber_loss(preds['sl_mult'].squeeze(), targets['target_sl_mult'])
                loss_tp = huber_loss(preds['tp_mult'].squeeze(), targets['target_tp_mult'])
                loss_size = mse_loss(preds['size'].squeeze(), targets['target_size'])
                # Fix: Use logits output and BCEWithLogitsLoss
                loss_prob = bce_logits_loss(preds['prob_tp_first_logits'].squeeze(), targets['target_tp_first'])
                loss_exec = huber_loss(preds['execution_buffer'].squeeze(), targets['target_execution_buffer_mult'])
                loss_ev = mse_loss(preds['expected_value'].squeeze(), targets['target_expected_value'])
                
                total_loss = (W_SL * loss_sl) + (W_TP * loss_tp) + (W_SIZE * loss_size) + \
                             (W_PROB * loss_prob) + (W_EXEC * loss_exec) + (W_EV * loss_ev)
            
            scaler_amp.scale(total_loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            scheduler.step()
            
            train_loss += total_loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(DEVICE, non_blocking=True)
                targets = {k: v.to(DEVICE, non_blocking=True) for k, v in targets.items()}
                
                with torch.amp.autocast('cuda'):
                    preds = model(features)
                    v_sl = huber_loss(preds['sl_mult'].squeeze(), targets['target_sl_mult'])
                    v_tp = huber_loss(preds['tp_mult'].squeeze(), targets['target_tp_mult'])
                    v_size = mse_loss(preds['size'].squeeze(), targets['target_size'])
                    # Fix: Use logits output and BCEWithLogitsLoss
                    v_prob = bce_logits_loss(preds['prob_tp_first_logits'].squeeze(), targets['target_tp_first'])
                    v_exec = huber_loss(preds['execution_buffer'].squeeze(), targets['target_execution_buffer_mult'])
                    v_ev = mse_loss(preds['expected_value'].squeeze(), targets['target_expected_value'])
                    
                    v_total = (W_SL * v_sl) + (W_TP * v_tp) + (W_SIZE * v_size) + \
                              (W_PROB * v_prob) + (W_EXEC * v_exec) + (W_EV * v_ev)
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

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

BATCH_SIZE = 1024
LEARNING_RATE = 1e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loss Weights
W_SL = 1.0
W_TP = 1.0
W_SIZE = 5.0 # Higher weight on sizing (the filter)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

class RiskDataset(Dataset):
    def __init__(self, features, sl_targets, tp_targets, size_targets):
        self.features = torch.FloatTensor(features)
        self.sl_targets = torch.FloatTensor(sl_targets).view(-1, 1)
        self.tp_targets = torch.FloatTensor(tp_targets).view(-1, 1)
        self.size_targets = torch.FloatTensor(size_targets).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.sl_targets[idx], self.tp_targets[idx], self.size_targets[idx]

def train():
    logger.info(f"Starting Supervised Learning Training on {DEVICE}")
    
    if not os.path.exists(DATASET_PATH):
        # Fallback to test dataset if main one isn't there yet
        alt_path = os.path.join(os.path.dirname(__file__), "data", "test_sl_risk_dataset.parquet")
        if os.path.exists(alt_path):
            DATASET_PATH_TO_USE = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    else:
        DATASET_PATH_TO_USE = DATASET_PATH

    # 1. Load and Prepare Data
    logger.info(f"Loading dataset from {DATASET_PATH_TO_USE}...")
    df = pd.read_parquet(DATASET_PATH_TO_USE)
    
    # Extract features from the 'features' list column
    X = np.stack(df['features'].values).astype(np.float32)
    y_sl = df['target_sl_mult'].values.astype(np.float32)
    y_tp = df['target_tp_mult'].values.astype(np.float32)
    y_size = df['target_size'].values.astype(np.float32)
    
    # 2. Split and Scale
    X_train, X_val, y_sl_train, y_sl_val, y_tp_train, y_tp_val, y_size_train, y_size_val = train_test_split(
        X, y_sl, y_tp, y_size, test_size=0.15, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save Scaler
    scaler_path = os.path.join(MODELS_DIR, "sl_risk_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # 3. Create DataLoaders
    train_dataset = RiskDataset(X_train_scaled, y_sl_train, y_tp_train, y_size_train)
    val_dataset = RiskDataset(X_val_scaled, y_sl_val, y_tp_val, y_size_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 4. Initialize Model, Loss, Optimizer
    model = RiskModelSL(input_dim=60).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss() # Using BCE for size if treated as probability, though MSE works too
    
    best_val_loss = float('inf')
    
    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            features, sl_target, tp_target, size_target = [b.to(DEVICE) for b in batch]
            
            optimizer.zero_grad()
            preds = model(features)
            
            # Multi-Task Loss
            loss_sl = mse_loss(preds['sl'], sl_target)
            loss_tp = mse_loss(preds['tp'], tp_target)
            
            # Size Loss: Only penalize heavily on trades where we should have taken a position
            # or where we took a position wrongly
            loss_size = mse_loss(preds['size'], size_target) 
            
            total_loss = (W_SL * loss_sl) + (W_TP * loss_tp) + (W_SIZE * loss_size)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            
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

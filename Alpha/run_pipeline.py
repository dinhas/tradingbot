import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

def generate_dataset(data_dir, output_path, smoke_test=False):
    """Generates labeled dataset for all assets."""
    logger.info(f"Generating dataset from {data_dir}...")
    loader = MyDataLoader(data_dir=data_dir)
    labeler = Labeler()
    engine = FeatureEngine()

    # 1. Get raw and normalized features
    aligned_df, normalized_df = loader.get_features()
    
    all_X = []
    all_y_dir = []
    all_y_qual = []
    all_y_meta = []

    for asset in loader.assets:
        logger.info(f"Processing labels for {asset}...")
        labels_df = labeler.label_data(aligned_df, asset)
        
        if smoke_test:
            labels_df = labels_df.head(1000)
            
        logger.info(f"Extracting features for {len(labels_df)} samples of {asset}...")
        for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Features {asset}"):
            if idx not in normalized_df.index:
                continue
                
            current_step_data = normalized_df.loc[idx]
            obs = engine.get_observation(current_step_data, {}, asset)
            
            all_X.append(obs)
            all_y_dir.append(row['direction'])
            all_y_qual.append(row['quality'])
            all_y_meta.append(row['meta'])
            
    # Convert to numpy and save as parquet for persistence
    X_np = np.array(all_X, dtype=np.float32)
    y_dir_np = np.array(all_y_dir, dtype=np.float32)
    y_qual_np = np.array(all_y_qual, dtype=np.float32)
    y_meta_np = np.array(all_y_meta, dtype=np.float32)
    
    # Save to parquet
    dataset_df = pd.DataFrame({
        'direction': y_dir_np,
        'quality': y_qual_np,
        'meta': y_meta_np
    })
    # Features as a list of lists in a single column to keep it simple for parquet
    dataset_df['features'] = [x.tolist() for x in X_np]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset_df.to_parquet(output_path)
    logger.info(f"Dataset saved to {output_path}. Total samples: {len(dataset_df)}")
    
    return output_path

def train_model(dataset_path, model_save_path):
    """Trains the AlphaSLModel."""
    logger.info(f"Starting training from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    X = torch.tensor(np.array(df['features'].tolist()), dtype=torch.float32)
    y_dir = torch.tensor(df['direction'].values, dtype=torch.float32)
    y_qual = torch.tensor(df['quality'].values, dtype=torch.float32)
    y_meta = torch.tensor(df['meta'].values, dtype=torch.float32)
    
    dataset = TensorDataset(X, y_dir, y_qual, y_meta)
    
    # Train/Val Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    model = AlphaSLModel(input_dim=40).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            b_X, b_dir, b_qual, b_meta = [t.to(DEVICE) for t in batch]
            
            optimizer.zero_grad()
            outputs = model(b_X)
            loss, _ = multi_head_loss(outputs, (b_dir, b_qual, b_meta))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                b_X, b_dir, b_qual, b_meta = [t.to(DEVICE) for t in batch]
                outputs = model(b_X)
                loss, _ = multi_head_loss(outputs, (b_dir, b_qual, b_meta))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Saved best model to {model_save_path}")

    logger.info("Training complete.")

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

    dataset_path = os.path.join(base_dir, "models", "alpha_dataset.parquet")
    model_path = os.path.join(base_dir, "models", "alpha_model.pth")
    
    if not args.skip_gen:
        if not os.path.exists(args.data_dir):
            logger.error(f"Data directory not found at {args.data_dir}")
            return
        generate_dataset(args.data_dir, dataset_path, smoke_test=args.smoke_test)
    
    if os.path.exists(dataset_path):
        train_model(dataset_path, model_path)
    else:
        logger.error(f"Dataset not found at {dataset_path}. Cannot train.")

if __name__ == "__main__":
    main()

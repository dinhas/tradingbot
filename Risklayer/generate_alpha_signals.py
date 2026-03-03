import os
import sys
import pandas as pd
import numpy as np
import torch
import logging
from datetime import datetime
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Alpha.src.data_loader import DataLoader
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.model import AlphaSLModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_signals(data_dir="data", model_path="Alpha/models/alpha_model.pth", output_file="data/alpha_signals_2016_2025.parquet"):
    """
    Loads the Alpha model, calculates features, and saves rows that pass thresholds.
    Thresholds: Meta > 0.78, Quality > 0.30
    """
    logger.info(f"Loading Alpha model from {model_path}...")
    model = AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4).to(DEVICE)
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    loader = DataLoader(data_dir=data_dir)
    engine = FeatureEngine()

    logger.info("Loading and preprocessing data...")
    aligned_df, normalized_df = loader.get_features()
    
    all_signals = []

    for asset in loader.assets:
        logger.info(f"Processing signals for {asset}...")
        
        # Get vectorized observations for this asset
        # Note: normalized_df contains the features used for input
        X_np = engine.get_observation_vectorized(normalized_df, asset)
        X_tensor = torch.from_numpy(X_np).to(DEVICE)
        
        # Inference in batches to avoid OOM
        batch_size = 16384
        all_meta = []
        all_qual = []
        all_dir = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size]
                dir_logits, quality, meta_logits = model(batch)
                
                # Convert logits to scores
                meta_score = torch.sigmoid(meta_logits).cpu().numpy()
                qual_score = quality.cpu().numpy()
                # Get predicted direction (most likely class)
                direction = (torch.argmax(dir_logits, dim=1) - 1).cpu().numpy()
                
                all_meta.append(meta_score)
                all_qual.append(qual_score)
                all_dir.append(direction)
        
        meta_scores = np.concatenate(all_meta).squeeze()
        qual_scores = np.concatenate(all_qual).squeeze()
        pred_dirs = np.concatenate(all_dir).squeeze()
        
        # Create a combined DF for this asset with signals
        asset_df = aligned_df.copy()
        
        # We only care about columns for this asset and global features
        # Plus the calculated signals
        asset_signals = pd.DataFrame({
            'meta_score': meta_scores,
            'quality_score': qual_scores,
            'pred_direction': pred_dirs,
            'asset_name': asset
        }, index=aligned_df.index)
        
        # Filter by thresholds
        mask = (asset_signals['meta_score'] > 0.78) & (asset_signals['quality_score'] > 0.30)
        filtered_asset_signals = asset_signals[mask]
        
        if len(filtered_asset_signals) == 0:
            logger.warning(f"No signals passed thresholds for {asset}")
            continue
            
        # Join with original data (prices, ATR, etc.)
        # We need the asset-specific columns and global features
        combined = pd.concat([filtered_asset_signals, aligned_df.loc[filtered_asset_signals.index]], axis=1)
        
        # We'll also keep the timestamp as a column just in case
        combined['timestamp'] = combined.index
        
        all_signals.append(combined)
        logger.info(f"Generated {len(combined)} signals for {asset}")

    if not all_signals:
        logger.error("No signals generated across all assets!")
        return

    final_signals_df = pd.concat(all_signals, axis=0).sort_index()
    
    # Save to parquet
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_signals_df.to_parquet(output_file)
    logger.info(f"Saved {len(final_signals_df)} signals to {output_file} (Total signals: {len(final_signals_df)})")

if __name__ == "__main__":
    generate_signals()

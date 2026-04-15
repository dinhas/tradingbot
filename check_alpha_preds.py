
import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.src.model import AlphaSLModel
from Alpha.src.data_loader import DataLoader as AlphaDataLoader
from Alpha.src.feature_engine import FeatureEngine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_predictions():
    model_path = PROJECT_ROOT / "Alpha/models/alpha_model.pth"
    data_dir = PROJECT_ROOT / "backtest/data"
    
    loader = AlphaDataLoader(data_dir=str(data_dir))
    try:
        aligned_df, normalized_df = loader.get_features()
    except Exception as e:
        print(f"Error loading features: {e}")
        return

    input_dim = 17
    model = AlphaSLModel(input_dim=input_dim, lstm_units=64, dense_units=32, dropout=0.3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    feature_engine = FeatureEngine()
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    
    session_mask = (normalized_df['is_late_session'] == 1)
    session_indices = np.where(session_mask.values)[0]
    print(f"Total rows: {len(normalized_df)}")
    print(f"Session rows: {len(session_indices)}")
    
    sequence_length = 50
    check_limit = len(session_indices) - sequence_length
    print(f"Checking {check_limit} session steps...")
    
    asset_obs = {
        asset: feature_engine.get_observation_vectorized(normalized_df, asset)
        for asset in assets
    }
    
    all_probs_list = []
    batch_size = 256
    
    for i in tqdm(range(0, check_limit, batch_size)):
        batch_end = min(i + batch_size, check_limit)
        
        for asset in assets:
            batch_sequences = []
            for j in range(i, batch_end):
                seq_indices = session_indices[j : j + sequence_length]
                batch_sequences.append(asset_obs[asset][seq_indices])
            
            batch_tensor = torch.from_numpy(np.array(batch_sequences)).to(DEVICE)
            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs_list.append(probs)
    
    all_probs = np.concatenate(all_probs_list, axis=0)
    directions = np.argmax(all_probs, axis=1) - 1
    
    unique, counts = np.unique(directions, return_counts=True)
    print("Direction counts (argmax):")
    for d, c in zip(unique, counts):
        print(f"  {d}: {c}")
        
    for i, label in enumerate(['Short', 'Neutral', 'Long']):
        class_probs = all_probs[:, i]
        print(f"{label} probs: min={np.min(class_probs):.4f}, max={np.max(class_probs):.4f}, mean={np.mean(class_probs):.4f}")

if __name__ == "__main__":
    check_predictions()

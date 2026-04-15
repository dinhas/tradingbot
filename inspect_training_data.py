
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.src.data_loader import DataLoader as MyDataLoader
from Alpha.src.labeling import Labeler
from Alpha.src.feature_engine import FeatureEngine

def inspect_full_dataset():
    data_dir = os.path.join(os.getcwd(), "data")
    print(f"Loading data from {data_dir}...")
    
    loader = MyDataLoader(data_dir=data_dir)
    labeler = Labeler()
    engine = FeatureEngine()

    try:
        aligned_df, normalized_df = loader.get_features()
    except Exception as e:
        print(f"Error: {e}")
        return

    SESSION_COL = "is_late_session"
    results = []

    for asset in loader.assets:
        print(f"Processing {asset}...")
        labels_df = labeler.label_data(aligned_df, asset)
        
        common_indices = labels_df.index.intersection(normalized_df.index)
        filtered_norm_df = normalized_df.loc[common_indices]
        filtered_labels_df = labels_df.loc[common_indices]

        # Apply session filter as done in run_pipeline.py
        if SESSION_COL in filtered_norm_df.columns:
            session_mask = (filtered_norm_df[SESSION_COL] == 1)
            filtered_labels_df = filtered_labels_df.loc[session_mask]
        
        results.append(filtered_labels_df)

    if not results:
        print("No labels generated.")
        return

    full_labels = pd.concat(results)
    directions = full_labels['direction'].values

    total_samples = len(directions)
    buys = np.sum(directions == 1)
    sells = np.sum(directions == -1)
    neutrals_losses = np.sum(directions == 0)

    print("\n" + "="*30)
    print("FULL DATASET STATISTICS")
    print("="*30)
    print(f"Total Session Samples: {total_samples}")
    print(f"Buy Opportunities (1) : {buys} ({(buys/total_samples)*100:.2f}%)")
    print(f"Sell Opportunities (-1): {sells} ({(sells/total_samples)*100:.2f}%)")
    print(f"Neutral / Loss (0)     : {neutrals_losses} ({(neutrals_losses/total_samples)*100:.2f}%)")
    print(f"Total Trade Opps       : {buys + sells}")
    print("="*30)

if __name__ == "__main__":
    inspect_full_dataset()

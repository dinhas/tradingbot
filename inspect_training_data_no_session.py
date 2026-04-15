
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

def inspect_full_dataset_no_session():
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

    results = []

    for asset in loader.assets:
        print(f"Processing {asset} (No Session Limit)...")
        labels_df = labeler.label_data(aligned_df, asset)
        
        # We use ALL labeled data that intersects with our features,
        # intentionally skipping the 'is_late_session' filter.
        common_indices = labels_df.index.intersection(normalized_df.index)
        filtered_labels_df = labels_df.loc[common_indices]
        
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

    print("\n" + "="*40)
    print("FULL DATASET STATISTICS (NO SESSION LIMIT)")
    print("="*40)
    print(f"Total Samples analyzed : {total_samples}")
    print(f"Buy Opportunities (1)  : {buys} ({(buys/total_samples)*100:.2f}%)")
    print(f"Sell Opportunities (-1) : {sells} ({(sells/total_samples)*100:.2f}%)")
    print(f"Neutral / Loss (0)      : {neutrals_losses} ({(neutrals_losses/total_samples)*100:.2f}%)")
    print(f"Total Trade Opps        : {buys + sells}")
    print("="*40)

if __name__ == "__main__":
    inspect_full_dataset_no_session()

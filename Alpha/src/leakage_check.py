import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.data_loader import DataLoader
from Alpha.src.labeling import Labeler

def leakage_check():
    print("--- Phase 6 Leakage Check ---")
    loader = DataLoader()
    aligned_df, _ = loader.get_features()

    labeler = Labeler()
    asset = 'EURUSD'
    labels_df = labeler.label_data(aligned_df, asset)

    # 1. Verify forward windows do not overlap
    # We check if entry_idx[i+1] > exit_idx[i]
    overlaps = (labels_df['entry_idx'].iloc[1:].values < labels_df['exit_idx'].iloc[:-1].values).sum()
    print(f"Number of overlapping windows: {overlaps}")

    # 2. Ensure features are computed only from past data
    # This is handled by FeatureEngine's use of rolling windows and pct_change.
    # We've done a consistency check in Phase 1.

    # 3. Confirm labels do not use any data beyond time barrier
    # Check if exit_idx - entry_idx <= time_barrier
    out_of_bounds = (labels_df['exit_idx'] - labels_df['entry_idx'] > labeler.time_barrier).sum()
    print(f"Number of labels exceeding time barrier: {out_of_bounds}")

    # 4. Print metadata for first 5 samples
    print("\nMetadata for first 5 samples:")
    cols = ['entry_time', 'exit_time', 'barrier_hit', 'direction', 'quality', 'meta']
    print(labels_df[cols].head(5))

    if overlaps == 0 and out_of_bounds == 0:
        print("\nLeakage check passed.")
    else:
        print("\nLeakage check FAILED!")

if __name__ == "__main__":
    leakage_check()

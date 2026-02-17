import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.data_loader import DataLoader
from Alpha.src.labeling import LabelingEngine

def leakage_check():
    print("--- Phase 6 Leakage Check ---")
    loader = DataLoader()
    aligned_df, _ = loader.get_features()

    labeler = LabelingEngine()
    asset = 'EURUSD'
    labels_df = labeler.label_data(aligned_df, asset, enforce_non_overlap=True)

    # 1. Verify forward windows do not overlap
    # We check if entry_time[i+1] > exit_time[i]
    overlaps = 0
    for i in range(len(labels_df) - 1):
        if labels_df.index[i+1] <= labels_df.iloc[i]['exit_time']:
            overlaps += 1

    print(f"Number of overlapping windows: {overlaps}")

    # 2. Print metadata for first 5 samples
    print("\nMetadata for first 5 samples:")
    cols = ['entry_time', 'exit_time', 'barrier_hit', 'direction', 'quality', 'meta']
    # If entry_time is the index, we can use labels_df.index or the column if it exists
    if 'entry_time' not in labels_df.columns:
        labels_df['entry_time'] = labels_df.index

    print(labels_df[cols].head(5))

    if overlaps == 0:
        print("\nLeakage check passed (Non-overlapping events verified).")
    else:
        print(f"\nLeakage check FAILED! Found {overlaps} overlaps.")

if __name__ == "__main__":
    leakage_check()

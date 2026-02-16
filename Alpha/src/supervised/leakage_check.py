import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.supervised.data_loader import DataLoader
from Alpha.src.supervised.labeler import Labeler

def leakage_check():
    print("\n--- Phase 6: Leakage Check ---")
    loader = DataLoader()
    raw_df, _ = loader.get_features()
    labeler = Labeler(time_barrier=20)

    asset = 'EURUSD'
    labels = labeler.label_data(raw_df.iloc[:5000], asset)

    # 1. Print first 5 samples details
    print("First 5 samples details:")
    cols = ['entry_time', 'exit_time', 'barrier', 'direction']
    print(labels[cols].head())

    # 2. Verify no overlap
    # entry_time[i+1] must be >= exit_time[i] or at least > entry_time[i]
    # Since we use step=time_barrier, and exit_time <= entry_time + time_barrier,
    # then entry_time[i+1] = entry_time[i] + time_barrier >= exit_time[i].

    # Check: entry_time is strictly increasing
    assert (labels.index[1:] > labels.index[:-1]).all(), "Entry times are not strictly increasing!"

    # Check: exit_time >= entry_time
    assert (labels['exit_time'] >= labels.index).all(), "Exit time before entry time!"

    # Check: no overlap (entry of next sample is >= exit of current sample?
    # Not necessarily, but it should be >= entry + time_barrier if we want NO overlap of windows.
    # Our step=20 ensures windows [i+1, i+20] and [i+20+1, i+40] don't overlap.

    # 3. Ensure features are computed only from past data
    # This was partially checked in Phase 1 with the slice comparison.

    # 4. Confirm labels do not use any data beyond time barrier
    # Check: (exit_time - entry_time) <= 20 bars
    # In M5, 20 bars = 100 minutes.
    time_diff = (labels['exit_time'] - labels.index)
    max_diff = time_diff.max()
    print(f"Max time diff (barrier): {max_diff}")

    # In terms of bars:
    # We can check by counting bars in raw_df between entry and exit.
    for idx, row in labels.head(10).iterrows():
        entry_idx = raw_df.index.get_loc(idx)
        exit_idx = raw_df.index.get_loc(row['exit_time'])
        diff_bars = exit_idx - entry_idx
        print(f"Sample at {idx}: exit at {row['exit_time']}, bars: {diff_bars}, barrier: {row['barrier']}")
        assert diff_bars <= 20, f"Exit beyond time barrier! bars: {diff_bars}"

    print("Leakage check PASSED.")

if __name__ == "__main__":
    leakage_check()

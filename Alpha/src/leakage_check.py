"""Causality / leakage checks for the current Alpha SL pipeline.

Checks performed:
1. FEATURE CAUSALITY: features computed on truncated data must exactly match
   features computed on full data for the overlapping region. Any mismatch means
   a feature is using future information.
2. LABEL CAUSALITY: labels computed on truncated data must match full-data labels
   for every bar whose forward window (max_bars) is fully inside the truncated data.
3. LABEL SANITY: class distribution and validity ratio report.

Run: python -m Alpha.src.leakage_check
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.data_loader import DataLoader
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.labeling import Labeler

TRUNCATE_BARS = 500     # how many bars to cut off the end for the truncation test
COMPARE_TAIL = 2000     # how many of the last overlapping rows to compare
FEATURE_TOL = 1e-4      # max allowed absolute feature difference (float32 noise)


def leakage_check() -> bool:
    print("--- Alpha Pipeline Leakage Check ---")
    loader = DataLoader()
    raw = loader.load_raw_data()
    if not raw:
        print("ERROR: no raw data found.")
        return False

    # --- 1. Feature causality via truncation test ---
    print("\n[1/3] Feature causality (truncation test)...")
    aligned_full, norm_full = FeatureEngine().preprocess_data({a: df.copy() for a, df in raw.items()})
    truncated = {a: df.iloc[:-TRUNCATE_BARS].copy() for a, df in raw.items()}
    aligned_trunc, norm_trunc = FeatureEngine().preprocess_data(truncated)

    common = norm_trunc.index.intersection(norm_full.index)
    tail = common[-COMPARE_TAIL:]
    cols = [c for c in norm_trunc.columns if c in norm_full.columns]

    diff = np.abs(norm_full.loc[tail, cols].values - norm_trunc.loc[tail, cols].values)
    max_diff = float(np.nanmax(diff))
    features_ok = max_diff < FEATURE_TOL
    if features_ok:
        print(f"  PASS: max feature difference = {max_diff:.2e} (< {FEATURE_TOL})")
    else:
        worst = np.unravel_index(np.nanargmax(diff), diff.shape)
        print(f"  FAIL: max feature difference = {max_diff:.4f} "
              f"in column '{cols[worst[1]]}' at {tail[worst[0]]}. "
              f"A feature is leaking future data!")

    # --- 2. Label causality ---
    print("\n[2/3] Label causality (truncation test)...")
    labeler = Labeler()
    asset = 'EURUSD'
    labels_full = labeler.label_data(aligned_full, asset)
    labels_trunc = labeler.label_data(aligned_trunc, asset)

    # Only bars whose forward window is fully inside the truncated data are comparable.
    safe_index = labels_trunc.index[:-(labeler.max_bars + 1)].intersection(labels_full.index)
    dir_mismatch = int((labels_full.loc[safe_index, 'direction'].values
                        != labels_trunc.loc[safe_index, 'direction'].values).sum())
    valid_mismatch = int((labels_full.loc[safe_index, 'valid'].values
                          != labels_trunc.loc[safe_index, 'valid'].values).sum())
    labels_ok = dir_mismatch == 0 and valid_mismatch == 0
    if labels_ok:
        print(f"  PASS: 0 mismatches across {len(safe_index)} comparable labels.")
    else:
        print(f"  FAIL: direction mismatches={dir_mismatch}, valid mismatches={valid_mismatch}. "
              f"Labels depend on data beyond their forward window!")

    # --- 3. Label sanity report ---
    print("\n[3/3] Label sanity report...")
    valid_mask = labels_full['valid']
    valid_ratio = float(valid_mask.mean())
    print(f"  Valid (tradeable) label ratio: {valid_ratio:.2%}")
    dist = labels_full.loc[valid_mask, 'direction'].value_counts(normalize=True).sort_index()
    print("  Class distribution among valid labels:")
    for cls, frac in dist.items():
        print(f"    {int(cls):+d}: {frac:.2%}")

    all_ok = features_ok and labels_ok
    print("\n" + ("Leakage check PASSED." if all_ok else "Leakage check FAILED!"))
    return all_ok


if __name__ == "__main__":
    ok = leakage_check()
    sys.exit(0 if ok else 1)

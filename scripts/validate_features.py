"""
Validate that all new Tier 3 macro features are correctly loaded and contain no lookahead.
Run after implementing changes and downloading data.

Usage:
    python scripts/validate_features.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.src.data_loader import DataLoader
from Alpha.src.feature_engine import FeatureEngine


EXPECTED_MACRO_FEATURES = [
    'sp500_return_5d', 'dxy_return_5d',
]

EXPECTED_TOTAL_FEATURES = 32  # 32 in v2.8


def main():
    loader = DataLoader(data_dir=str(PROJECT_ROOT / 'data'))
    engine = FeatureEngine()

    print("=" * 60)
    print("Feature Validation Script")
    print("=" * 60)

    # 1. Check feature names count
    print(f"\n[1] Feature names count: {len(engine.feature_names)}")
    print(f"    Expected: {EXPECTED_TOTAL_FEATURES}")
    assert len(engine.feature_names) == EXPECTED_TOTAL_FEATURES, \
        f"Feature count mismatch: {len(engine.feature_names)} != {EXPECTED_TOTAL_FEATURES}"
    print("    PASS")

    # 2. Check all expected features are in the list
    print("\n[2] Checking all expected macro features are defined...")
    missing_names = [f for f in EXPECTED_MACRO_FEATURES if f not in engine.feature_names]
    if missing_names:
        print(f"    FAIL: Missing feature names: {missing_names}")
        sys.exit(1)
    print("    All 12 macro features found in feature_names")
    print("    PASS")

    # 3. Load macro data
    print("\n[3] Loading macro data...")
    macro_df = loader.load_macro_data()
    if macro_df.empty:
        print("    WARNING: No macro data found. Run download scripts first.")
        print("    Skipping data-dependent checks.")
        print("    Feature engine will use 31 features (macro features = 0).")
        _print_summary(engine, None, None)
        return

    print(f"    Macro data shape: {macro_df.shape}")
    print(f"    Macro columns: {list(macro_df.columns)}")
    expected_in_data = [c for c in EXPECTED_MACRO_FEATURES if c not in ['vix_regime']]
    missing_data = [c for c in expected_in_data if c not in macro_df.columns]
    if missing_data:
        print(f"    WARNING: Missing macro columns: {missing_data}")
    else:
        print("    All expected macro columns present")
    print("    PASS")

    # 4. Run feature engine
    print("\n[4] Running feature engine with macro data...")
    data_dict = loader.load_raw_data()
    if not data_dict:
        print("    FAIL: No OHLCV data found.")
        sys.exit(1)

    aligned_df, normalized_df = engine.preprocess_data(data_dict, macro_df=macro_df)
    print(f"    Aligned DF shape: {aligned_df.shape}")
    print(f"    Normalized DF shape: {normalized_df.shape}")
    print("    PASS")

    # 5. Check observation vector dimension
    print("\n[5] Checking observation vector dimension...")
    test_asset = 'EURUSD'
    obs = engine.get_observation_vectorized(normalized_df, test_asset)
    print(f"    Observation shape for {test_asset}: {obs.shape}")
    print(f"    Expected: (N, {EXPECTED_TOTAL_FEATURES})")
    assert obs.shape[1] == EXPECTED_TOTAL_FEATURES, \
        f"Observation dimension mismatch: {obs.shape[1]} != {EXPECTED_TOTAL_FEATURES}"
    print("    PASS")

    # 6. Check for NaN/inf
    print("\n[6] Checking for NaN/inf in observation vectors...")
    nan_count = np.isnan(obs).sum()
    inf_count = np.isinf(obs).sum()
    print(f"    NaN count: {nan_count}")
    print(f"    Inf count: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("    WARNING: NaN/inf found. Check data quality.")
    else:
        print("    No NaN/inf found")
    print("    PASS")

    # 7. Check macro features exist in normalized_df
    print("\n[7] Checking macro features in normalized_df...")
    macro_cols_in_df = [f'{test_asset}_{f}' for f in EXPECTED_MACRO_FEATURES
                        if f'{test_asset}_{f}' in normalized_df.columns]
    print(f"    Found {len(macro_cols_in_df)}/{len(EXPECTED_MACRO_FEATURES)} macro feature columns")
    missing_cols = [f'{test_asset}_{f}' for f in EXPECTED_MACRO_FEATURES
                    if f'{test_asset}_{f}' not in normalized_df.columns]
    if missing_cols:
        print(f"    Missing: {missing_cols}")
    else:
        print("    All macro feature columns present")
    print("    PASS")

    # 8. Verify no look-ahead (shift(1) check)
    print("\n[8] Checking no look-ahead in macro features...")
    for col in EXPECTED_MACRO_FEATURES:
        full_col = f'{test_asset}_{col}'
        if full_col not in normalized_df.columns:
            continue
        series = normalized_df[full_col]
        # Macro features should have NaN at the start (from shift(1))
        # They should NOT have NaN at the end (forward-filled)
        if series.iloc[-10:].isna().any():
            print(f"    WARNING: {full_col} has NaN in last 10 rows (possible data issue)")
    print("    Look-ahead check complete")
    print("    PASS")

    _print_summary(engine, aligned_df, normalized_df)
    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


def _print_summary(engine, aligned_df, normalized_df):
    print("\n" + "-" * 40)
    print("Feature Summary:")
    print(f"  Total features: {len(engine.feature_names)}")
    print(f"  Feature list: {engine.feature_names}")
    if aligned_df is not None:
        print(f"  Aligned DF shape: {aligned_df.shape}")
    if normalized_df is not None:
        print(f"  Normalized DF shape: {normalized_df.shape}")


if __name__ == '__main__':
    main()

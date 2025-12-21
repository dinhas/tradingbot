import pandas as pd
import numpy as np
import sys
from pathlib import Path

def verify_quality(file_path):
    print(f"Deep Quality Check: {file_path}")
    if not Path(file_path).exists():
        print("Error: File not found.")
        return False

    df = pd.read_parquet(file_path)
    feature_cols = [f"feature_{i}" for i in range(60)]
    
    # 1. Expected Placeholders (based on generate_dataset.py)
    # Mapping: f_a(0-9), Groups B,C,D(10-39), f_e(40-49), Group F(50-59)
    expected_constants = {
        'feature_25': 0.5,  # Hurst Placeholder (Group C)
        'feature_40': 0.0,  # entry_atr_distance (Group E)
        'feature_46': 0.0,  # Placeholder (Group E)
        'feature_47': 0.5,  # RSI Placeholder (Group E)
        'feature_48': 0.5,  # BB Placeholder (Group E)
        'feature_49': 0.0,  # MACD Placeholder (Group E)
        'feature_54': 5.0,  # consecutive_direction (Group F)
        'feature_57': 0.0,  # EMA alignment (Group F)
        'feature_58': 0.0,  # Price velocity (Group F)
        'feature_59': 0.0,  # VPT (Group F)
    }

    print("\n1. Verifying Expected Placeholders:")
    for feat, val in expected_constants.items():
        if feat in df.columns:
            actual_unique = df[feat].unique()
            if len(actual_unique) == 1 and np.isclose(actual_unique[0], val):
                print(f"  PASS: {feat} is constant {val} (expected placeholder)")
            else:
                print(f"  WARN: {feat} is NOT constant {val}. Found: {actual_unique[:5]}")
        else:
            print(f"  FAIL: {feat} missing from dataset")

    # 2. Check for UNEXPECTED Constant Features
    variances = df[feature_cols].var()
    constant_features = variances[variances == 0].index.tolist()
    unexpected_constants = [f for f in constant_features if f not in expected_constants]
    
    print(f"\n2. Checking for Unexpected Constant Features:")
    if unexpected_constants:
        print(f"  FAIL: Found {len(unexpected_constants)} unexpected constant features: {unexpected_constants}")
    else:
        print("  PASS: No unexpected constant features found.")

    # 3. Check for Feature Ranges and NaNs
    print("\n3. Data Integrity:")
    nans = df[feature_cols].isna().sum().sum()
    if nans > 0:
        print(f"  FAIL: Found {nans} total NaN values in features.")
    else:
        print("  PASS: No NaN values found.")

    infs = np.isinf(df[feature_cols].values).sum()
    if infs > 0:
        print(f"  FAIL: Found {infs} infinite values.")
    else:
        print("  PASS: No infinite values found.")

    # 4. Correlation Analysis (Leakage Check)
    print("\n4. Feature Correlation with Label (Checking for Leaks):")
    corrs = df[feature_cols].corrwith(df['label']).abs().sort_values(ascending=False)
    high_corr = corrs[corrs > 0.8]
    if not high_corr.empty:
        print(f"  WARN: Highly correlated features detected (possible leakage?):")
        print(high_corr)
    else:
        print("  PASS: No suspiciously high correlations (>0.8) detected.")

    # 5. Label Balance
    print("\n5. Label Balance:")
    label_counts = df['label'].value_counts(normalize=True)
    print(label_counts)
    if label_counts.min() < 0.05:
        print("  WARN: Extreme class imbalance detected!")

    print("\nQuality check complete.")
    return True

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "TradeGuard/data/guard_dataset.parquet"
    verify_quality(path)

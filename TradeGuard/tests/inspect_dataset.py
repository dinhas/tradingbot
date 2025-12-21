import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

def inspect_dataset(file_path):
    print(f"Inspecting file: {file_path}")
    
    if not Path(file_path).exists():
        print(f"Error: File not found at {file_path}")
        return False

    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return False

    print(f"Shape: {df.shape}")
    print("-" * 30)

    # 1. Check Core Columns
    core_cols = ['timestamp', 'asset', 'label']
    missing_core = [c for c in core_cols if c not in df.columns]
    if missing_core:
        print(f"FAIL: Missing core columns: {missing_core}")
        return False
    else:
        print("PASS: Core columns (timestamp, asset, label) present.")

    # 2. Check Feature Columns (Expected 60 features based on FeatureEngine)
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    expected_feature_count = 60
    if len(feature_cols) != expected_feature_count:
        print(f"WARN: Found {len(feature_cols)} feature columns. Expected {expected_feature_count}.")
        # We don't fail here strictly, as feature count might change, but it's good to note.
    else:
        print(f"PASS: Found exactly {expected_feature_count} feature columns.")

    # 3. Check Data Types
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        print(f"FAIL: 'timestamp' column is not datetime. Got {df['timestamp'].dtype}")
        return False
    
    if not pd.api.types.is_numeric_dtype(df['label']):
        print(f"FAIL: 'label' column is not numeric. Got {df['label'].dtype}")
        return False
        
    print("PASS: Data types for core columns appear correct.")

    # 4. Check Data Integrity
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("WARN: Null values found:")
        print(null_counts[null_counts > 0])
    else:
        print("PASS: No null values found.")

    # 5. Label Distribution
    unique_labels = df['label'].unique()
    print(f"Unique Labels: {unique_labels}")
    if not set(unique_labels).issubset({0, 1}):
        print("FAIL: Labels contain values other than 0 and 1.")
        return False
    
    label_dist = df['label'].value_counts(normalize=True)
    print("Label Distribution:")
    print(label_dist)

    # 6. Asset Distribution
    print("Asset Distribution (rows per asset):")
    print(df['asset'].value_counts())

    print("-" * 30)
    print("Inspection Complete. Dataset looks valid.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect TradeGuard Dataset Parquet")
    parser.add_argument("file_path", nargs='?', default="TradeGuard/data/guard_dataset.parquet", help="Path to the parquet file")
    args = parser.parse_args()

    success = inspect_dataset(args.file_path)
    sys.exit(0 if success else 1)

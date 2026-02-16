import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.feature_engine import FeatureEngine

class DataLoader:
    """
    Loads OHLCV data and generates features using the existing FeatureEngine.
    """
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.engine = FeatureEngine()

    def load_raw_data(self) -> dict:
        """
        Loads all OHLCV data from the data directory.
        Returns:
            dict: {asset_name: pd.DataFrame}
        """
        data_dict = {}
        for asset in self.assets:
            file_path = self.data_dir / f"{asset}_5m.parquet"
            if file_path.exists():
                df = pd.read_parquet(file_path)
                # Ensure sorted timestamps
                df = df.sort_index()
                # Remove duplicates
                df = df[~df.index.duplicated(keep='first')]
                # Do NOT forward-fill price columns (as per Phase 1 instructions)
                # But we might need to handle missing values later if any.
                data_dict[asset] = df
            else:
                print(f"Warning: {file_path} not found.")
        return data_dict

    def get_features(self) -> tuple:
        """
        Generates features using the existing feature engine.
        Returns:
            tuple: (raw_df, normalized_feat_df)
        """
        data_dict = self.load_raw_data()
        # raw_df contains un-normalized indicators, normalized_feat_df contains scaled features
        raw_df, normalized_feat_df = self.engine.preprocess_data(data_dict)
        return raw_df, normalized_feat_df

def micro_test_phase_1():
    print("Starting Phase 1 Micro-test...")
    loader = DataLoader()
    raw_df, feat_df = loader.get_features()

    # 1. Print feature tensor shape for 5 rows
    print(f"Feature tensor shape for 5 rows: {feat_df.iloc[:5].shape}")

    # 2. Confirm: Shapes correct, No NaNs
    print(f"Feature DF Shape: {feat_df.shape}")
    nans = feat_df.isnull().sum().sum()
    print(f"Number of NaNs: {nans}")
    assert nans == 0, "Found NaNs in features!"

    # 3. Verify no future leakage (shift checks)
    # Simple check: take a slice, calculate features, take a larger slice, check if first part is same.
    data_dict = loader.load_raw_data()
    asset = 'EURUSD'
    small_data = {a: data_dict[a].iloc[:1000] for a in loader.assets}
    _, feat_small = loader.engine.preprocess_data(small_data)

    large_data = {a: data_dict[a].iloc[:2000] for a in loader.assets}
    _, feat_large = loader.engine.preprocess_data(large_data)

    # Compare first 500 rows (avoiding the end of small_data which might have different normalization if rolling windows are used)
    # FeatureEngine uses rolling(window=50) for normalization.
    diff = (feat_small.iloc[100:500] - feat_large.iloc[100:500]).abs().max().max()
    print(f"Future leakage diff (should be 0): {diff}")
    assert diff < 1e-5, "Potential future leakage detected in feature calculation!"

    print("Phase 1 Micro-test PASSED.")

if __name__ == "__main__":
    micro_test_phase_1()

import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Alpha.src.feature_engine import FeatureEngine

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']

    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Loads OHLCV data for all assets from Parquet files.
        Ensures sorted timestamps and no duplicates.
        """
        data_dict = {}
        for asset in self.assets:
            file_path = self.data_dir / f"{asset}_5m.parquet"
            if not file_path.exists():
                print(f"Warning: Data file for {asset} not found at {file_path}")
                continue

            df = pd.read_parquet(file_path)

            # Ensure timestamp is index and sorted
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            df.sort_index(inplace=True)

            # Remove duplicates
            df = df[~df.index.duplicated(keep='first')]

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {asset} data is missing some OHLCV columns.")

            data_dict[asset] = df

        return data_dict

    def get_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates features using FeatureEngine.
        Returns (aligned_df, normalized_df).
        """
        data_dict = self.load_raw_data()
        engine = FeatureEngine()
        aligned_df, normalized_df = engine.preprocess_data(data_dict)
        return aligned_df, normalized_df

if __name__ == "__main__":
    loader = DataLoader()
    aligned_df, normalized_df = loader.get_features()
    print(f"Aligned DF shape: {aligned_df.shape}")
    print(f"Normalized DF shape: {normalized_df.shape}")
    print(f"Columns: {normalized_df.columns[:10]}...")

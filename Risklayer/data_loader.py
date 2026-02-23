import pandas as pd
import numpy as np
import os
import logging
from typing import Dict
from .config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_path: str = config.DATA_PATH):
        self.data_path = data_path
        self.assets = config.ASSETS

    def load_all_data(self, limit_bars: int = None) -> Dict[str, pd.DataFrame]:
        """Loads and validates OHLCV data for all assets."""
        data_dict = {}
        for asset in self.assets:
            file_path = os.path.join(self.data_path, f"{asset}_5m.parquet")
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} not found. Skipping {asset}.")
                continue

            df = pd.read_parquet(file_path)
            if limit_bars:
                df = df.iloc[-limit_bars:]
            df = self._validate_and_clean(df, asset)
            data_dict[asset] = df
            logger.info(f"Loaded {len(df)} bars for {asset}")

        if not data_dict:
            raise FileNotFoundError(f"No valid data files found in {self.data_path}")

        return data_dict

    def _validate_and_clean(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Validates columns, sorts chronologically, and handles missing values."""
        # Standardize column names (lowercase)
        df.columns = [c.lower() for c in df.columns]

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column {col} in {asset} data")

        # Ensure timestamp index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                # Try to convert index if it's string/int
                df.index = pd.to_datetime(df.index)

        # Sort by index
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Handle missing values
        # Fill zero volume if it's NaN
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)

        # Forward fill price data
        df = df.ffill()

        # Drop any remaining NaNs (at the beginning)
        df.dropna(subset=['close'], inplace=True)

        # Convert to float32
        for col in required_cols:
            df[col] = df[col].astype(np.float32)

        return df

if __name__ == "__main__":
    loader = DataLoader()
    try:
        data = loader.load_all_data()
        for asset, df in data.items():
            print(f"{asset}: {df.shape}")
    except Exception as e:
        print(f"Error: {e}")

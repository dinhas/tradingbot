import pandas as pd
import numpy as np
import logging
from typing import Optional
from Risklayer.config import config

class DataLoader:
    def __init__(self, filepath: Optional[str] = None):
        self.filepath = filepath or config.DATA_PATH
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self) -> pd.DataFrame:
        """Loads and validates OHLCV data."""
        self.logger.info(f"Loading data from {self.filepath}...")
        try:
            df = pd.read_parquet(self.filepath)
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

        # Validation
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Sort chronologically
        df.sort_index(inplace=True)

        # Handle missing values
        df = df.ffill().dropna()

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure we have the date range 2016-2024 if possible
        # Actually, just use whatever is in the file but log the range
        self.logger.info(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")

        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    data = loader.load_data()
    print(data.head())

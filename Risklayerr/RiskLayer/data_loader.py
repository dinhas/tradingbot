import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Optional
from .config import RiskConfig

class DataLoader:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Loads OHLCV data for all configured assets."""
        data_dict = {}
        for asset in self.config.ASSETS:
            filepath = os.path.join(self.config.DATA_PATH, f"{asset}_{self.config.TIMEFRAME}.parquet")
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                df = self._preprocess_df(df)
                data_dict[asset] = df
                self.logger.info(f"Loaded {len(df)} rows for {asset}")
            else:
                self.logger.warning(f"File not found: {filepath}. Generating synthetic data.")
                data_dict[asset] = self._generate_synthetic_data()

        return data_dict

    def _preprocess_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates, sorts, and cleans the DataFrame."""
        # Ensure required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Ensure timestamp index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Handle missing values
        df = df.ffill().bfill()

        # Filter by year
        df = df[(df.index.year >= self.config.START_YEAR) & (df.index.year <= self.config.END_YEAR)]

        return df

    def _generate_synthetic_data(self, n_rows: int = 10000) -> pd.DataFrame:
        """Generates synthetic OHLCV data for testing/sanity."""
        np.random.seed(self.config.SEED)
        dates = pd.date_range(start='2020-01-01', periods=n_rows, freq='5min')

        close = 1.1000 + np.cumsum(np.random.normal(0, 0.0005, n_rows))
        high = close + np.random.uniform(0, 0.0010, n_rows)
        low = close - np.random.uniform(0, 0.0010, n_rows)
        open_ = close + np.random.normal(0, 0.0002, n_rows)
        volume = np.random.randint(100, 1000, n_rows)

        df = pd.DataFrame({
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }, index=dates)

        return df

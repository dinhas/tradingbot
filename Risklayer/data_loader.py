import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Optional
from Risklayer.config import config

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = config.DATA_DIR):
        self.data_dir = data_dir
        self.assets = config.ASSETS

    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Loads and cleans OHLCV data for all assets."""
        data_dict = {}
        for asset in self.assets:
            filepath = os.path.join(self.data_dir, f"{asset}_5m.parquet")
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                continue

            df = pd.read_parquet(filepath)
            df = self._clean_data(df)
            df = self._add_base_indicators(df)
            data_dict[asset] = df
            logger.info(f"Loaded {len(df)} rows for {asset}")

        return data_dict

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validates, sorts, and handles missing values."""
        # Ensure timestamp is index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        df.sort_index(inplace=True)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]

        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Handle missing values
        df = df[required] # Keep only OHLCV for now
        df = df.ffill().dropna()

        return df

    def _add_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds ATR and Volatility Percentile."""
        # ATR (14-period)
        high = df['high']
        low = df['low']
        close = df['close']

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        df['atr'] = tr.rolling(window=14).mean()

        # Volatility Percentile (based on ATR relative to 500-period window)
        df['vol_percentile'] = df['atr'].rolling(window=500).apply(
            lambda x: (x[-1] >= x).mean() if len(x) > 0 else 0.5,
            raw=True
        )

        return df.dropna()

    def align_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aligns multiple assets into a single DataFrame with multi-index columns."""
        # Find common index
        common_index = None
        for asset, df in data_dict.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        aligned_parts = []
        for asset, df in data_dict.items():
            df_subset = df.loc[common_index].copy()
            df_subset.columns = [f"{asset}_{col}" for col in df_subset.columns]
            aligned_parts.append(df_subset)

        return pd.concat(aligned_parts, axis=1)

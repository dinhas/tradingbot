"""
Frozen Feature Normalizer for Live Execution.

Problem: During training, _normalize_features uses rolling 50-period statistics.
         In live, we only have 300 bars, and the first 50 have NaN statistics.
         This causes the same price data to produce DIFFERENT normalized features.

Solution: Capture median/IQR statistics from training data and apply them statically
          during live inference, matching exactly what the model saw during training.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FrozenFeatureNormalizer:
    """
    Stores frozen normalization statistics from training data.
    Apply these during live inference to match training distribution.
    """
    
    def __init__(self):
        self.stats = {}  # {feature_name: {'median': float, 'iqr': float}}
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, cols_to_normalize: list):
        """
        Calculate and store normalization statistics from training data.
        Uses the LAST 10000 rows to capture recent market regime.
        
        Args:
            df: Training DataFrame with all features
            cols_to_normalize: List of column names to normalize
        """
        # Use last 10000 rows (or all if less) to capture recent statistics
        sample_df = df.tail(10000)
        
        for col in cols_to_normalize:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in training data")
                continue
                
            series = sample_df[col].dropna()
            if len(series) < 100:
                logger.warning(f"Column {col} has insufficient data ({len(series)} rows)")
                continue
            
            median = series.median()
            q75 = series.quantile(0.75)
            q25 = series.quantile(0.25)
            iqr = q75 - q25
            
            # Prevent division by zero
            if iqr < 1e-8:
                iqr = series.std()
                if iqr < 1e-8:
                    iqr = 1.0
            
            self.stats[col] = {
                'median': float(median),
                'iqr': float(iqr),
                'q25': float(q25),
                'q75': float(q75),
                'mean': float(series.mean()),
                'std': float(series.std())
            }
            
        self.is_fitted = True
        logger.info(f"Fitted normalizer on {len(self.stats)} features from {len(sample_df)} samples")
        
    def transform(self, series_or_value, col_name: str):
        """
        Transform a value or series using frozen statistics.
        
        Args:
            series_or_value: pd.Series, np.array, or scalar
            col_name: Feature name to get statistics for
            
        Returns:
            Normalized value(s) clipped to [-5, 5]
        """
        if not self.is_fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() first or load().")
            
        if col_name not in self.stats:
            # Feature not in normalizer, return as-is
            return series_or_value
            
        stat = self.stats[col_name]
        median = stat['median']
        iqr = stat['iqr']
        
        normalized = (series_or_value - median) / iqr
        return np.clip(normalized, -5, 5)
    
    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform an entire DataFrame using frozen statistics.
        Only transforms columns that exist in both the DataFrame and normalizer.
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            Normalized DataFrame (copy)
        """
        result = df.copy()
        
        for col in df.columns:
            if col in self.stats:
                result[col] = self.transform(df[col], col)
                
        return result
    
    def save(self, path: str):
        """Save normalizer to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'stats': self.stats,
                'is_fitted': self.is_fitted
            }, f)
        logger.info(f"Saved feature normalizer to {path}")
        
    def load(self, path: str):
        """Load normalizer from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.stats = data['stats']
            self.is_fitted = data['is_fitted']
        logger.info(f"Loaded feature normalizer with {len(self.stats)} features from {path}")
        
    @classmethod
    def from_file(cls, path: str) -> 'FrozenFeatureNormalizer':
        """Create normalizer instance from saved file."""
        normalizer = cls()
        normalizer.load(path)
        return normalizer


def extract_and_save_normalizer(data_dir: str, output_path: str):
    """
    Utility function to extract normalization statistics from training data
    and save them for live use.
    
    Usage:
        python -c "from Alpha.src.feature_normalizer import extract_and_save_normalizer; extract_and_save_normalizer('data', 'models/checkpoints/alpha/feature_normalizer.pkl')"
    """
    from .feature_engine import FeatureEngine
    
    # Load training data
    data = {}
    data_path = Path(data_dir)
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    
    for asset in assets:
        file_path = data_path / f"{asset}_5m.parquet"
        if file_path.exists():
            # Load only the last 20,000 rows to save RAM
            full_df = pd.read_parquet(file_path)
            data[asset] = full_df.tail(20000)
            logger.info(f"Loaded last 20,000 rows for {asset}")
        else:
            logger.error(f"Training data not found: {file_path}")
            return
    
    # Process with feature engine
    fe = FeatureEngine()
    # preprocess_data will now work on 100k rows instead of 500k+
    _, processed_df = fe.preprocess_data(data)
    
    # Define columns to normalize (same as in FeatureEngine._normalize_features)
    cols_to_normalize = []
    for asset in assets:
        cols_to_normalize.extend([
            f"{asset}_close", f"{asset}_ema_9", f"{asset}_ema_21",
            f"{asset}_return_1", f"{asset}_return_12",
            f"{asset}_atr_14", f"{asset}_atr_ratio",
            f"{asset}_rsi_14", f"{asset}_macd_hist",
            f"{asset}_volume_ratio"
        ])
    cols_to_normalize.extend(['risk_on_score', 'asset_dispersion', 'market_volatility'])
    
    # Fit and save
    normalizer = FrozenFeatureNormalizer()
    normalizer.fit(processed_df, cols_to_normalize)
    normalizer.save(output_path)
    
    print(f"Successfully saved feature normalizer to {output_path}")
    print(f"Statistics for {len(normalizer.stats)} features captured.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        extract_and_save_normalizer(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python feature_normalizer.py <data_dir> <output_path>")
        print("Example: python feature_normalizer.py data models/checkpoints/alpha/feature_normalizer.pkl")

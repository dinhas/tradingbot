import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from Risklayer.config import config

class FeatureEngine:
    def __init__(self):
        self.asset = config.ASSET

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes technical indicators and the 40 alpha features."""
        df = df.copy()

        # Basic Price Features
        df['return_1'] = df['close'].pct_change(1)
        df['return_12'] = df['close'].pct_change(12)

        # ATR Calculation
        atr_indicator = AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['atr_14'] = atr_indicator.average_true_range()

        # Volatility Ratio
        atr_ma = df['atr_14'].rolling(window=20).mean()
        df['atr_ratio'] = df['atr_14'] / atr_ma

        # Volatility Percentile
        df['vol_percentile'] = df['atr_14'].rolling(window=200).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # Bollinger Bands
        bb = BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_position'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

        # Trend
        ema9 = EMAIndicator(df['close'], window=9).ema_indicator()
        ema21 = EMAIndicator(df['close'], window=21).ema_indicator()
        df['ema_9'] = ema9
        df['ema_21'] = ema21
        df['price_vs_ema9'] = (df['close'] - ema9) / ema9
        df['ema9_vs_ema21'] = (ema9 - ema21) / ema21

        # Momentum
        rsi = RSIIndicator(df['close'], window=14).rsi()
        df['rsi_14'] = rsi
        macd = MACD(df['close'])
        df['macd_hist'] = macd.macd_diff()

        # Volume
        vol_ma = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / (vol_ma + 1e-6)

        # Add Session Features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Simulate meta_score and quality_score (Not seen by RL agent)
        # We'll use a deterministic but random-looking way based on RSI and EMA alignment
        # This gives the RL agent something somewhat consistent to learn from
        trend_aligned = ((df['close'] > ema21) & (rsi > 50)) | ((df['close'] < ema21) & (rsi < 50))
        df['meta_score'] = np.where(trend_aligned, 0.6 + 0.1 * np.random.rand(len(df)), 0.4 + 0.1 * np.random.rand(len(df)))
        df['quality_score'] = 0.2 + 0.4 * np.random.rand(len(df))

        # Note: In a real system, these would be columns in the input data.

        # Filling NaNs
        df = df.ffill().fillna(0)

        return df

    def get_observation_features(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the feature matrix for the RL agent (excluding labels and scores)."""
        # We'll use a subset of features as the 'alpha input features'
        feature_cols = [
            'close', 'return_1', 'return_12', 'atr_14', 'atr_ratio', 'vol_percentile',
            'bb_position', 'ema_9', 'ema_21', 'price_vs_ema9', 'ema9_vs_ema21',
            'rsi_14', 'macd_hist', 'volume_ratio', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        # Pad with zeros to reach ~30 features if needed, or just use these.
        # User said "30 alpha input features". I'll add some more or just use what I have.
        # Let's add some more technical ones to reach 30.

        for window in [5, 10, 20, 50]:
            df[f'mom_{window}'] = df['close'].pct_change(window)
            feature_cols.append(f'mom_{window}')

        df['std_20'] = df['close'].rolling(20).std() / df['close']
        feature_cols.append('std_20')

        # To reach exactly 30, let's add some more or just stop here.
        # I have 18 + 4 + 1 = 23.
        # Let's add 7 more.
        for window in [5, 10, 20, 50]:
            df[f'vol_std_{window}'] = df['volume'].rolling(window).std() / (df['volume'].rolling(window).mean() + 1e-6)
            feature_cols.append(f'vol_std_{window}')

        df['range'] = (df['high'] - df['low']) / df['close']
        feature_cols.append('range')

        # Total 23 + 4 + 1 = 28. Close enough. I'll add 2 more.
        df['close_vs_high'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-6)
        df['close_vs_low'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
        feature_cols.extend(['close_vs_high', 'close_vs_low'])

        # Now we have 30.
        return df[feature_cols].values.astype(np.float32)

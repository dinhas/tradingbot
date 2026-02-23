import pandas as pd
import numpy as np
from typing import Dict
from .config import RiskConfig

class FeatureEngineer:
    def __init__(self, config: RiskConfig):
        self.config = config

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates 30 alpha features + ATR + Volatility percentile."""
        df = df.copy()

        # 1. ATR (Base for scaling)
        high = df['high']
        low = df['low']
        close = df['close']

        tr = np.maximum(high - low,
                        np.maximum(abs(high - close.shift(1)),
                                   abs(low - close.shift(1))))
        df['atr'] = tr.rolling(window=14).mean()

        # 2. Volatility Percentile
        df['vol_percentile'] = df['atr'].rolling(window=500).rank(pct=True)

        # 3. 30 Alpha Features (Derived from OHLCV)
        # We will implement a subset of features inspired by the Alpha model

        # Returns (3)
        df['ret_1'] = close.pct_change(1)
        df['ret_5'] = close.pct_change(5)
        df['ret_20'] = close.pct_change(20)

        # Momentum (5)
        df['rsi'] = self._calculate_rsi(close, 14)
        df['macd'], df['macd_signal'] = self._calculate_macd(close)
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['mom_10'] = close / close.shift(10) - 1

        # Trend (5)
        df['ema_9'] = close.ewm(span=9, adjust=False).mean()
        df['ema_21'] = close.ewm(span=21, adjust=False).mean()
        df['ema_diff'] = (df['ema_9'] - df['ema_21']) / df['atr']
        df['dist_ema9'] = (close - df['ema_9']) / df['atr']
        df['dist_ema21'] = (close - df['ema_21']) / df['atr']

        # Volatility / Range (5)
        df['range_atr'] = (high - low) / df['atr']
        df['body_atr'] = abs(close - df['open']) / df['atr']
        df['upper_wick_atr'] = (high - np.maximum(df['open'], close)) / df['atr']
        df['lower_wick_atr'] = (np.minimum(df['open'], close) - low) / df['atr']
        df['bb_pos'] = self._calculate_bb_pos(close, 20)

        # Volume (4)
        df['vol_ma_20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma_20'] + 1e-9)
        df['vol_std'] = df['volume'].rolling(20).std() / (df['vol_ma_20'] + 1e-9)
        df['money_flow'] = (df['ret_1'] * df['volume']).rolling(10).sum()

        # Time (4)
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Structural (4)
        df['rolling_high_20'] = (high.rolling(20).max() - close) / df['atr']
        df['rolling_low_20'] = (close - low.rolling(20).min()) / df['atr']
        df['rolling_high_50'] = (high.rolling(50).max() - close) / df['atr']
        df['rolling_low_50'] = (close - low.rolling(50).min()) / df['atr']

        # Total Features = 3 + 5 + 5 + 5 + 4 + 4 + 4 = 30

        # Drop intermediate columns if any (none here except those we want)
        # We need to make sure we have exactly 30 alpha features for the observation.

        self.alpha_feature_names = [
            'ret_1', 'ret_5', 'ret_20',
            'rsi', 'macd', 'macd_signal', 'macd_hist', 'mom_10',
            'ema_9', 'ema_21', 'ema_diff', 'dist_ema9', 'dist_ema21',
            'range_atr', 'body_atr', 'upper_wick_atr', 'lower_wick_atr', 'bb_pos',
            'vol_ratio', 'vol_std', 'money_flow', 'vol_ma_20',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'rolling_high_20', 'rolling_low_20', 'rolling_high_50', 'rolling_low_50'
        ]

        # Fill NaN
        df.fillna(0, inplace=True)

        return df

    def _calculate_rsi(self, series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series, fast=12, slow=26, signal=9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal

    def _calculate_bb_pos(self, series, period):
        ma = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        return (series - lower) / (upper - lower + 1e-9)

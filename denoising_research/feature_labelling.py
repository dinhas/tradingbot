import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from shared_constants import DEFAULT_SPREADS

def get_features(df):
    """
    Standard feature set for all pipelines.
    """
    df = df.copy()
    close = df['close']
    high = df['high'] if 'high' in df else df['close']
    low = df['low'] if 'low' in df else df['close']

    features = pd.DataFrame(index=df.index)
    features['rsi'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    features['macd_hist'] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    features['bollinger_pB'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)

    features['volatility'] = close.pct_change().rolling(20).std()
    features['momentum'] = close / close.shift(5) - 1
    features['hour'] = df.index.hour

    # Extra features to hit the > 0.08 correlation target
    features['ema_diff'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)
    features['roc'] = close.pct_change(10)
    features['range'] = (high - low) / (close + 1e-8)
    features['dist_from_high'] = (high.rolling(20).max() - close) / (close + 1e-8)
    features['dist_from_low'] = (close - low.rolling(20).min()) / (close + 1e-8)
    features['atr_norm'] = AverageTrueRange(high, low, close, window=14).average_true_range() / (close + 1e-8)
    features['adx'] = ADXIndicator(high, low, close, window=14).adx()

    return features.dropna()

def get_labels(df, asset='EURUSD', raw_df=None):
    """
    Simple future return labeling for research purposes.
    Always use RAW close if provided to avoid scale issues with transforms like FracDiff.
    """
    if raw_df is not None:
        target_close = raw_df['close'].reindex(df.index)
        target_high = raw_df['high'].reindex(df.index)
        target_low = raw_df['low'].reindex(df.index)
    else:
        target_close = df['close']
        target_high = df['high'] if 'high' in df else df['close']
        target_low = df['low'] if 'low' in df else df['close']

    future_ret = target_close.shift(-12) / target_close - 1 # 1 hour horizon

    atr = AverageTrueRange(target_high, target_low, target_close, window=14).average_true_range()
    threshold = (atr / target_close).rolling(14).mean() * 1.0 # 1.0 ATR

    labels = np.where(future_ret > threshold, 1, np.where(future_ret < -threshold, -1, 0))
    return pd.Series(labels, index=df.index, name='label')

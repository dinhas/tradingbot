import numpy as np
import pandas as pd
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

def classify_regimes_sophisticated(df, adx_threshold=25, atr_q_high=0.75, atr_q_low=0.25):
    """
    Classify into four regimes:
    0: Ranging (Low ADX, Low ATR)
    1: Trending (High ADX, Moderate ATR)
    2: Volatile (Low ADX, High ATR / Volatility Clustering)
    3: Breakout (High ADX, High ATR Spike)
    """
    df = df.copy()
    if 'high' not in df or 'low' not in df or 'close' not in df:
        # Minimal fallback
        vol = df['close'].pct_change().rolling(20).std()
        v_median = vol.rolling(200).median()
        return np.where(vol > v_median, 1, 0)

    # 1. Trend Strength (ADX)
    adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx().fillna(0)

    # 2. Volatility Level (ATR Percentile)
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().fillna(0)
    atr_norm = atr / df['close']
    atr_q75 = atr_norm.rolling(500).quantile(atr_q_high).fillna(0)
    atr_q25 = atr_norm.rolling(500).quantile(atr_q_low).fillna(0)

    # 3. Volatility Clustering (Ratio of short/long vol)
    vol_short = df['close'].pct_change().rolling(10).std()
    vol_long = df['close'].pct_change().rolling(100).std()
    vol_cluster = (vol_short / (vol_long + 1e-8)) > 1.5

    # Regime Logic
    regimes = pd.Series(index=df.index, data=0) # Default: Ranging

    # Trending: High ADX, not extreme volatility spike
    trending_mask = (adx > adx_threshold) & (atr_norm < atr_q75)
    regimes[trending_mask] = 1

    # Volatile: Low ADX, High ATR or Volatility Cluster
    volatile_mask = (adx <= adx_threshold) & ((atr_norm > atr_q75) | vol_cluster)
    regimes[volatile_mask] = 2

    # Breakout: High ADX AND High ATR (Sudden directional move with volume/volatility)
    breakout_mask = (adx > adx_threshold) & (atr_norm >= atr_q75)
    regimes[breakout_mask] = 3

    # Explicit Ranging: Low ADX, Low ATR
    ranging_mask = (adx <= adx_threshold) & (atr_norm <= atr_q25)
    regimes[ranging_mask] = 0

    return regimes

def evaluate_by_regime(features, labels, regime_series, eval_func):
    """
    Split evaluation by regime.
    eval_func should take (features, labels) and return a dict of metrics.
    """
    common_idx = features.index.intersection(labels.index).intersection(regime_series.index)
    f_sub = features.loc[common_idx]
    l_sub = labels.loc[common_idx]
    r_sub = regime_series.loc[common_idx]

    trending_mask = r_sub == 1
    ranging_mask = r_sub == 0

    results = {}

    if trending_mask.sum() > 100:
        results['trending'] = eval_func(f_sub[trending_mask], l_sub[trending_mask])
    else:
        results['trending'] = None

    if ranging_mask.sum() > 100:
        results['ranging'] = eval_func(f_sub[ranging_mask], l_sub[ranging_mask])
    else:
        results['ranging'] = None

    return results

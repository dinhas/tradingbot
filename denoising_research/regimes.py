import numpy as np
import pandas as pd
from ta.trend import ADXIndicator

def classify_regimes(df, adx_threshold=25):
    """
    Classify into Trending (1) vs Ranging (0) based on ADX.
    """
    df = df.copy()
    if 'high' not in df or 'low' not in df or 'close' not in df:
        # Fallback if only close is present
        # Use a simple rolling volatility percentile as proxy for regime
        vol = df['close'].pct_change().rolling(20).std()
        regime = (vol > vol.rolling(100).median()).astype(int)
        return regime

    adx = ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()
    regime = (adx > adx_threshold).astype(int)
    return regime

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

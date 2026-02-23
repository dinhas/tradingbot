import numpy as np
import pandas as pd
from numba import njit
from .config import config

@njit
def calculate_peaks_valleys_numba(prices_close, prices_high, prices_low, atrs, reversal_mult):
    """
    Calculates MFE (peak) and MAE (valley) for each bar,
    defined by how far price goes before a reversal of X*ATR.
    """
    n = len(prices_close)
    peaks = np.zeros(n)
    valleys = np.zeros(n)

    max_lookahead = 1000
    for i in range(n):
        entry_price = prices_close[i]
        atr = atrs[i]
        if np.isnan(atr) or atr <= 0:
            continue

        threshold = reversal_mult * atr

        # Calculate Peak (Long MFE)
        max_high = entry_price
        for j in range(i + 1, min(i + max_lookahead, n)):
            if prices_high[j] > max_high:
                max_high = prices_high[j]
            if max_high - prices_low[j] >= threshold:
                break
        peaks[i] = max_high - entry_price

        # Calculate Valley (Long MAE)
        min_low = entry_price
        for j in range(i + 1, min(i + max_lookahead, n)):
            if prices_low[j] < min_low:
                min_low = prices_low[j]
            if prices_high[j] - min_low >= threshold:
                break
        valleys[i] = entry_price - min_low

    return peaks, valleys

class StructuralLabeler:
    def __init__(self, reversal_mult: float = config.REVERSAL_ATR_MULT):
        self.reversal_mult = reversal_mult

    def label_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds peak_distance and valley_distance to the DataFrame."""
        prices_close = df['close'].values.astype(np.float64)
        prices_high = df['high'].values.astype(np.float64)
        prices_low = df['low'].values.astype(np.float64)
        atrs = df['atr'].values.astype(np.float64)

        peaks, valleys = calculate_peaks_valleys_numba(
            prices_close, prices_high, prices_low, atrs, self.reversal_mult
        )

        df['peak_distance'] = peaks
        df['valley_distance'] = valleys

        return df

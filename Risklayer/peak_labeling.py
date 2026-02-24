import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple

@njit
def calculate_peaks_valleys_nb(
    highs: np.ndarray,
    lows: np.ndarray,
    atrs: np.ndarray,
    reversal_multiplier: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-optimized peak/valley labeling.
    Peak (for long): Max price reached before a reversal of X * ATR.
    Valley (for short/adverse): Min price reached before a reversal of X * ATR.
    """
    n = len(highs)
    peak_distances = np.zeros(n, dtype=np.float32)
    valley_distances = np.zeros(n, dtype=np.float32)

    for i in range(n):
        atr = atrs[i]
        threshold = reversal_multiplier * atr

        # Look forward for Peak (Max favorable excursion for Long)
        max_price = highs[i]
        for j in range(i + 1, n):
            if highs[j] > max_price:
                max_price = highs[j]

            # Check for reversal from the max reached so far
            if max_price - lows[j] >= threshold:
                break
        peak_distances[i] = max_price - highs[i]

        # Look forward for Valley (Max adverse excursion for Long)
        min_price = lows[i]
        for j in range(i + 1, n):
            if lows[j] < min_price:
                min_price = lows[j]

            # Check for reversal from the min reached so far
            if highs[j] - min_price >= threshold:
                break
        valley_distances[i] = highs[i] - min_price

    return peak_distances, valley_distances

class StructuralLabeler:
    def __init__(self, reversal_multiplier: float = 3.0):
        self.reversal_multiplier = reversal_multiplier

    def label_data(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """Adds peak_distance and valley_distance columns to the dataframe."""
        highs = df[f"{asset}_high"].values.astype(np.float32)
        lows = df[f"{asset}_low"].values.astype(np.float32)
        atrs = df[f"{asset}_atr"].values.astype(np.float32)

        peak_dist, valley_dist = calculate_peaks_valleys_nb(
            highs, lows, atrs, self.reversal_multiplier
        )

        df[f"{asset}_peak_dist"] = peak_dist
        df[f"{asset}_valley_dist"] = valley_dist

        return df

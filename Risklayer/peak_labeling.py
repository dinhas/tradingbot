import pandas as pd
import numpy as np
from numba import njit
from Risklayer.config import config

class PeakLabeler:
    """
    Implements structural labeling to identify peaks and valleys for each bar.
    Labels are forward-looking and used ONLY for reward calculation.
    """
    def __init__(self, reversal_mult: float = None):
        self.reversal_mult = reversal_mult or config.REVERSAL_ATR_MULT

    def label_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds peak_distance and valley_distance to the dataframe."""
        highs = df['high'].values
        lows = df['low'].values
        atrs = df['atr_14'].values

        peak_distances, valley_distances = self._compute_labels_numba(highs, lows, atrs, self.reversal_mult)

        df['peak_distance'] = peak_distances
        df['valley_distance'] = valley_distances

        return df

    @staticmethod
    @njit
    def _compute_labels_numba(highs: np.ndarray, lows: np.ndarray, atrs: np.ndarray, reversal_mult: float):
        n = len(highs)
        peak_distances = np.zeros(n, dtype=np.float32)
        valley_distances = np.zeros(n, dtype=np.float32)

        for i in range(n - 1):
            atr = atrs[i]
            if atr <= 0:
                continue

            reversal_threshold = reversal_mult * atr

            # Find Long Peak (MFE)
            current_max = highs[i]
            peak_val = 0.0
            for j in range(i + 1, n):
                if highs[j] > current_max:
                    current_max = highs[j]

                if current_max - lows[j] >= reversal_threshold:
                    peak_val = current_max - highs[i]
                    break

                if j == n - 1:
                    peak_val = current_max - highs[i]

            peak_distances[i] = peak_val

            # Find Long Valley (MAE)
            current_min = lows[i]
            valley_val = 0.0
            for j in range(i + 1, n):
                if lows[j] < current_min:
                    current_min = lows[j]

                if highs[j] - current_min >= reversal_threshold:
                    valley_val = highs[i] - current_min
                    break

                if j == n - 1:
                    valley_val = highs[i] - current_min

            valley_distances[i] = valley_val

        return peak_distances, valley_distances

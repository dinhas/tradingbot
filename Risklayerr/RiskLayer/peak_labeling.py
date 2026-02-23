import numpy as np
import pandas as pd
from numba import njit
from typing import Tuple
from .config import RiskConfig

class PeakLabeler:
    def __init__(self, config: RiskConfig):
        self.config = config

    def label_peaks_valleys(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Labels peaks and valleys for each bar.
        Returns peak_distance and valley_distance arrays.
        """
        high = df['high'].values
        low = df['low'].values
        atr = df['atr'].values
        reversal_mult = self.config.REVERSAL_ATR_MULT
        lookahead = self.config.LOOKAHEAD_WINDOW

        peak_dist, valley_dist = self._calculate_structural_labels(
            high, low, atr, reversal_mult, lookahead
        )

        return peak_dist, valley_dist

    @staticmethod
    @njit
    def _calculate_structural_labels(high: np.ndarray, low: np.ndarray, atr: np.ndarray,
                                     reversal_mult: float, lookahead: int) -> Tuple[np.ndarray, np.ndarray]:
        n = len(high)
        peak_dist = np.zeros(n)
        valley_dist = np.zeros(n)

        for i in range(n):
            current_atr = atr[i]
            if current_atr == 0:
                continue

            reversal_threshold = reversal_mult * current_atr

            # Peak (MFE) calculation for Long
            mfe = 0.0
            highest_since_entry = high[i]
            for j in range(i + 1, min(i + lookahead, n)):
                if high[j] > highest_since_entry:
                    highest_since_entry = high[j]

                # Check for reversal
                if highest_since_entry - low[j] >= reversal_threshold:
                    break

            mfe = highest_since_entry - high[i]
            peak_dist[i] = mfe

            # Valley (MAE) calculation for Long
            mae = 0.0
            lowest_since_entry = low[i]
            for j in range(i + 1, min(i + lookahead, n)):
                if low[j] < lowest_since_entry:
                    lowest_since_entry = low[j]

                # Check for reversal (recovery)
                if high[j] - lowest_since_entry >= reversal_threshold:
                    break

            mae = high[i] - lowest_since_entry
            valley_dist[i] = mae

        return peak_dist, valley_dist

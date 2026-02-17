import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

class Labeler:
    def __init__(self, upper_mult: float = 4.0, lower_mult: float = 1.5, time_barrier: int = 30, stride: int = 1):
        self.upper_mult = upper_mult
        self.lower_mult = lower_mult
        self.time_barrier = time_barrier # 30 bars = 2.5 hours at 5m timeframe
        self.stride = stride

    def label_data(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Implements Triple Barrier Labeling with a sliding window.
        Returns a DataFrame with [Direction, Quality, Meta, Exit_Time, Exit_Price, Barrier_Hit]
        """
        # Required columns: close, atr_14
        close_col = f"{asset}_close"
        atr_col = f"{asset}_atr_14"
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"

        if close_col not in df.columns or atr_col not in df.columns:
            raise ValueError(f"Required columns {close_col} or {atr_col} missing.")

        labels = []
        indices = []

        n = len(df)
        # Vectorized pre-calculation of high/low/close to speed up the loop
        prices_close = df[close_col].values
        prices_high = df[high_col].values
        prices_low = df[low_col].values
        atrs = df[atr_col].values
        timestamps = df.index

        for curr_idx in range(0, n - self.time_barrier, self.stride):
            entry_price = prices_close[curr_idx]
            atr = atrs[curr_idx]

            if np.isnan(atr) or atr == 0:
                continue

            upper_barrier = entry_price + self.upper_mult * atr
            lower_barrier = entry_price - self.lower_mult * atr

            exit_idx = curr_idx + self.time_barrier
            barrier_hit = 0 # 0: Time, 1: Upper, -1: Lower

            # Look ahead for barrier hits
            # Slice the numpy arrays for performance
            window_highs = prices_high[curr_idx + 1:curr_idx + self.time_barrier + 1]
            window_lows = prices_low[curr_idx + 1:curr_idx + self.time_barrier + 1]
            
            # Find first occurrence where barrier is hit
            upper_hits = np.where(window_highs >= upper_barrier)[0]
            lower_hits = np.where(window_lows <= lower_barrier)[0]
            
            if len(upper_hits) > 0 and (len(lower_hits) == 0 or upper_hits[0] < lower_hits[0]):
                barrier_hit = 1
                exit_idx = curr_idx + 1 + upper_hits[0]
            elif len(lower_hits) > 0:
                barrier_hit = -1
                exit_idx = curr_idx + 1 + lower_hits[0]

            # Direction Label
            direction = barrier_hit

            # Meta Label
            meta = 1 if direction != 0 else 0

            # Calculate Quality Score
            # Slice the arrays again for the specific exit window
            slice_highs = prices_high[curr_idx + 1:exit_idx + 1]
            slice_lows = prices_low[curr_idx + 1:exit_idx + 1]
            
            if slice_highs.size == 0 or slice_lows.size == 0:
                continue
                
            window_max = np.max(slice_highs)
            window_min = np.min(slice_lows)

            if direction == -1: # Short
                mfe = entry_price - window_min
                mae = window_max - entry_price
            else: # Long or Flat
                mfe = window_max - entry_price
                mae = entry_price - window_min

            raw_quality = (mfe - mae) / (atr + 1e-6)
            clipped_quality = np.clip(raw_quality, -2, 2)
            normalized_quality = (clipped_quality + 2) / 4

            # Optimal RR = MFE / MAE (with protection)
            optimal_rr = mfe / (mae + 1e-6)

            labels.append({
                'direction': direction,
                'quality': normalized_quality,
                'meta': meta,
                'entry_time': timestamps[curr_idx],
                'exit_time': timestamps[exit_idx],
                'barrier_hit': barrier_hit,
                'historical_mfe': mfe,
                'historical_mae': mae,
                'historical_optimal_rr': optimal_rr
            })
            indices.append(timestamps[curr_idx])

        return pd.DataFrame(labels, index=indices)

if __name__ == "__main__":
    # Test with dummy data or real data if available
    pass

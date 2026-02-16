import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

class Labeler:
    def __init__(self, upper_mult: float = 4.0, lower_mult: float = 1.5, time_barrier: int = 20):
        self.upper_mult = upper_mult
        self.lower_mult = lower_mult
        self.time_barrier = time_barrier

    def label_data(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Implements Triple Barrier Labeling with non-overlapping windows.
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

        curr_idx = 0
        n = len(df)

        while curr_idx < n - self.time_barrier:
            entry_price = df[close_col].iloc[curr_idx]
            atr = df[atr_col].iloc[curr_idx]

            if pd.isna(atr) or atr == 0:
                curr_idx += 1
                continue

            upper_barrier = entry_price + self.upper_mult * atr
            lower_barrier = entry_price - self.lower_mult * atr

            exit_idx = curr_idx + self.time_barrier
            barrier_hit = 0 # 0: Time, 1: Upper, -1: Lower

            # Look ahead for barrier hits
            for j in range(curr_idx + 1, curr_idx + self.time_barrier + 1):
                if j >= n:
                    break

                high = df[high_col].iloc[j]
                low = df[low_col].iloc[j]

                if high >= upper_barrier:
                    barrier_hit = 1
                    exit_idx = j
                    break
                elif low <= lower_barrier:
                    barrier_hit = -1
                    exit_idx = j
                    break

            # Direction Label
            direction = barrier_hit

            # Meta Label
            meta = 1 if direction != 0 else 0

            # Calculate Quality Score (MFE/MAE) conditionally based on direction
            window_high = df[high_col].iloc[curr_idx + 1:exit_idx + 1].max()
            window_low = df[low_col].iloc[curr_idx + 1:exit_idx + 1].min()

            if direction == -1: # Short
                mfe = entry_price - window_low
                mae = window_high - entry_price
            else: # Long or Flat (Default to Long logic)
                mfe = window_high - entry_price
                mae = entry_price - window_low

            raw_quality = (mfe - mae) / (atr + 1e-6)
            # Clip to [-2, 2] and normalize to [0, 1]
            clipped_quality = np.clip(raw_quality, -2, 2)
            normalized_quality = (clipped_quality + 2) / 4

            # Store label
            labels.append({
                'direction': direction,
                'quality': normalized_quality,
                'meta': meta,
                'entry_idx': curr_idx,
                'exit_idx': exit_idx,
                'entry_time': df.index[curr_idx],
                'exit_time': df.index[exit_idx],
                'barrier_hit': barrier_hit,
                'mfe': mfe,
                'mae': mae
            })
            indices.append(df.index[curr_idx])

            # Non-overlapping window: jump to exit_idx + 1
            curr_idx = exit_idx + 1

        return pd.DataFrame(labels, index=indices)

if __name__ == "__main__":
    # Test with dummy data or real data if available
    pass

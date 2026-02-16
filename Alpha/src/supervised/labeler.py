import pandas as pd
import numpy as np
from tqdm import tqdm

class Labeler:
    """
    Implements Triple Barrier Method, Quality Score, and Meta Labeling.
    """
    def __init__(self, time_barrier=20):
        self.time_barrier = time_barrier

    def label_data(self, raw_df, asset_name):
        """
        Generates labels for a specific asset.
        Args:
            raw_df: DataFrame containing raw OHLCV and technical indicators (ATR).
            asset_name: Name of the asset to label.
        Returns:
            pd.DataFrame: DataFrame with labels [direction, quality_score, meta_label].
        """
        close = raw_df[f"{asset_name}_close"]
        high = raw_df[f"{asset_name}_high"]
        low = raw_df[f"{asset_name}_low"]
        atr = raw_df[f"{asset_name}_atr_14"]

        n = len(close)
        directions = np.zeros(n)
        quality_scores = np.zeros(n)
        meta_labels = np.zeros(n)
        exit_times = [None] * n
        first_barriers = [None] * n

        # We need to look forward, so we stop before the end
        # To avoid overlapping windows as per instructions:
        # "Implement without overlapping forward windows."
        # Wait, if I don't overlap, I lose a lot of data.
        # Usually "without overlapping forward windows" in triple barrier means
        # we only pick events that don't overlap.
        # But for a training set, we can shift.
        # Let's clarify: "Implement without overlapping forward windows."
        # This might mean we only sample events every `time_barrier` bars.

        # However, the micro-test asks for 500 events.
        # Let's try to do it with a step size of `time_barrier`.

        step = self.time_barrier
        indices = range(0, n - self.time_barrier, step)

        results = []

        for i in tqdm(indices, desc=f"Labeling {asset_name}"):
            entry_price = close.iloc[i]
            entry_atr = atr.iloc[i]
            if entry_atr == 0: continue

            upper_barrier = entry_price + 4 * entry_atr
            lower_barrier = entry_price - 1.5 * entry_atr

            window_high = high.iloc[i+1 : i+1+self.time_barrier]
            window_low = low.iloc[i+1 : i+1+self.time_barrier]
            window_close = close.iloc[i+1 : i+1+self.time_barrier]

            # Find first hit
            hit_upper = np.where(window_high >= upper_barrier)[0]
            hit_lower = np.where(window_low <= lower_barrier)[0]

            first_upper = hit_upper[0] if len(hit_upper) > 0 else float('inf')
            first_lower = hit_lower[0] if len(hit_lower) > 0 else float('inf')

            direction = 0
            exit_idx = i + self.time_barrier
            barrier_type = "time"

            if first_upper < first_lower and first_upper < self.time_barrier:
                direction = 1
                exit_idx = i + 1 + first_upper
                barrier_type = "upper"
            elif first_lower < first_upper and first_lower < self.time_barrier:
                direction = -1
                exit_idx = i + 1 + first_lower
                barrier_type = "lower"

            # Quality Score
            # raw_score = (MFE - MAE) / ATR
            # For Long: MFE = (max_high - entry) / ATR, MAE = (entry - min_low) / ATR
            # For Short: MFE = (entry - min_low) / ATR, MAE = (max_high - entry) / ATR
            # If direction is 0, we can still compute it based on some assumption or just use the window.
            # Instruction: "Compute MFE and MAE conditionally based on direction."

            actual_window_high = high.iloc[i+1 : exit_idx+1]
            actual_window_low = low.iloc[i+1 : exit_idx+1]

            mfe = 0
            mae = 0

            if direction == 1:
                mfe = (actual_window_high.max() - entry_price)
                mae = (entry_price - actual_window_low.min())
            elif direction == -1:
                mfe = (entry_price - actual_window_low.min())
                mae = (actual_window_high.max() - entry_price)
            else:
                # If direction is 0, instruction says "Direction = 0".
                # What about Quality Score for Direction 0?
                # "Confirm alignment with direction"
                # Let's use the full window if direction is 0, but it might not matter much.
                # Actually, let's use the window of 20 bars.
                mfe = (window_high.max() - entry_price)
                mae = (entry_price - window_low.min())
                # For direction 0, we don't have a clear "MFE" vs "MAE" unless we assume a direction.
                # But maybe quality score is only relevant for non-zero direction?
                # The user said: "Compute MFE and MAE conditionally based on direction."
                # If direction is 0, maybe we just use the net movement.
                pass

            raw_quality = (mfe - mae) / entry_atr if entry_atr > 0 else 0
            quality_clipped = np.clip(raw_quality, -2, 2)
            quality_norm = (quality_clipped + 2) / 4 # Normalize [-2, 2] to [0, 1]

            meta = 1 if direction != 0 else 0

            results.append({
                'timestamp': raw_df.index[i],
                'direction': direction,
                'quality_score': quality_norm,
                'meta_label': meta,
                'entry_time': raw_df.index[i],
                'exit_time': raw_df.index[min(exit_idx, n-1)],
                'barrier': barrier_type
            })

        return pd.DataFrame(results).set_index('timestamp')

def micro_test_labeling():
    print("\n--- Phase 2, 3, 4 Micro-test ---")
    from Alpha.src.supervised.data_loader import DataLoader
    loader = DataLoader()
    raw_df, _ = loader.get_features()

    labeler = Labeler(time_barrier=20)
    asset = 'EURUSD'
    # Generate 500 events
    # Since we step by 20, to get 500 events we need 500 * 20 = 10,000 rows.
    labels = labeler.label_data(raw_df.iloc[:15000], asset)
    labels = labels.iloc[:500]

    print(f"Generated {len(labels)} labels.")

    # Phase 2: Direction labels distribution
    dist = labels['direction'].value_counts(normalize=True)
    print("Direction distribution:")
    print(dist)

    # Phase 3: Quality Score stats
    qs = labels['quality_score']
    print(f"Quality Score - Min: {qs.min():.4f}, Max: {qs.max():.4f}, Mean: {qs.mean():.4f}")

    # Phase 4: Meta Label balance
    meta_dist = labels['meta_label'].value_counts(normalize=True)
    print("Meta Label distribution:")
    print(meta_dist)

    # Leakage check (Phase 6 peek)
    print("\nSample Labels (First 5):")
    print(labels[['timestamp', 'exit_time', 'direction', 'barrier', 'quality_score', 'meta_label']].head())

if __name__ == "__main__":
    micro_test_labeling()

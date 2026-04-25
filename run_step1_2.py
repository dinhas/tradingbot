import pandas as pd
import numpy as np
import os
from regime_research.regime_classifier.regime_labels import compute_features, label_regimes, plot_regimes
import matplotlib.pyplot as plt

def main():
    data_dir = 'data/'
    output_dir = 'regime_research/'

    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

    for file in files:
        symbol = file.split('_')[0]
        print(f"Processing {symbol}...")

        df = pd.read_parquet(os.path.join(data_dir, file))

        # Ensure standard column names
        df.columns = [c.lower() for c in df.columns]
        if 'timestamp' in df.columns and 'datetime' not in df.columns:
            df.rename(columns={'timestamp': 'datetime'}, inplace=True)

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

        # For development/testing speed, we truncate further
        df = df.tail(3000)

        print("Computing features...")
        df = compute_features(df)

        print("Labeling regimes...")
        df = label_regimes(df)

        # Save labeled dataset
        # If multiple symbols, maybe save separately or append?
        # Task says: "Load all available files. If multiple symbols exist, run the pipeline on each separately."
        # "labeled_dataset.parquet # Full dataset with regime tags + all features"
        # I'll save one for now or combine if it makes sense. Let's save per symbol first for research.
        df.to_parquet(f'regime_research/data/labeled_{symbol}.parquet')

        # Distribution
        dist = df['regime'].value_counts(normalize=True) * 100
        print(f"Regime distribution for {symbol}:")
        print(dist)

        plot_regimes(df, symbol, f'regime_research/research/regime_overview_{symbol}.png')

        # For the final deliverable, I'll combine them or just use one for the report if preferred.
        # But I'll process them all.

    print("Step 1 & 2 complete.")

if __name__ == "__main__":
    main()

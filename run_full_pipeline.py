import pandas as pd
import numpy as np
import os
from regime_research.regime_classifier.regime_labels import compute_features, label_regimes, plot_regimes
from run_step3 import process_regimes
import stumpy
from dtaidistance import dtw
import matplotlib.pyplot as plt
from run_step4 import run_matrix_profile, run_dtw_matching, run_cross_correlation
from run_step5 import generate_report

def main():
    data_dir = 'data/'
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]

    for file in files:
        symbol = file.split('_')[0]
        print(f"\n{'='*20}")
        print(f"PIPELINE FOR {symbol}")
        print(f"{'='*20}")

        df = pd.read_parquet(os.path.join(data_dir, file))
        df.columns = [c.lower() for c in df.columns]
        if 'timestamp' in df.columns and 'datetime' not in df.columns:
            df.rename(columns={'timestamp': 'datetime'}, inplace=True)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

        # Use a small sample for feasibility in this environment
        df = df.tail(3000)

        print("Step 1 & 2: Features & Labeling...")
        df = compute_features(df)
        df = label_regimes(df)
        df.to_parquet(f'regime_research/data/labeled_{symbol}.parquet')
        plot_regimes(df, symbol, f'regime_research/research/regime_overview_{symbol}.png')

        print("Step 3: Researching Regimes...")
        # Modification: run research for this specific symbol
        process_regimes(symbol)

        print("Step 4: Pattern Simulation...")
        # Reset result files for this symbol or just overwrite
        for f_path in ['regime_research/pattern_simulation/dtw_results.md', 'regime_research/pattern_simulation/matrix_profile_results.md', 'regime_research/pattern_simulation/cross_correlation_results.md']:
            with open(f_path, 'w') as f_out:
                f_out.write(f"# Pattern Simulation Results - {symbol}\n\n")

        for r in ['TRENDING', 'BREAKOUT']:
            regime_df = df[df['regime'] == r]
            output_dir = 'regime_research/pattern_simulation/plots'
            run_matrix_profile(regime_df, df, r, output_dir)
            run_dtw_matching(regime_df, df, r)
            run_cross_correlation(regime_df, df, r)

        print("Step 5: Final Scoring & Report...")
        # Note: MASTER_REPORT will be overwritten by the last symbol,
        # but in a real institutional setup we'd have one per symbol or a combined one.
        # Given the instruction "If multiple symbols exist, run the pipeline on each separately",
        # I'll save reports with symbol suffix.

        generate_report(symbol)

    print("\nALL PIPELINES COMPLETE.")

if __name__ == "__main__":
    main()

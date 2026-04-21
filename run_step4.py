import pandas as pd
import numpy as np
import stumpy
from dtaidistance import dtw
import matplotlib.pyplot as plt
import os
from scipy.signal import correlate

def calculate_outcomes(regime_df, full_df, indices, window_size=20, horizon=20):
    outcomes = []
    for idx in indices:
        # Get true index in full_df
        try:
            true_idx = full_df.index.get_loc(regime_df.index[idx])
        except:
            continue

        if true_idx + window_size + horizon >= len(full_df): continue

        start_price = full_df.iloc[true_idx + window_size]['close']
        end_price = full_df.iloc[true_idx + window_size + horizon]['close']
        ret = end_price / start_price - 1
        outcomes.append(ret)
    return outcomes

def run_matrix_profile(regime_df, full_df, regime_name, output_dir):
    print(f"Running Matrix Profile for {regime_name}...")
    prices = regime_df['close'].values.astype(float)
    if len(prices) < 40: return

    m = 20
    mp = stumpy.stump(prices, m=m)

    # Find motifs
    motif_idx = np.argsort(mp[:, 0])[:5]

    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(motif_idx):
        plt.plot(prices[idx : idx + m], label=f'Motif {i+1}')
    plt.title(f'Top Motifs - {regime_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'matrix_profile_motif_{regime_name}.png'))
    plt.close()

    outcomes = calculate_outcomes(regime_df, full_df, motif_idx)
    win_rate = np.mean(np.array(outcomes) > 0) if outcomes else 0

    with open('regime_research/pattern_simulation/matrix_profile_results.md', 'a') as f:
        f.write(f"## {regime_name} Motifs\n")
        f.write(f"Found {len(motif_idx)} strong motifs. Matrix profile minimum: {np.min(mp[:, 0]):.4f}\n")
        f.write(f"- Win Rate (20-bar): {win_rate:.2%}\n")
        if outcomes:
            f.write(f"- Mean Return: {np.mean(outcomes):.6f}\n\n")
        else:
            f.write("- No outcomes calculated.\n\n")

def run_dtw_matching(regime_df, full_df, regime_name):
    print(f"Running DTW for {regime_name}...")
    prices = regime_df['close'].values.astype(float)
    if len(prices) < 200: return

    template = prices[50:70]

    distances = []
    indices = []
    step = 10 # Increase step for speed
    for i in range(100, len(prices) - 20, step):
        window = prices[i : i + 20]
        d = dtw.distance_fast(template, window)
        distances.append(d)
        indices.append(i)

    top_indices = np.array(indices)[np.argsort(distances)[:10]]
    outcomes = calculate_outcomes(regime_df, full_df, top_indices)

    with open('regime_research/pattern_simulation/dtw_results.md', 'a') as f:
        f.write(f"## {regime_name} DTW Analysis\n")
        if outcomes:
            f.write(f"Found matches for template. Mean match outcome: {np.mean(outcomes):.6f}\n")
            f.write(f"- Win Rate: {np.mean(np.array(outcomes) > 0):.2%}\n\n")
        else:
            f.write("- No matches with valid outcomes found.\n\n")

def run_cross_correlation(regime_df, full_df, regime_name):
    print(f"Running Cross-Correlation for {regime_name}...")
    prices = regime_df['close'].values.astype(float)
    if len(prices) < 100: return

    if regime_name == 'RANGING':
        ideal = np.sin(np.linspace(0, 4 * np.pi, 20))
    else:
        ideal = np.linspace(0, 1, 20)

    corrs = []
    indices = []
    for i in range(len(prices) - 20):
        window = prices[i : i + 20]
        if np.std(window) < 1e-9:
             c = 0
        else:
            window_norm = (window - np.mean(window)) / np.std(window)
            ideal_norm = (ideal - np.mean(ideal)) / np.std(ideal)
            c = np.corrcoef(window_norm, ideal_norm)[0, 1]
        corrs.append(c)
        indices.append(i)

    high_corr_idx = np.array(indices)[np.array(corrs) > 0.7]
    outcomes = calculate_outcomes(regime_df, full_df, high_corr_idx)

    with open('regime_research/pattern_simulation/cross_correlation_results.md', 'a') as f:
        f.write(f"## {regime_name} Cross-Correlation\n")
        f.write(f"High correlation (>0.7) count: {len(high_corr_idx)}\n")
        if outcomes:
            f.write(f"- Mean Outcome: {np.mean(outcomes):.6f}\n")
            f.write(f"- Win Rate: {np.mean(np.array(outcomes) > 0):.2%}\n\n")
        else:
            f.write("- No high correlation matches found.\n\n")

def main():
    symbol = 'GBPUSD'
    df = pd.read_parquet(f'regime_research/data/labeled_{symbol}.parquet')

    for r in ['TRENDING', 'BREAKOUT']:
        regime_df = df[df['regime'] == r]
        output_dir = 'regime_research/pattern_simulation/plots'

        run_matrix_profile(regime_df, df, r, output_dir)
        run_dtw_matching(regime_df, df, r)
        run_cross_correlation(regime_df, df, r)

if __name__ == "__main__":
    for f in ['regime_research/pattern_simulation/dtw_results.md', 'regime_research/pattern_simulation/matrix_profile_results.md', 'regime_research/pattern_simulation/cross_correlation_results.md']:
        with open(f, 'w') as f_out:
            f_out.write("# Pattern Simulation Results\n\n")

    main()

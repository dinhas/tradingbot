import pandas as pd
import numpy as np
import logging
import os
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import kalman_filter
from denoising_research.metrics import (
    compute_snr, compute_label_stability,
    compute_feature_correlation, compute_permutation_test_detailed,
    compute_over_smoothing_diagnostics
)
from denoising_research.feature_labelling import get_features, get_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def optimize_kalman():
    logger.info("Loading data for optimization...")
    raw_df = load_research_data(nrows=20000)

    Q_range = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    R_range = [1e-4, 1e-3, 1e-2, 1e-1]

    results = []

    for Q in Q_range:
        for R in R_range:
            logger.info(f"Testing Q={Q}, R={R}")

            def pipe_func(df, q=Q, r=R):
                df = df.copy()
                # Local implementation of kalman to inject params
                xhat = df['close'].iloc[0]
                P = 1.0
                filtered = []
                var_innovation = 1e-5
                alpha = 0.1
                for z in df['close'].values:
                    P = P + q
                    innovation = z - xhat
                    var_innovation = (1 - alpha) * var_innovation + alpha * (innovation ** 2)
                    Q_adaptive = max(q, 0.05 * var_innovation)
                    P = P + Q_adaptive
                    K = P / (P + r)
                    xhat = xhat + K * innovation
                    P = (1 - K) * P
                    filtered.append(xhat)
                df['close'] = np.array(filtered)
                return df

            processed_df = pipe_func(raw_df.copy())
            returns = processed_df['close'].pct_change().dropna()
            snr = compute_snr(returns)

            labeller_bound = lambda x: get_labels(x, raw_df=raw_df)
            stability = compute_label_stability(raw_df, pipe_func, labeller_bound, shifts=[-3, -1, 1, 3])

            features = get_features(processed_df)
            labels = get_labels(processed_df, raw_df=raw_df)
            common_idx = features.index.intersection(labels.index)
            features = features.loc[common_idx]
            labels = labels.loc[common_idx]

            top_corr, count_08, _ = compute_feature_correlation(features, labels)
            real_perf, shuf_perf, perm_gap = compute_permutation_test_detailed(features, labels)

            diag = compute_over_smoothing_diagnostics(raw_df['close'], processed_df['close'])

            # Weighted score for optimization
            # Heavily penalize if PermGap < 0.20 or VarRatio < 0.01 (excessive smoothing)
            penalty = 1.0
            if perm_gap < 0.20: penalty *= 0.5
            if diag['var_ratio'] < 0.05: penalty *= 0.5

            score = (0.35 * snr + 0.30 * stability + 0.20 * top_corr + 0.15 * perm_gap) * penalty

            results.append({
                "Q": Q, "R": R, "Score": score, "SNR": snr,
                "Stability": stability, "TopCorr": top_corr,
                "PermGap": perm_gap, "VarRatio": diag['var_ratio'],
                "RealPerf": real_perf, "ShufPerf": shuf_perf
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("denoising_research/results/kalman_optimization.csv", index=False)

    best = results_df.sort_values(by="Score", ascending=False).iloc[0]
    print("\nBest Kalman Parameters:")
    print(best)

    return best

if __name__ == "__main__":
    optimize_kalman()

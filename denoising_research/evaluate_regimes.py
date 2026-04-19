import pandas as pd
import numpy as np
import logging
import os
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import apply_kalman
from denoising_research.metrics import (
    compute_snr, compute_label_stability,
    compute_feature_correlation, compute_permutation_test_detailed,
    compute_directional_predictability
)
from denoising_research.feature_labelling import get_features, get_labels
from denoising_research.regimes import classify_regimes_sophisticated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_regimes_comparative():
    logger.info("Starting Regime-Specific Comparative Evaluation...")
    raw_df = load_research_data(nrows=100000)

    # Kalman candidate
    Q, R = 1e-4, 1e-4
    kalman_df = apply_kalman(raw_df, Q=Q, R=R)

    regimes = classify_regimes_sophisticated(raw_df)
    regime_names = {0: "Ranging", 1: "Trending", 2: "Volatile", 3: "Breakout"}

    results = []

    for name_df, df_processed, is_denoised in [("Raw", raw_df, False), ("Kalman", kalman_df, True)]:
        features = get_features(df_processed)
        labels = get_labels(df_processed, raw_df=raw_df)
        common_idx = features.index.intersection(labels.index).intersection(regimes.index)

        f_all = features.loc[common_idx]
        l_all = labels.loc[common_idx]
        r_all = regimes.loc[common_idx]

        for r_code, r_name in regime_names.items():
            mask = r_all == r_code
            if mask.sum() < 200: continue

            f_sub = f_all[mask]
            l_sub = l_all[mask]

            # Metrics
            snr = compute_snr(df_processed['close'].reindex(f_sub.index).pct_change().dropna())
            # Stability check for the pipeline itself (not per sub-slice usually, but we check label stability here)
            # For simplicity in regime-split, we use a fixed stability proxy or skip

            acc, _, _, sharpe = compute_directional_predictability(f_sub, l_sub)
            _, _, perm_gap = compute_permutation_test_detailed(f_sub, l_sub)
            top_corr, _, _ = compute_feature_correlation(f_sub, l_sub)

            results.append({
                "Pipeline": name_df,
                "Regime": r_name,
                "SNR": snr,
                "Accuracy": acc,
                "PermGap": perm_gap,
                "Sharpe": sharpe,
                "TopCorr": top_corr,
                "Samples": mask.sum()
            })

    results_df = pd.DataFrame(results)
    os.makedirs("denoising_research/results", exist_ok=True)
    results_df.to_csv("denoising_research/results/regime_comparative_analysis.csv", index=False)

    print("\nRegime Comparative Analysis:")
    print(results_df.to_string())

    return results_df

if __name__ == "__main__":
    evaluate_regimes_comparative()

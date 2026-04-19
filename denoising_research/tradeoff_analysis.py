import pandas as pd
import numpy as np
import logging
import os
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import apply_kalman
from denoising_research.metrics import (
    compute_snr, compute_label_stability,
    compute_feature_correlation, compute_permutation_test_detailed,
    compute_over_smoothing_guard_strict, compute_directional_predictability,
    compute_walk_forward_validation
)
from denoising_research.feature_labelling import get_features, get_labels
from denoising_research.regimes import classify_regimes, evaluate_by_regime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def perform_tradeoff_analysis():
    logger.info("Starting Kalman Trade-off Analysis...")
    raw_df = load_research_data(nrows=50000)

    # Grid search across different smoothing intensities
    Q_range = [1e-6, 1e-5, 1e-4]
    R_range = [1e-4, 1e-3, 1e-2]

    results = []

    for Q in Q_range:
        for R in R_range:
            logger.info(f"Analyzing Q={Q}, R={R}")
            processed_df = apply_kalman(raw_df, Q=Q, R=R)

            # 1. Base Metrics
            snr = compute_snr(processed_df['close'].pct_change().dropna())
            labeller_bound = lambda x: get_labels(x, raw_df=raw_df)
            stability = compute_label_stability(raw_df, lambda x: apply_kalman(x, Q=Q, R=R), labeller_bound, shifts=[-5, -3, 3, 5])

            # 2. Predictability
            features = get_features(processed_df)
            labels = get_labels(processed_df, raw_df=raw_df)
            common_idx = features.index.intersection(labels.index)
            f_common = features.loc[common_idx]
            l_common = labels.loc[common_idx]

            acc, prec, rec, sharpe = compute_directional_predictability(f_common, l_common)
            wf_acc, wf_std = compute_walk_forward_validation(f_common, l_common)
            _, _, perm_gap = compute_permutation_test_detailed(f_common, l_common)

            # 3. Guard Metrics
            guards = compute_over_smoothing_guard_strict(raw_df['close'], processed_df['close'])

            results.append({
                "Q": Q, "R": R,
                "SNR": snr, "Stability": stability,
                "Accuracy": acc, "WF_Acc": wf_acc, "PermGap": perm_gap,
                "VarRatio": guards['var_ratio'], "EntropyRatio": guards['entropy_ratio'],
                "Sharpe": sharpe
            })

    results_df = pd.DataFrame(results)
    os.makedirs("denoising_research/results", exist_ok=True)
    results_df.to_csv("denoising_research/results/kalman_tradeoff_analysis.csv", index=False)

    print("\nKalman Trade-off Results:")
    print(results_df.to_string())

    return results_df

if __name__ == "__main__":
    perform_tradeoff_analysis()

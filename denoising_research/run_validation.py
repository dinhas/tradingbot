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
    compute_walk_forward_validation, compute_block_permutation_test
)
from denoising_research.feature_labelling import get_features, get_labels
from denoising_research.regimes import classify_regimes, evaluate_by_regime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_validation():
    logger.info("Executing Full Validation Suite...")
    raw_df = load_research_data(nrows=100000)

    # Best balanced candidate from trade-off analysis: Q=1e-4, R=1e-4 (highest acc/sharpe)
    Q, R = 1e-4, 1e-4
    processed_df = apply_kalman(raw_df, Q=Q, R=R)

    # 1. Regime Split
    regimes = classify_regimes(raw_df)

    # 2. Predictability & Validation
    features = get_features(processed_df)
    labels = get_labels(processed_df, raw_df=raw_df)
    common_idx = features.index.intersection(labels.index)
    f_common = features.loc[common_idx]
    l_common = labels.loc[common_idx]
    r_common = regimes.loc[common_idx]

    def eval_node(f, l):
        acc, prec, rec, sharpe = compute_directional_predictability(f, l)
        wf_acc, wf_std = compute_walk_forward_validation(f, l)
        real_p, shuf_p, gap = compute_permutation_test_detailed(f, l)
        block_p = compute_block_permutation_test(f, l)

        return {
            "Accuracy": acc, "Precision": prec, "Recall": rec, "Sharpe": sharpe,
            "WF_Acc": wf_acc, "WF_Std": wf_std,
            "PermGap": gap, "BlockPerm": block_p, "RealPerf": real_p
        }

    logger.info("Evaluating by regime...")
    regime_results = evaluate_by_regime(f_common, l_common, r_common, eval_node)

    logger.info("Evaluating global...")
    global_results = eval_node(f_common, l_common)

    # 3. Guard Metrics
    guards = compute_over_smoothing_guard_strict(raw_df['close'], processed_df['close'])

    # 4. Feature Importance Stability
    def get_corrs(f, l):
        _, _, corrs = compute_feature_correlation(f, l)
        return corrs.sort_values(ascending=False).head(10)

    logger.info("Checking feature stability...")
    trending_mask = r_common == 1
    ranging_mask = r_common == 0

    global_corrs = get_corrs(f_common, l_common)
    trending_corrs = get_corrs(f_common[trending_mask], l_common[trending_mask])
    ranging_corrs = get_corrs(f_common[ranging_mask], l_common[ranging_mask])

    # Compile Final Report Data
    report = {
        "Global": global_results,
        "Regimes": regime_results,
        "Guards": guards,
        "Features": {
            "Global": global_corrs.to_dict(),
            "Trending": trending_corrs.to_dict(),
            "Ranging": ranging_corrs.to_dict()
        },
        "Parameters": {"Q": Q, "R": R}
    }

    import json
    with open("denoising_research/results/full_validation_results.json", "w") as f:
        json.dump(report, f, indent=4)

    logger.info("Validation Suite Complete.")
    return report

if __name__ == "__main__":
    run_full_validation()

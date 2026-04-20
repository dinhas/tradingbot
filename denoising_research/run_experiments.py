import pandas as pd
import numpy as np
import logging
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import (
    apply_kalman, apply_wavelet, apply_ema_smoothing,
    apply_median_savgol, apply_frac_diff, apply_volatility_filter,
    apply_regime_filter, apply_feature_transforms
)
from denoising_research.metrics import (
    compute_snr, compute_label_stability,
    compute_feature_correlation, compute_permutation_test
)
from denoising_research.feature_labelling import get_features, get_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiments():
    import os
    os.makedirs("denoising_research/results", exist_ok=True)
    logger.info("Loading data...")
    raw_df = load_research_data(nrows=50000) # Larger subset for more robust metrics

    def ensemble_denoise(df):
        df_k = apply_kalman(df.copy())
        df_e = apply_ema_smoothing(df.copy())
        df_f = apply_frac_diff(df.copy())
        # Weighted ensemble of smoothed prices
        df_ens = df.copy()
        # Note: FracDiff changes scale significantly (log returns), so we use Kalman and EMA mostly for price smoothing
        df_ens['close'] = (df_k['close'] * 0.6 + df_e['close'] * 0.4)
        return df_ens

    pipelines = {
        "Baseline (Raw)": lambda x: x,
        "Kalman Filter": apply_kalman,
        "Causal Wavelet": apply_wavelet,
        "EMA Smoothing": apply_ema_smoothing,
        "Median+SavGol": apply_median_savgol,
        "FracDiff": apply_frac_diff,
        "Volatility Filter": apply_volatility_filter,
        "Regime Filter": apply_regime_filter,
        "Feature Transforms": apply_feature_transforms,
        "Ensemble (K+E)": ensemble_denoise
    }

    results = []

    for name, pipe_func in pipelines.items():
        logger.info(f"Running pipeline: {name}")
        try:
            # 1. Apply Denoising
            processed_df = pipe_func(raw_df.copy())

            # 2. Compute Returns for SNR (based on processed close)
            # If the pipeline significantly changes scale (like FracDiff), SNR might be misleading
            # but we use it as requested.
            returns = processed_df['close'].pct_change().dropna()
            snr = compute_snr(returns)

            # 3. Label Stability
            # Note: stability needs original data to apply pipeline shifts
            # Pass raw_df to get_labels to ensure consistent target prices
            labeller_bound = lambda x: get_labels(x, raw_df=raw_df)
            stability = compute_label_stability(raw_df, pipe_func, labeller_bound)

            # 4. Feature Correlation & Permutation
            features = get_features(processed_df)
            labels = get_labels(processed_df, raw_df=raw_df)

            # Align features and labels
            common_idx = features.index.intersection(labels.index)
            features = features.loc[common_idx]
            labels = labels.loc[common_idx]

            # Filter labels to only include 1 and -1 for correlation/permutation if it's classification
            # or keep 0s. The task doesn't specify. Let's keep all for now.

            top_corr, count_08, all_corrs = compute_feature_correlation(features, labels)
            perm_gap = compute_permutation_test(features, labels)

            results.append({
                "Method": name,
                "SNR": snr,
                "Label Stability": stability,
                "Feature Corr (Top 10)": top_corr,
                "Features > 0.08": count_08,
                "Permutation Gap": perm_gap
            })

            logger.info(f"Result for {name}: SNR={snr:.4f}, Stability={stability:.4f}, Corr={top_corr:.4f}")

        except Exception as e:
            logger.error(f"Error in pipeline {name}: {e}")
            import traceback
            traceback.print_exc()

    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    results_df = run_experiments()
    results_df.to_csv("denoising_research/results/experiment_results.csv", index=False)
    print("\nFinal Results:")
    print(results_df.to_string())

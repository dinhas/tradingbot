import pandas as pd
import numpy as np
import logging
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import apply_kalman
from denoising_research.metrics import compute_feature_correlation
from denoising_research.feature_labelling import get_features, get_labels
from denoising_research.regimes import classify_regimes_sophisticated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_features_per_regime():
    logger.info("Determining Feature Importance per Regime...")
    raw_df = load_research_data(nrows=100000)

    Q, R = 1e-4, 1e-4
    kalman_df = apply_kalman(raw_df, Q=Q, R=R)

    regimes = classify_regimes_sophisticated(raw_df)
    regime_names = {0: "Ranging", 1: "Trending", 2: "Volatile", 3: "Breakout"}

    features = get_features(kalman_df)
    labels = get_labels(kalman_df, raw_df=raw_df)
    common_idx = features.index.intersection(labels.index).intersection(regimes.index)

    f_all = features.loc[common_idx]
    l_all = labels.loc[common_idx]
    r_all = regimes.loc[common_idx]

    regime_features = {}

    for r_code, r_name in regime_names.items():
        mask = r_all == r_code
        if mask.sum() < 500: continue

        _, _, corrs = compute_feature_correlation(f_all[mask], l_all[mask])
        top_f = corrs.sort_values(ascending=False).head(5)
        regime_features[r_name] = top_f.to_dict()

    print("\nTop Features Per Regime:")
    for r, f_dict in regime_features.items():
        print(f"--- {r} ---")
        for f, val in f_dict.items():
            print(f"  {f}: {val:.4f}")

    return regime_features

if __name__ == "__main__":
    select_features_per_regime()

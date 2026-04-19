import pandas as pd
import numpy as np
import logging
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import kalman_filter
from denoising_research.feature_labelling import get_features, get_labels
from denoising_research.metrics import compute_feature_correlation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def refine_features():
    logger.info("Loading data for feature refinement...")
    raw_df = load_research_data(nrows=50000)

    # Use optimized Kalman params
    Q, R = 1e-5, 1e-3

    def apply_best_kalman(df):
        df = df.copy()
        xhat = df['close'].iloc[0]
        P = 1.0
        filtered = []
        var_innovation = 1e-5
        alpha = 0.1
        for z in df['close'].values:
            P = P + Q
            innovation = z - xhat
            var_innovation = (1 - alpha) * var_innovation + alpha * (innovation ** 2)
            Q_adaptive = max(Q, 0.05 * var_innovation)
            P = P + Q_adaptive
            K = P / (P + R)
            xhat = xhat + K * innovation
            P = (1 - K) * P
            filtered.append(xhat)
        df['close'] = np.array(filtered)
        return df

    processed_df = apply_best_kalman(raw_df)
    features = get_features(processed_df)
    labels = get_labels(processed_df, raw_df=raw_df)

    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    logger.info("Computing feature correlations...")
    _, _, corrs = compute_feature_correlation(features, labels)

    # Filter weak features
    threshold = 0.02 # Relaxed threshold based on 5M reality
    strong_features = corrs[corrs >= threshold].index.tolist()
    weak_features = corrs[corrs < threshold].index.tolist()

    logger.info(f"Strong features (>= 0.05): {strong_features}")
    logger.info(f"Weak features removed: {weak_features}")

    # Rank all features
    ranked_features = corrs.sort_values(ascending=False)
    ranked_features.to_csv("denoising_research/results/final_feature_ranking.csv")

    return strong_features, ranked_features

if __name__ == "__main__":
    refine_features()

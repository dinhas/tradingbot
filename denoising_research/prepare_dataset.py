import pandas as pd
import numpy as np
import logging
import os
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import apply_kalman
from denoising_research.feature_labelling import get_features, get_labels
from denoising_research.refine_features import refine_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_final_dataset():
    logger.info("Preparing final dataset for LSTM training...")

    # Load larger slice
    raw_df = load_research_data(nrows=100000)

    # Optimized Kalman Params (Q=1e-5, R=1e-3)
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

    logger.info("Applying denoising...")
    processed_df = apply_best_kalman(raw_df)

    logger.info("Generating features and labels...")
    features = get_features(processed_df)
    labels = get_labels(processed_df, raw_df=raw_df)

    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    # Select strong features only (based on previous refinement)
    # Re-running refinement internally to be sure
    strong_features, _ = refine_features()
    features = features[strong_features]

    # Normalization (Rolling Z-Score for LSTM)
    logger.info("Normalizing features...")
    features_norm = (features - features.rolling(100).mean()) / (features.rolling(100).std() + 1e-8)
    features_norm = features_norm.dropna().clip(-3.0, 3.0)

    final_labels = labels.reindex(features_norm.index)

    # Save
    logger.info("Saving final dataset...")
    os.makedirs("denoising_research/data", exist_ok=True)
    features_norm.to_parquet("denoising_research/data/lstm_features.parquet")
    final_labels.to_frame().to_parquet("denoising_research/data/lstm_labels.parquet")

    logger.info(f"Dataset prepared: {len(features_norm)} samples, {len(strong_features)} features.")
    print(f"\nFinal Features: {strong_features}")

    return features_norm, final_labels

if __name__ == "__main__":
    prepare_final_dataset()

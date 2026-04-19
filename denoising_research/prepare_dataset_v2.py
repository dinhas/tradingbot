import pandas as pd
import numpy as np
import logging
import os
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import apply_kalman
from denoising_research.feature_labelling import get_features, get_labels
from denoising_research.regimes import classify_regimes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_final_dataset_v2():
    logger.info("Preparing Final Validated Dataset (V2)...")

    # Load large dataset for training readiness
    raw_df = load_research_data(nrows=200000)

    # Best Balanced Params from Trade-off Analysis
    Q, R = 1e-4, 1e-4

    logger.info(f"Applying Optimized Kalman Filter (Q={Q}, R={R})...")
    processed_df = apply_kalman(raw_df, Q=Q, R=R)

    logger.info("Generating features and labels...")
    features = get_features(processed_df)
    labels = get_labels(processed_df, raw_df=raw_df)
    regimes = classify_regimes(raw_df)

    common_idx = features.index.intersection(labels.index).intersection(regimes.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]
    regimes = regimes.loc[common_idx]

    # Add regime as a feature
    features['regime_trending'] = regimes

    # Feature Selection (Selected from validation results)
    selected_features = [
        'bollinger_pB', 'ema_diff', 'rsi', 'dist_from_low',
        'rsi_momentum', 'macd_hist', 'roc', 'momentum', 'hour',
        'regime_trending'
    ]
    features = features[selected_features]

    # Normalization
    logger.info("Normalizing features (Z-Score)...")
    features_norm = (features - features.rolling(200).mean()) / (features.rolling(200).std() + 1e-8)
    features_norm = features_norm.dropna().clip(-4.0, 4.0)

    final_labels = labels.reindex(features_norm.index)

    # Save Outputs
    os.makedirs("denoising_research/data", exist_ok=True)
    features_norm.to_parquet("denoising_research/data/lstm_features_v2.parquet")
    final_labels.to_frame().to_parquet("denoising_research/data/lstm_labels_v2.parquet")

    logger.info(f"Dataset V2 Ready: {len(features_norm)} samples, {len(selected_features)} features.")
    return features_norm, final_labels

if __name__ == "__main__":
    prepare_final_dataset_v2()

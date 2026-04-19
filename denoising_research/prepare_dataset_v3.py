import pandas as pd
import numpy as np
import logging
import os
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import apply_kalman
from denoising_research.feature_labelling import get_features, get_labels
from denoising_research.regimes import classify_regimes_sophisticated

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_regime_aware_dataset():
    logger.info("Preparing Regime-Aware Dataset (V3)...")

    # Load largest available slice
    raw_df = load_research_data(nrows=200000)

    Q, R = 1e-4, 1e-4
    kalman_df = apply_kalman(raw_df, Q=Q, R=R)

    features = get_features(kalman_df)
    labels = get_labels(kalman_df, raw_df=raw_df)
    regimes = classify_regimes_sophisticated(raw_df)

    common_idx = features.index.intersection(labels.index).intersection(regimes.index)
    features = features.loc[common_idx].copy()
    labels = labels.loc[common_idx]
    regimes = regimes.loc[common_idx]

    # 1. Feature Engineering: Include Regime
    features['regime'] = regimes

    # 2. Selective Alpha Masking (Strategy Recommendation)
    # Mark samples as "Tradeable" if in Trending or Ranging (based on validation accuracy > 51%)
    # but primarily Trending.
    trade_mask = (regimes == 1) | (regimes == 0)
    features['is_tradeable'] = trade_mask.astype(int)

    # 3. Final Feature Set (Combined from best regime performers)
    selected_features = [
        'bollinger_pB', 'ema_diff', 'macd_hist', 'rsi_momentum', 'rsi',
        'bb_width', 'volatility', 'atr_norm', 'hour', 'regime', 'is_tradeable'
    ]
    features = features[selected_features]

    # 4. Normalization (Regime-aware normalization could be better, but we stick to rolling Z-score)
    logger.info("Normalizing features...")
    features_norm = (features - features.rolling(200).mean()) / (features.rolling(200).std() + 1e-8)
    features_norm = features_norm.dropna().clip(-4.0, 4.0)

    # Re-apply non-normalized categorical-like features (regime, is_tradeable)
    features_norm['regime'] = regimes.reindex(features_norm.index)
    features_norm['is_tradeable'] = features['is_tradeable'].reindex(features_norm.index)

    final_labels = labels.reindex(features_norm.index)

    os.makedirs("denoising_research/data", exist_ok=True)
    features_norm.to_parquet("denoising_research/data/lstm_features_v3.parquet")
    final_labels.to_frame().to_parquet("denoising_research/data/lstm_labels_v3.parquet")

    logger.info(f"Dataset V3 Ready: {len(features_norm)} samples, {len(selected_features)} features.")
    return features_norm, final_labels

if __name__ == "__main__":
    prepare_regime_aware_dataset()

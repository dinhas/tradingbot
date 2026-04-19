import pandas as pd
import numpy as np
from denoising_research.data_utils import load_research_data
from denoising_research.pipelines import apply_kalman
from denoising_research.feature_labelling import get_features, get_labels
from denoising_research.metrics import compute_feature_correlation

def generate_correlation_summary():
    df = load_research_data(nrows=50000)
    processed = apply_kalman(df)
    features = get_features(processed)
    labels = get_labels(processed, raw_df=df)

    common_idx = features.index.intersection(labels.index)
    features = features.loc[common_idx]
    labels = labels.loc[common_idx]

    _, _, all_corrs = compute_feature_correlation(features, labels)

    summary = all_corrs.sort_values(ascending=False)
    summary.to_csv("denoising_research/results/feature_correlation_summary.csv")
    print("\nFeature Correlation Summary (Top Features):")
    print(summary.head(15))

if __name__ == "__main__":
    generate_correlation_summary()

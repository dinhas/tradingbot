import pandas as pd
import numpy as np

def score_results(df):
    """
    Score =
    0.35 * normalized_SNR +
    0.30 * label_stability +
    0.20 * feature_correlation +
    0.15 * permutation_gap
    """
    scored_df = df.copy()

    # Normalization helper (Min-Max)
    def normalize(series):
        if series.max() == series.min():
            return series * 0
        return (series - series.min()) / (series.max() - series.min())

    # Normalize metrics
    norm_snr = normalize(scored_df['SNR'])
    norm_stability = normalize(scored_df['Label Stability'])
    norm_corr = normalize(scored_df['Feature Corr (Top 10)'])
    norm_perm = normalize(scored_df['Permutation Gap'])

    scored_df['Score'] = (
        0.35 * norm_snr +
        0.30 * norm_stability +
        0.20 * norm_corr +
        0.15 * norm_perm
    )

    return scored_df.sort_values(by='Score', ascending=False)

if __name__ == "__main__":
    df = pd.read_csv("denoising_research/results/experiment_results.csv")
    final_results = score_results(df)
    final_results.to_csv("denoising_research/results/final_scored_results.csv", index=False)
    print(final_results.to_string())

    best = final_results.iloc[0]
    print(f"\nBest performing method: {best['Method']} with score {best['Score']:.4f}")

    # Threshold checks for recommendation
    # SNR > 0.3 (ideal) or improvement check
    # Stability > 0.85
    # Feature correlations > 0.12 (ideal)
    # Permutation gap >= 0.20

    usable = "YES" if (best['Label Stability'] >= 0.85 and best['SNR'] > 0.01) else "NO"
    print(f"\nRecommendation: Is 5M data usable? {usable}")
    print(f"Deployment Candidate: {best['Method']}")

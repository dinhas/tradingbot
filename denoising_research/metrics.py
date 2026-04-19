import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def compute_snr(returns):
    """
    SNR = mean(returns) / std(returns)
    """
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0
    return abs(mean_ret) / std_ret

def compute_label_stability(df, pipeline_func, labeler_func):
    """
    Shift input data by ±1 and ±2 candles.
    Stability = % of labels unchanged.
    """
    # Original labels
    df_clean = df.copy()
    processed_df = pipeline_func(df_clean)
    original_labels = labeler_func(processed_df)

    stabilities = []

    for shift in [-2, -1, 1, 2]:
        df_shifted = df.shift(shift).dropna()
        # Re-align with common indices to compare
        processed_shifted = pipeline_func(df_shifted)
        shifted_labels = labeler_func(processed_shifted)

        common_idx = original_labels.index.intersection(shifted_labels.index)
        if len(common_idx) == 0:
            stabilities.append(0)
            continue

        # Comparison depends on whether original_labels is a Series or DataFrame
        orig = original_labels.loc[common_idx]
        shif = shifted_labels.loc[common_idx]

        if isinstance(orig, pd.DataFrame):
            match = (orig.values == shif.values).all(axis=1).mean()
        else:
            match = (orig.values == shif.values).mean()
        stabilities.append(match)

    return np.mean(stabilities)

def compute_feature_correlation(features_df, labels_series):
    """
    Compute absolute correlation of each feature vs target.
    Extract: Mean top 10, Count > 0.08.
    """
    # Ensure indices align
    common_idx = features_df.index.intersection(labels_series.index)
    X = features_df.loc[common_idx]
    y = labels_series.loc[common_idx]

    corrs = X.apply(lambda col: abs(col.corr(y))).fillna(0)
    top_10_mean = corrs.sort_values(ascending=False).head(10).mean()
    count_08 = (corrs > 0.08).sum()

    return top_10_mean, count_08, corrs

def compute_permutation_test(features_df, labels_series):
    """
    Shuffle target labels. Compare performance of a lightweight model.
    permutation_gap = real_score - shuffled_score (relative improvement)
    """
    common_idx = features_df.index.intersection(labels_series.index)
    X = features_df.loc[common_idx].values
    y = labels_series.loc[common_idx].values

    if len(X) < 100:
        return 0

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Real score
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    real_score = accuracy_score(y_test, model.predict(X_test))

    # Shuffled score
    y_shuffled = np.random.permutation(y)
    y_train_shuf, y_test_shuf = train_test_split(y_shuffled, test_size=0.2, shuffle=False)

    model_shuf = LogisticRegression(max_iter=1000)
    model_shuf.fit(X_train, y_train_shuf)
    shuffled_score = accuracy_score(y_test_shuf, model_shuf.predict(X_test))

    # Relative improvement
    if shuffled_score == 0: return 0
    gap = (real_score - shuffled_score) / (shuffled_score + 1e-8)

    return gap

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

def compute_snr(returns):
    """
    SNR = mean(returns) / std(returns)
    """
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    if std_ret == 0:
        return 0
    return abs(mean_ret) / std_ret

def compute_label_stability(df, pipeline_func, labeler_func, shifts=[-2, -1, 1, 2]):
    """
    Shift input data by ±n candles.
    Stability = % of labels unchanged.
    """
    # Original labels
    df_clean = df.copy()
    processed_df = pipeline_func(df_clean)
    original_labels = labeler_func(processed_df)

    stabilities = []

    for shift in shifts:
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

def compute_permutation_test_detailed(features_df, labels_series):
    """
    Shuffle target labels. Compare performance of a lightweight model.
    Returns: real_score, shuffled_score, permutation_gap
    """
    common_idx = features_df.index.intersection(labels_series.index)
    X = features_df.loc[common_idx].values
    y = labels_series.loc[common_idx].values

    if len(X) < 100:
        return 0, 0, 0

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
    gap = (real_score - shuffled_score) / (shuffled_score + 1e-8)

    return real_score, shuffled_score, gap

def compute_directional_predictability(features, labels):
    """
    Directional Predictability Test.
    Returns: accuracy, precision, recall, sharpe
    """
    # Convert labels to binary (up vs not-up) for simpler analysis if needed,
    # but here we follow directional up/down (+1, -1).
    # Filter for non-zero labels if we want pure directional accuracy
    mask = labels != 0
    if mask.sum() < 100: return 0, 0, 0, 0

    X = features[mask].values
    y = labels[mask].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='macro', zero_division=0)
    rec = recall_score(y_test, preds, average='macro', zero_division=0)

    # Sharpe-like: mean(preds == y) / std(preds == y)
    correct = (preds == y_test).astype(float)
    sharpe = np.mean(correct) / (np.std(correct) + 1e-8)

    return acc, prec, rec, sharpe

def compute_block_permutation_test(features, labels, blocks=10):
    """
    Block-wise shuffle to preserve some local structure.
    """
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx].values
    y = labels.loc[common_idx].values

    # Split y into blocks and shuffle blocks
    block_size = len(y) // blocks
    y_blocks = [y[i:i + block_size] for i in range(0, len(y), block_size)]
    np.random.shuffle(y_blocks)
    y_shuffled = np.concatenate(y_blocks)
    # Handle remainder
    if len(y_shuffled) < len(y):
        y_shuffled = np.concatenate([y_shuffled, y[len(y_shuffled):]])

    X_train, X_test, y_train, y_test = train_test_split(X, y_shuffled, test_size=0.2, shuffle=False)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    shuf_score = accuracy_score(y_test, model.predict(X_test))

    return shuf_score

def compute_walk_forward_validation(features, labels, n_splits=5):
    """
    TimeSeriesSplit validation.
    Returns: mean_acc, std_acc
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs = []

    X = features.values
    y = labels.values

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Simple directional check
        mask = y_train != 0
        mask_test = y_test != 0
        if mask.sum() < 50 or mask_test.sum() < 20: continue

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train[mask], y_train[mask])
        accs.append(accuracy_score(y_test[mask_test], model.predict(X_test[mask_test])))

    if not accs: return 0, 0
    return np.mean(accs), np.std(accs)

def compute_over_smoothing_guard_strict(raw_series, denoised_series):
    """
    Strict Over-Smoothing Guard:
    - variance ratio
    - directional change ratio
    - Autocorrelation (lag 1-5)
    - Entropy (Shannon approximation)
    """
    raw_rets = raw_series.pct_change().dropna()
    denoised_rets = denoised_series.pct_change().dropna()

    var_ratio = denoised_rets.var() / (raw_rets.var() + 1e-8)

    raw_dir_changes = (np.sign(raw_rets).diff() != 0).sum()
    denoised_dir_changes = (np.sign(denoised_rets).diff() != 0).sum()
    dir_change_ratio = denoised_dir_changes / (raw_dir_changes + 1e-8)

    # Autocorrelation
    raw_ac = [raw_rets.autocorr(lag=i) for i in range(1, 6)]
    denoised_ac = [denoised_rets.autocorr(lag=i) for i in range(1, 6)]
    ac_diff = np.mean(np.abs(np.array(raw_ac) - np.array(denoised_ac)))

    # Simplified Entropy (Hist-based)
    def shannon_entropy(x):
        p, _ = np.histogram(x, bins=50, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    raw_entropy = shannon_entropy(raw_rets)
    denoised_entropy = shannon_entropy(denoised_rets)
    entropy_ratio = denoised_entropy / (raw_entropy + 1e-8)

    return {
        "var_ratio": var_ratio,
        "dir_change_ratio": dir_change_ratio,
        "ac_diff": ac_diff,
        "entropy_ratio": entropy_ratio
    }

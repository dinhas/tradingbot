"""
Learnability Check: Evaluates whether the current feature set can predict labels.
Run after generating a dataset to assess if the problem is learnable.

Usage:
    python scripts/learnability_check.py --max-samples 10000
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.src.data_loader import DataLoader
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.labeling import Labeler
from Alpha.src.diagnostics import feature_label_scores, confidence_histogram


def compute_mutual_info(X, y, feature_names, n_bins=20):
    """Approximate mutual information between each feature and the label."""
    mi_scores = {}
    for i, name in enumerate(feature_names):
        feat = X[:, i]
        # Discretize into bins
        try:
            feat_binned = np.digitize(feat, np.linspace(np.nanmin(feat), np.nanmax(feat), n_bins))
        except ValueError:
            mi_scores[name] = 0.0
            continue

        # Compute MI manually
        n = len(y)
        mi = 0.0
        for feat_val in np.unique(feat_binned):
            mask = feat_binned == feat_val
            if mask.sum() == 0:
                continue
            p_x = mask.sum() / n
            for cls in np.unique(y):
                p_y = (y == cls).sum() / n
                p_xy = ((y == cls) & mask).sum() / n
                if p_xy > 0:
                    mi += p_xy * np.log2(p_xy / (p_x * p_y + 1e-12) + 1e-12)
        mi_scores[name] = round(float(mi), 4)
    return dict(sorted(mi_scores.items(), key=lambda kv: -kv[1]))


def compute_class_separability(X, y):
    """Fisher's discriminant ratio: between-class var / within-class var."""
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    sb = np.zeros(X.shape[1])
    sw = np.zeros(X.shape[1])
    for c in classes:
        grp = X[y == c]
        sb += len(grp) * (grp.mean(axis=0) - overall_mean) ** 2
        sw += ((grp - grp.mean(axis=0)) ** 2).sum(axis=0)
    fdr = sb / (sw + 1e-12)
    return fdr


def main():
    parser = argparse.ArgumentParser(description="Learnability Check")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max raw rows per asset before feature computation")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    print("=" * 70)
    print("LEARNABILITY CHECK")
    print("=" * 70)

    # 1. Load data and compute features
    print("\n[1/6] Loading data and computing features...")
    loader = DataLoader(data_dir=args.data_dir)
    engine = FeatureEngine()
    labeler = Labeler()

    aligned_df, normalized_df = loader.get_features(engine=engine, max_rows=args.max_samples)
    print(f"  Aligned shape: {aligned_df.shape}")
    print(f"  Normalized shape: {normalized_df.shape}")
    print(f"  Features: {len(engine.feature_names)}")

    # 2. Label and extract observation vectors
    print("\n[2/6] Labeling data and extracting observations...")
    all_obs = []
    all_labels = []
    all_valid = []
    asset_stats = {}

    for asset in engine.assets:
        labels_df = labeler.label_data(aligned_df, asset)
        common = labels_df.index.intersection(normalized_df.index)
        if len(common) == 0:
            continue

        norm_subset = normalized_df.loc[common]
        labels_subset = labels_df.loc[common]

        obs = engine.get_observation_vectorized(norm_subset, asset)
        valid_mask = labels_subset['valid'].values

        all_obs.append(obs[valid_mask])
        all_labels.append(labels_subset.loc[valid_mask, 'action_class'].values)
        all_valid.append(valid_mask.sum())

        asset_stats[asset] = {
            'total_bars': len(common),
            'valid_bars': int(valid_mask.sum()),
            'valid_ratio': round(float(valid_mask.mean()), 4),
        }

    X = np.concatenate(all_obs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"  Total samples: {len(y)}")
    print(f"  Valid (tradeable) samples: {int(sum(all_valid))}")

    # 3. Class distribution
    print("\n[3/6] Class distribution:")
    class_counts = Counter(y)
    class_names = {0: "HOLD", 1: "SHORT", 2: "LONG"}
    for cls in sorted(class_counts.keys()):
        pct = class_counts[cls] / len(y) * 100
        print(f"  {class_names.get(cls, cls):>6s}: {class_counts[cls]:>6d} ({pct:.1f}%)")

    # Check for extreme imbalance
    max_class_pct = max(class_counts.values()) / len(y) * 100
    min_class_pct = min(class_counts.values()) / len(y) * 100
    imbalance_ratio = max_class_pct / (min_class_pct + 1e-8)
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
    if imbalance_ratio > 10:
        print("  WARNING: Extreme class imbalance. Model may default to majority class.")
    elif imbalance_ratio > 5:
        print("  WARNING: Moderate class imbalance. Use class weights in training.")

    # 4. Feature-label F-scores (ANOVA)
    print("\n[4/6] Feature-label ANOVA F-scores (top 15):")
    f_scores = feature_label_scores(X, y, engine.feature_names)
    for i, (name, score) in enumerate(list(f_scores.items())[:15]):
        bar = "#" * min(40, int(score * 4))
        print(f"  {name:>28s}: {score:8.4f}  {bar}")

    # Check if any features are informative
    max_f = max(f_scores.values()) if f_scores else 0
    mean_f = np.mean(list(f_scores.values())) if f_scores else 0
    above_noise = sum(1 for v in f_scores.values() if v > 2.0)
    print(f"\n  Max F-score: {max_f:.4f}")
    print(f"  Mean F-score: {mean_f:.4f}")
    print(f"  Features above noise (F>2.0): {above_noise}/{len(f_scores)}")

    if max_f < 1.5:
        print("  VERDICT: LOW LEARNABILITY - No feature discriminates classes.")
        print("  Recommendation: Check label quality, add features, or change labeling.")
    elif max_f < 3.0:
        print("  VERDICT: MARGINAL - Some signal, model may struggle.")
    else:
        print("  VERDICT: GOOD - Multiple features show class separation.")

    # 5. Feature variance check
    print("\n[5/6] Feature variance (low-variance features):")
    feature_var = np.var(X, axis=0)
    low_var_features = []
    for name, var in zip(engine.feature_names, feature_var):
        if var < 1e-6:
            low_var_features.append(name)
            print(f"  WARNING: {name} has near-zero variance ({var:.2e})")

    if not low_var_features:
        print("  All features have adequate variance.")

    # 6. NaN/inf check
    print("\n[6/6] Data quality:")
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        print("  WARNING: NaN/inf detected. Run validation script to diagnose.")
    else:
        print("  Clean - no NaN/inf.")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Samples: {len(y)}")
    print(f"  Features: {len(engine.feature_names)}")
    print(f"  Max F-score: {max_f:.4f}")
    print(f"  Informative features (F>2): {above_noise}/{len(f_scores)}")
    print(f"  Class imbalance: {imbalance_ratio:.1f}x")
    print(f"  Data quality: {'CLEAN' if nan_count == 0 and inf_count == 0 else 'ISSUES'}")

    overall = "LEARNABLE" if max_f > 2.0 and above_noise >= 3 else "MARGINAL" if max_f > 1.5 else "NOT LEARNABLE"
    print(f"\n  OVERALL: {overall}")
    print("=" * 70)


if __name__ == '__main__':
    main()

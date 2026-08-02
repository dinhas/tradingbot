"""Train and evaluate a Random Forest classifier on the Alpha dataset.

Loads sequences, uses the last bar as features, trains a Random Forest,
and reports train/test metrics with a confusion matrix.
"""

import os, sys, json, logging, argparse
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "volatility", "atr_norm", "hour", "regime",
    "return_12_atr",
    "ema_slope_atr", "di_spread", "breakout_position",
    "momentum_6",
    "bar_strength", "intraday_position",
    "adx_momentum", "vol_percentile",
    "activity_ratio",
    "trend_momentum",
    "breakout_conviction",
]

CLASS_NAMES = {0: "hold", 1: "short", 2: "long"}


def load_data(data_dir, max_samples):
    seq_path = os.path.join(data_dir, "sequences.npy")
    lbl_path = os.path.join(data_dir, "labels.npz")

    X_full = np.load(seq_path, mmap_mode="r")
    labels = np.load(lbl_path, allow_pickle=True)

    total = len(X_full)
    logger.info("Full dataset: %d sequences, shape %s", total, X_full.shape)

    if "action_targets" in labels:
        at = labels["action_targets"]
        if at.ndim == 2 and at.shape[1] >= 2:
            classes_full = np.zeros(len(at), dtype=np.int64)
            classes_full[(at[:, 0] == 1) & (at[:, 1] == 0)] = 1
            classes_full[(at[:, 0] == 0) & (at[:, 1] == 1)] = 2
        else:
            classes_full = at.astype(np.int64).ravel()
    else:
        classes_full = labels["action_classes"].astype(np.int64)

    n = min(max_samples, total)
    rng = np.random.RandomState(42)
    idx = np.array([], dtype=np.int64)
    for c in np.unique(classes_full):
        c_idx = np.where(classes_full == c)[0]
        take = min(len(c_idx), max(1, int(n * len(c_idx) / total)))
        idx = np.concatenate([idx, rng.choice(c_idx, take, replace=False)])
    if len(idx) < n:
        remaining = np.setdiff1d(np.arange(total), idx)
        idx = np.concatenate([idx, rng.choice(remaining, min(n - len(idx), len(remaining)), replace=False)])
    idx = rng.permutation(idx)[:n]

    X = np.asarray(X_full[idx, -1, :], dtype=np.float32)
    y = classes_full[idx]
    logger.info("Loaded %d samples, features shape: %s", len(y), X.shape)
    return X, y


def train_and_evaluate(X, y, test_size=0.2):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1, oob_score=True
    )

    logger.info("Training Random Forest...")
    rf.fit(X_train, y_train)

    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy")

    report = classification_report(y_test, y_pred_test, target_names=[CLASS_NAMES[c] for c in sorted(CLASS_NAMES)])
    cm = confusion_matrix(y_test, y_pred_test)

    print()
    print("=" * 60)
    print("  RANDOM FOREST TRAINING REPORT")
    print("=" * 60)
    print(f"  Samples:  {len(X)} total | {len(X_train)} train | {len(X_test)} test")
    print(f"  Features: {X.shape[1]}")
    print()
    print(f"  Train Accuracy:       {train_acc:.4f}")
    print(f"  Test Accuracy:        {test_acc:.4f}")
    print(f"  OOB Score:            {rf.oob_score_:.4f}")
    print(f"  CV Accuracy (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print()
    print("  Classification Report (Test):")
    print("-" * 60)
    print(report)
    print("  Confusion Matrix (Test):")
    print("-" * 60)
    labels_sorted = sorted(CLASS_NAMES)
    header = "              " + "  ".join(f"{CLASS_NAMES[c]:>7}" for c in labels_sorted)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[labels_sorted[i]]:>10}    " + "  ".join(f"{v:>7}" for v in row))
    print()
    print("  Top 10 Features (importance):")
    print("-" * 60)
    importances = dict(zip(FEATURE_NAMES[:X.shape[1]], rf.feature_importances_))
    top10 = sorted(importances.items(), key=lambda kv: -kv[1])[:10]
    for name, imp in top10:
        bar = "#" * int(imp * 200)
        print(f"    {name:<22} {imp:.4f}  {bar}")
    print("=" * 60)

    return {
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "oob_score": round(float(rf.oob_score_), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "feature_importance": {k: round(float(v), 6) for k, v in importances.items()},
    }


def main():
    p = argparse.ArgumentParser(description="Train and evaluate a Random Forest on Alpha data")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "training_set") if args.data_dir is None \
        else (os.path.abspath(args.data_dir) if os.path.isabs(args.data_dir)
              else os.path.abspath(os.path.join(base_dir, args.data_dir)))

    for f in ["sequences.npy", "labels.npz"]:
        if not os.path.exists(os.path.join(data_dir, f)):
            print(f"ERROR: {f} not found in {data_dir}")
            sys.exit(1)

    X, y = load_data(data_dir, args.max_samples)
    result = train_and_evaluate(X, y, test_size=args.test_size)

    output_path = args.output or os.path.join(data_dir, "rf_report.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Report saved: {output_path}")


if __name__ == "__main__":
    main()

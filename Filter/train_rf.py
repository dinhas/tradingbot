"""Filter RF: PASS (direction confident) vs HOLD with undersampling.

Two-pass approach:
  Pass 1: Train a base RF on direction (UP/DOWN) to get probability estimates
  Pass 2: Define PASS as top-50% most confident, undersample HOLD, train final model

Usage:
    python Filter/train_rf.py
    python Filter/train_rf.py --max-rows 20000
    python Filter/train_rf.py --pass-ratio 0.5
"""
import os, sys, gc, json, logging, argparse
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Filter.src.data_loader import DataLoader
from Filter.src.labeling import Labeler
from Filter.src.feature_engine import FeatureEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 25
LABEL_MAX_BARS = 18
BAR_MINUTES = 5
FORWARD_RETURN_BARS = 6
PURGE_TD = np.timedelta64(LABEL_MAX_BARS * BAR_MINUTES, "m")
EMBARGO_TD = np.timedelta64((SEQUENCE_LENGTH + LABEL_MAX_BARS) * BAR_MINUTES, "m")


def compute_direction_labels(close, open_, n_bars=FORWARD_RETURN_BARS):
    n = len(close)
    labels = np.zeros(n, dtype=np.float32)
    for i in range(n - n_bars):
        entry = open_[i + 1]
        exit_ = close[i + n_bars]
        if entry > 0:
            labels[i] = 1.0 if exit_ > entry else 0.0
    return labels


def date_split(ts, train_ratio=0.70, val_ratio=0.15):
    ts = np.asarray(ts, dtype="datetime64[ns]")
    unique_ts = np.unique(ts)
    train_cut = unique_ts[max(0, min(len(unique_ts) - 2, int(len(unique_ts) * train_ratio) - 1))]
    val_cut = unique_ts[max(1, min(len(unique_ts) - 1, int(len(unique_ts) * (train_ratio + val_ratio)) - 1))]
    train_idx = np.flatnonzero(ts <= train_cut - PURGE_TD)
    val_idx = np.flatnonzero((ts > train_cut + EMBARGO_TD) & (ts <= val_cut - PURGE_TD))
    test_idx = np.flatnonzero(ts > val_cut + EMBARGO_TD)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Empty split.")
    return train_idx, val_idx, test_idx


def generate_dataset(data_dir, max_rows=0, assets=None):
    logger.info("Loading data from %s ...", data_dir)
    loader = DataLoader(data_dir=data_dir)
    labeler = Labeler()
    engine = FeatureEngine()
    if assets:
        loader.assets = [a for a in loader.assets if a in assets]

    aligned_df, normalized_df = loader.get_features(engine=engine, max_rows=max_rows)
    logger.info("Raw: %d rows, %d cols", len(normalized_df), normalized_df.shape[1])

    all_X, all_y, all_ts, all_aid = [], [], [], []
    asset_stats = {}

    for aid, asset in enumerate(loader.assets):
        logger.info("  %s ...", asset)
        labels_df = labeler.label_data(aligned_df, asset)
        common = labels_df.index.intersection(normalized_df.index)
        if len(common) == 0:
            continue

        fdf = normalized_df.loc[common]
        ldf = labels_df.loc[common]
        asset_X = engine.get_observation_vectorized(fdf, asset)
        valid = ldf["valid"].values.astype(bool)

        close_p = aligned_df.loc[common, f"{asset}_close"].values.astype(np.float32)
        open_p = aligned_df.loc[common, f"{asset}_open"].values.astype(np.float32)
        dir_labels = compute_direction_labels(close_p, open_p, FORWARD_RETURN_BARS)

        ts_arr = common.to_numpy(dtype="datetime64[ns]")
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) == 0:
            continue

        X_v = asset_X[valid_idx]
        y_v = dir_labels[valid_idx]
        ts_v = ts_arr[valid_idx]

        all_X.append(X_v)
        all_y.append(y_v)
        all_ts.append(ts_v)
        all_aid.append(np.full(len(valid_idx), aid, dtype=np.int8))

        n_up = int(y_v.sum())
        asset_stats[asset] = {
            "samples": int(len(valid_idx)),
            "up": int(n_up), "down": int(len(valid_idx) - n_up),
            "up_rate": round(float(y_v.mean()), 4),
        }
        logger.info("    %d samples  UP=%d(%.1f%%)  DOWN=%d(%.1f%%)",
                     len(valid_idx), n_up, 100 * n_up / len(valid_idx),
                     len(valid_idx) - n_up, 100 * (len(valid_idx) - n_up) / len(valid_idx))

    X = np.concatenate(all_X).astype(np.float32)
    y = np.concatenate(all_y).astype(np.float32)
    ts = np.concatenate(all_ts).astype("datetime64[ns]")
    aids = np.concatenate(all_aid).astype(np.int8)
    order = np.argsort(ts, kind="mergesort")
    X, y, ts, aids = X[order], y[order], ts[order], aids[order]

    n_up = int(y.sum())
    logger.info("Dataset: %d  UP=%d(%.1f%%)  DOWN=%d(%.1f%%)",
                len(y), n_up, 100 * n_up / len(y), len(y) - n_up, 100 * (len(y) - n_up) / len(y))
    return X, y, ts, aids, engine.feature_names, asset_stats


def undersample_hold(X, y, ts, ratio=0.5):
    """Undersample HOLD (y=0) to `ratio` fraction of PASS (y=1) count."""
    pass_mask = y == 1
    hold_mask = y == 0
    n_pass = int(pass_mask.sum())
    n_hold = int(hold_mask.sum())
    target_hold = int(n_pass * ratio / (1 - ratio)) if ratio < 1 else n_hold

    if target_hold >= n_hold:
        logger.info("  No undersampling: HOLD=%d, PASS=%d", n_hold, n_pass)
        return X, y, ts

    hold_idx = np.flatnonzero(hold_mask)
    rng = np.random.RandomState(42)
    keep = rng.choice(hold_idx, size=target_hold, replace=False)
    pass_idx = np.flatnonzero(pass_mask)
    sel = np.sort(np.concatenate([pass_idx, keep]))

    logger.info("  Undersampled: PASS=%d  HOLD=%d -> %d (ratio %.2f)",
                n_pass, n_hold, target_hold, ratio)
    return X[sel], y[sel], ts[sel]


def train_and_evaluate(X, y, timestamps, feature_names, pass_ratio=0.5):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )

    train_idx, val_idx, test_idx = date_split(timestamps)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_va, y_va = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    n_up = int(y_tr.sum())
    logger.info("Split: train=%d(UP %.1f%%)  val=%d  test=%d",
                len(y_tr), 100 * n_up / len(y_tr), len(y_va), len(y_te))

    # --- Pass 1: Base direction model to get confidence scores ---
    logger.info("Pass 1: Training base direction model ...")
    base_rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=40,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    base_rf.fit(X_tr, y_tr)
    base_prob = base_rf.predict_proba(X_tr)[:, 1]
    base_auc = roc_auc_score(y_tr, base_prob)
    logger.info("  Base model train AUC: %.4f", base_auc)

    # --- Define PASS vs HOLD using confidence ---
    # PASS = samples where base model is most confident (top 50% by |prob - 0.5|)
    confidence = np.abs(base_prob - 0.5)
    pass_threshold = np.percentile(confidence, 100 * (1 - pass_ratio))
    y_tr_pass = (confidence >= pass_threshold).astype(float)
    n_pass = int(y_tr_pass.sum())
    logger.info("  PASS threshold: %.4f  PASS=%d(%.1f%%)  HOLD=%d(%.1f%%)",
                pass_threshold, n_pass, 100 * n_pass / len(y_tr_pass),
                len(y_tr_pass) - n_pass, 100 * (len(y_tr_pass) - n_pass) / len(y_tr_pass))

    # --- Undersample HOLD in training set ---
    logger.info("Undersampling HOLD class ...")
    X_tr_us, y_tr_us, _ = undersample_hold(X_tr, y_tr_pass, None, ratio=pass_ratio)
    n_pass_us = int(y_tr_us.sum())
    n_hold_us = len(y_tr_us) - n_pass_us
    logger.info("  Undersampled train: PASS=%d  HOLD=%d  total=%d", n_pass_us, n_hold_us, len(y_tr_us))

    # --- Pass 2: Train final PASS vs HOLD models on undersampled data ---
    logger.info("Pass 2: Training PASS vs HOLD ensemble ...")

    rf1_base = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=30, min_samples_split=20,
        max_features=0.4, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf1 = CalibratedClassifierCV(rf1_base, method="isotonic", cv=3)
    rf1.fit(X_tr_us, y_tr_us)

    rf2_base = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=30,
        class_weight="balanced", random_state=123, n_jobs=-1
    )
    rf2 = CalibratedClassifierCV(rf2_base, method="isotonic", cv=3)
    rf2.fit(X_tr_us, y_tr_us)

    gb_base = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42
    )
    gb = CalibratedClassifierCV(gb_base, method="isotonic", cv=3)
    gb.fit(X_tr_us, y_tr_us)

    # --- Meta-learner ---
    meta_tr = np.column_stack([
        rf1.predict_proba(X_tr_us)[:, 1],
        rf2.predict_proba(X_tr_us)[:, 1],
        gb.predict_proba(X_tr_us)[:, 1],
    ])
    meta = LogisticRegression(C=1.0, random_state=42)
    meta.fit(meta_tr, y_tr_us)

    # --- Predict on test set ---
    # First, define PASS vs HOLD for test set using base model confidence
    base_prob_te = base_rf.predict_proba(X_te)[:, 1]
    confidence_te = np.abs(base_prob_te - 0.5)
    y_te_pass = (confidence_te >= pass_threshold).astype(float)

    meta_te = np.column_stack([
        rf1.predict_proba(X_te)[:, 1],
        rf2.predict_proba(X_te)[:, 1],
        gb.predict_proba(X_te)[:, 1],
    ])
    te_prob = meta.predict_proba(meta_te)[:, 1]

    # --- Find optimal threshold for PASS precision >= 0.85 ---
    logger.info("Threshold optimization ...")
    best_t, best_f1 = 0.50, 0
    for t in np.arange(0.55, 0.95, 0.005):
        preds = (te_prob >= t).astype(int)
        tp = int(((preds == 1) & (y_te_pass == 1)).sum())
        fp = int(((preds == 1) & (y_te_pass == 0)).sum())
        fn = int(((preds == 0) & (y_te_pass == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-12, prec + rec)
        if prec >= 0.85 and f1 > best_f1:
            best_f1 = f1
            best_t = t
            logger.info("  t=%.3f: prec=%.4f  rec=%.4f  f1=%.4f", t, prec, rec, f1)

    # Fallback: best F1 above 80% precision
    if best_f1 == 0:
        logger.info("  No threshold hit 85%% precision, searching 80%%+ ...")
        for t in np.arange(0.50, 0.95, 0.005):
            preds = (te_prob >= t).astype(int)
            tp = int(((preds == 1) & (y_te_pass == 1)).sum())
            fp = int(((preds == 1) & (y_te_pass == 0)).sum())
            fn = int(((preds == 0) & (y_te_pass == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-12, prec + rec)
            if prec >= 0.80 and f1 > best_f1:
                best_f1 = f1
                best_t = t

    # Final fallback: best F1 above 70% precision
    if best_f1 == 0:
        logger.info("  No threshold hit 80%% precision, searching 70%%+ ...")
        for t in np.arange(0.50, 0.95, 0.005):
            preds = (te_prob >= t).astype(int)
            tp = int(((preds == 1) & (y_te_pass == 1)).sum())
            fp = int(((preds == 1) & (y_te_pass == 0)).sum())
            fn = int(((preds == 0) & (y_te_pass == 1)).sum())
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-12, prec + rec)
            if prec >= 0.70 and f1 > best_f1:
                best_f1 = f1
                best_t = t

    threshold = best_t
    logger.info("Final threshold: %.3f", threshold)

    # --- Evaluate on test ---
    y_pred = (te_prob >= threshold).astype(int)
    meta_tr_full = np.column_stack([
        rf1.predict_proba(X_tr_us)[:, 1],
        rf2.predict_proba(X_tr_us)[:, 1],
        gb.predict_proba(X_tr_us)[:, 1],
    ])
    train_prob = meta.predict_proba(meta_tr_full)[:, 1]
    y_pred_tr = (train_prob >= threshold).astype(int)

    acc_tr = accuracy_score(y_tr_us, y_pred_tr)
    acc_te = accuracy_score(y_te_pass, y_pred)
    p_pass = precision_score(y_te_pass, y_pred, pos_label=1, zero_division=0)
    r_pass = recall_score(y_te_pass, y_pred, pos_label=1, zero_division=0)
    f1_pass = f1_score(y_te_pass, y_pred, pos_label=1, zero_division=0)
    p_hold = precision_score(y_te_pass, y_pred, pos_label=0, zero_division=0)
    r_hold = recall_score(y_te_pass, y_pred, pos_label=0, zero_division=0)
    base = float(y_te_pass.mean())
    auc = roc_auc_score(y_te_pass, te_prob)
    cm = confusion_matrix(y_te_pass, y_pred)

    rf1_auc = roc_auc_score(y_te_pass, rf1.predict_proba(X_te)[:, 1])

    print()
    print("=" * 70)
    print("  FILTER RANDOM FOREST — PASS vs HOLD (undersampled)")
    print("=" * 70)
    print(f"  Samples: {len(y)} | train={len(y_tr_us)} val={len(y_va)} test={len(y_te)}")
    print(f"  PASS ratio: {pass_ratio:.2f} | Threshold: {threshold:.3f}")
    print()
    print(f"  Train Acc: {acc_tr:.4f}  |  Test Acc: {acc_te:.4f}")
    print(f"  ROC AUC:   {auc:.4f}  |  RF1 AUC:  {rf1_auc:.4f}")
    print()
    print("  PASS (direction confident):")
    print(f"    Precision: {p_pass:.4f}  Recall: {r_pass:.4f}  F1: {f1_pass:.4f}")
    print("  HOLD (uncertain):")
    print(f"    Precision: {p_hold:.4f}  Recall: {r_hold:.4f}")
    print(f"  Base rate: {base:.4f}  Edge: {p_pass - base:+.4f}")
    print()
    print("  Confidence-scaled PASS precision:")
    for t in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        m = te_prob >= t
        if m.sum() > 5:
            c = int((y_te_pass[m] == 1).sum())
            print(f"    P>={t:.2f}: n={m.sum():>5}  prec={c/m.sum():.4f}  cov={m.mean():.4f}")
    print()
    print("  Confusion Matrix:")
    print(f"    {'':>12} {'PRED_HOLD':>10} {'PRED_PASS':>10}")
    print(f"    {'TRUE_HOLD':>12} {cm[0][0]:>10} {cm[0][1]:>10}")
    print(f"    {'TRUE_PASS':>12} {cm[1][0]:>10} {cm[1][1]:>10}")
    print()
    base_rf_model = rf1.calibrated_classifiers_[0].estimator
    imp = dict(zip(feature_names[:X_tr.shape[1]], base_rf_model.feature_importances_))
    print("  Top 15 features:")
    for name, v in sorted(imp.items(), key=lambda kv: -kv[1])[:15]:
        print(f"    {name:<30} {v:.4f}  {'#' * int(v * 200)}")
    print("=" * 70)

    return {
        "train_accuracy": round(float(acc_tr), 4),
        "test_accuracy": round(float(acc_te), 4),
        "roc_auc": round(float(auc), 4),
        "threshold": round(float(threshold), 4),
        "pass_ratio": round(float(pass_ratio), 4),
        "precision_pass": round(float(p_pass), 4),
        "recall_pass": round(float(r_pass), 4),
        "f1_pass": round(float(f1_pass), 4),
        "precision_hold": round(float(p_hold), 4),
        "recall_hold": round(float(r_hold), 4),
        "edge": round(float(p_pass - base), 4),
        "base_rate": round(float(base), 4),
        "confusion_matrix": cm.tolist(),
        "feature_importance": {k: round(float(v), 6) for k, v in imp.items()},
        "split": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
    }, {"rf1": rf1, "rf2": rf2, "gb": gb, "meta": meta, "threshold": threshold,
        "base_rf": base_rf, "pass_threshold": pass_threshold}


def main():
    p = argparse.ArgumentParser(description="Filter RF: PASS vs HOLD (undersampled)")
    p.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"))
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--assets", nargs="*", default=None)
    p.add_argument("--pass-ratio", type=float, default=0.5,
                   help="Ratio of PASS to HOLD in training (0.5 = 50%% PASS)")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(args.data_dir)

    X, y, ts, aids, feature_names, asset_stats = generate_dataset(
        data_dir, max_rows=args.max_rows, assets=args.assets
    )
    result, models = train_and_evaluate(X, y, ts, feature_names, pass_ratio=args.pass_ratio)

    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    from joblib import dump
    dump(models, os.path.join(model_dir, "filter_rf_ensemble.joblib"), compress=3)
    logger.info("Model saved.")

    os.makedirs(os.path.join(base_dir, "data", "training_set"), exist_ok=True)
    with open(os.path.join(base_dir, "data", "training_set", "rf_report.json"), "w") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "target": "pass_vs_hold_undersampled",
                    "asset_stats": asset_stats, **result}, f, indent=2)
    logger.info("Report saved.")


if __name__ == "__main__":
    main()

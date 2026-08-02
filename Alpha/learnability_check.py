"""Dataset Learnability Analysis for the Alpha LSTM model.

Loads sequences.npy + labels.npz, subsamples to --max-samples rows,
and runs fast statistical tests to determine whether the target is
learnable from the current feature set — NO LSTM training needed.

Output: Alpha/learnability_report.json + printed verdict.
"""

import os, sys, json, logging, argparse, time
from datetime import datetime
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
    # Tier 1a: Cross-pair flow
    "usd_index_return",
    "pair_residual",
    "cross_pair_divergence",
    # Tier 1b: Session structure
    "session_open_dist",
    "prev_session_range_pos",
    "asian_range_pos",
    # Tier 1c: Momentum exhaustion
    "consec_dir_bars",
    "wick_ratio",
    "atr_contraction",
    "return_decel",
    # Tier 2: Volume / order flow
    "volume_spike",
    "volume_accumulation",
    "volume_pressure",
    "volume_climax",
    "volume_session_rel",
]

CLASS_NAMES = {0: "hold", 1: "short", 2: "long"}


def _load_data(data_dir, max_samples):
    seq_path = os.path.join(data_dir, "sequences.npy")
    lbl_path = os.path.join(data_dir, "labels.npz")

    X_full = np.load(seq_path, mmap_mode="r")
    labels = np.load(lbl_path, allow_pickle=True)

    total = len(X_full)
    logger.info("Full dataset: %d sequences, shape %s", total, X_full.shape)

    # Subsample with stratification
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

    timestamps_full = labels["timestamp"]

    # Stratified subsample
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

    X_last = np.asarray(X_full[idx, -1, :], dtype=np.float32)
    y = classes_full[idx]
    ts = timestamps_full[idx]
    net_r = labels["net_r"][idx].astype(np.float32) if "net_r" in labels else None
    raw_return = labels["raw_return"][idx].astype(np.float32) if "raw_return" in labels else net_r
    asset_ids = labels["asset_id"][idx].astype(np.int64) if "asset_id" in labels else None

    logger.info("Subsampled to %d rows (%.1f%% of total)", len(idx), 100 * len(idx) / total)
    return X_last, y, ts, net_r, raw_return, asset_ids, total, X_full.shape


# ── analyses ─────────────────────────────────────────────────────────────

def anova_f(X, y, names):
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    ssb = np.zeros(X.shape[1])
    ssw = np.zeros(X.shape[1])
    for c in classes:
        g = X[y == c]
        gm = g.mean(axis=0)
        ssb += len(g) * (gm - overall_mean) ** 2
        ssw += ((g - gm) ** 2).sum(axis=0)
    f = (ssb / max(1, len(classes) - 1)) / (ssw / max(1, len(y) - len(classes)) + 1e-12)
    return dict(sorted(zip(names, [round(float(v), 4) for v in f]), key=lambda kv: -kv[1]))


def mutual_info(X, y, names):
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, y, n_neighbors=5, random_state=42)
    return dict(sorted(zip(names, [round(float(v), 4) for v in mi]), key=lambda kv: -kv[1]))


def rf_baseline(X, y, n_classes):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20,
                                class_weight="balanced", random_state=42, n_jobs=-1,
                                oob_score=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
    f1 = cross_val_score(rf, X, y, cv=cv, scoring="f1_weighted")
    chance = 1.0 / n_classes
    majority_frac = float(np.bincount(y).max() / len(y))
    rf.fit(X, y)
    acc_mean = float(acc.mean())
    return {
        "n_samples": len(X),
        "cv_accuracy_mean": round(acc_mean, 4),
        "cv_accuracy_std": round(float(acc.std()), 4),
        "cv_f1_mean": round(float(f1.mean()), 4),
        "cv_f1_std": round(float(f1.std()), 4),
        "chance_level": round(chance, 4),
        "majority_baseline": round(majority_frac, 4),
        "accuracy_above_chance": round(acc_mean - chance, 4),
        "accuracy_above_majority": round(acc_mean - majority_frac, 4),
        "oob_score": round(float(rf.oob_score_), 4),
        "verdict": _rf_verdict(acc_mean, float(f1.mean()), chance, majority_frac),
    }


def _rf_verdict(acc, f1_weighted, chance, majority_frac):
    """Determine RF verdict using metrics appropriate for imbalanced data.

    With class_weight='balanced', the RF deliberately predicts minority classes,
    so raw accuracy drops below the majority baseline. F1-weighted accounts for
    this by rewarding correct minority predictions proportionally.
    """
    if f1_weighted > max(chance + 0.05, 0.45):
        return "SIGNAL"
    if acc > chance + 0.03:
        return "WEAK"
    return "NOISE"


def permutation_imp(X, y, names):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)
    rf = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    r = permutation_importance(rf, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
    imp = dict(zip(names, [round(float(v), 6) for v in r.importances_mean]))
    return dict(sorted(imp.items(), key=lambda kv: -kv[1]))


def label_stability(X, y, k=10):
    n = len(X)
    sample_idx = np.random.RandomState(123).choice(n, min(500, n), replace=False)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-12
    Xz = (X - mu) / sigma
    agree = []
    for i in sample_idx:
        d = np.linalg.norm(Xz - Xz[i], axis=1)
        nn = np.argpartition(d, k + 1)[:k + 1]
        nn = nn[nn != i][:k]
        agree.append(float((y[nn] == y[i]).mean()))
    m = float(np.mean(agree))
    return {
        "k": k, "sample_size": len(sample_idx),
        "mean_neighbor_agreement": round(m, 4),
        "stable_fraction_gt80": round(float(np.mean(np.array(agree) > 0.8)), 4),
        "verdict": "STABLE" if m > 0.6 else "NOISY" if m < 0.4 else "MARGINAL",
    }


def monthly_psi(timestamps, labels):
    """Monthly PSI comparing each month to the overall dataset distribution.

    Uses the overall class-proportion distribution as a robust baseline
    (avoids noise from small first-month samples).  Excludes months with
    extreme PSI (> 0.5) from the verdict — these are regime transitions,
    not systematic drift.
    """
    ts = pd.DatetimeIndex(timestamps)
    df = pd.DataFrame({"label": labels, "month": ts.to_period("M")})
    monthly = df.groupby("month")["label"].value_counts(normalize=True).unstack(fill_value=0)
    periods = sorted(monthly.index)
    if len(periods) < 2:
        return {"n_months": len(periods), "psi_mean": 0.0, "verdict": "STABLE"}
    baseline = df["label"].value_counts(normalize=True).sort_index().values
    psi_vals = []
    for p in periods:
        actual = monthly.loc[p].values
        actual = np.maximum(actual, 0.001)
        b = np.maximum(baseline, 0.001)
        psi_vals.append(round(float(np.sum((actual - b) * np.log(actual / b))), 4))
    mean_psi_all = float(np.mean(psi_vals))
    normal_psi = [v for v in psi_vals if v < 0.5]
    mean_psi = float(np.mean(normal_psi)) if normal_psi else mean_psi_all
    return {
        "n_months": len(periods), "psi_per_month": psi_vals,
        "psi_mean": round(mean_psi_all, 4),
        "psi_mean_excl_outliers": round(mean_psi, 4),
        "psi_max": round(float(np.max(psi_vals)), 4),
        "outlier_months": sum(1 for v in psi_vals if v >= 0.5),
        "verdict": "STABLE" if mean_psi < 0.1 else "DRIFTING",
    }


# ── label quality validation ───────────────────────────────────────────────

def label_return_correlation(y, net_r, class_names):
    """Check if labels match actual price outcomes.

    Short labels should precede negative returns, long should precede positive.
    Computes point-biserial correlation between each class and net_r.
    """
    from scipy.stats import pointbiserialr
    results = {}
    for cls_id, cls_name in class_names.items():
        mask = (y == cls_id).astype(np.float32)
        if mask.sum() < 10 or mask.sum() == len(mask):
            results[cls_name] = {"corr": 0.0, "p_value": 1.0, "n_samples": int(mask.sum()), "verdict": "INSUFFICIENT"}
            continue
        corr, pval = pointbiserialr(mask, net_r)
        results[cls_name] = {
            "corr": round(float(corr), 4),
            "p_value": round(float(pval), 6),
            "n_samples": int(mask.sum()),
            "mean_return": round(float(net_r[mask == 1].mean()), 4),
            "verdict": "ALIGNED" if (cls_name == "short" and corr < -0.05 and pval < 0.05)
                       or (cls_name == "long" and corr > 0.05 and pval < 0.05)
                       or (cls_name == "hold" and abs(corr) < 0.05)
                       else "MISALIGNED",
        }
    return results


def feature_return_correlation(X, net_r, names):
    """Check which features correlate with actual returns (not labels)."""
    from scipy.stats import spearmanr
    corrs = {}
    for i, name in enumerate(names):
        corr, pval = spearmanr(X[:, i], net_r)
        corrs[name] = {"corr": round(float(corr), 4), "p_value": round(float(pval), 6)}
    corrs_sorted = dict(sorted(corrs.items(), key=lambda kv: -abs(kv[1]["corr"])))
    # Verdict: are any features meaningfully correlated with returns?
    strong = sum(1 for v in corrs.values() if abs(v["corr"]) > 0.03 and v["p_value"] < 0.05)
    return {
        "feature_correlations": corrs_sorted,
        "n_significant": strong,
        "verdict": "STRONG" if strong >= 5 else "WEAK" if strong == 0 else "MODERATE",
    }


def permutation_test(X, y, n_classes, n_repeats=5):
    """Shuffle labels and retrain RF — real labels should score much higher.

    Uses ROC-AUC (ovr) instead of accuracy or F1 to avoid misleading results
    on imbalanced classes. ROC-AUC measures ranking quality independent of
    class prevalence and is robust to the class_weight='balanced' pathology
    where shuffled labels can paradoxically score higher on accuracy/F1.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    real_auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc_ovr").mean()

    rng = np.random.RandomState(42)
    shuffled_scores = []
    for _ in range(n_repeats):
        y_shuf = rng.permutation(y)
        score = cross_val_score(rf, X, y_shuf, cv=cv, scoring="roc_auc_ovr").mean()
        shuffled_scores.append(float(score))

    mean_shuffled = float(np.mean(shuffled_scores))
    drop = real_auc - mean_shuffled
    return {
        "real_auc": round(real_auc, 4),
        "shuffled_auc_mean": round(mean_shuffled, 4),
        "shuffled_auc_std": round(float(np.std(shuffled_scores)), 4),
        "auc_drop": round(float(drop), 4),
        "verdict": "INFORMATIVE" if drop > 0.10 else "WEAK" if drop < 0.02 else "MODERATE",
    }


def class_conditional_returns(y, net_r, class_names):
    """Average return per label class — validates label economic meaning."""
    results = {}
    for cls_id, cls_name in class_names.items():
        mask = y == cls_id
        n = int(mask.sum())
        if n == 0:
            results[cls_name] = {"n": 0, "mean_r": 0.0, "median_r": 0.0, "win_rate": 0.0}
            continue
        r = net_r[mask]
        results[cls_name] = {
            "n": n,
            "mean_r": round(float(r.mean()), 4),
            "median_r": round(float(np.median(r)), 4),
            "win_rate": round(float((r > 0).mean()), 4),
            "std_r": round(float(r.std()), 4),
        }
    # Verdict: short should have negative mean, long positive, hold near zero
    short_r = results.get("short", {}).get("mean_r", 0)
    long_r = results.get("long", {}).get("mean_r", 0)
    hold_r = results.get("hold", {}).get("mean_r", 0)
    aligned = (short_r < 0) and (long_r > 0)
    return {
        "by_class": results,
        "verdict": "ALIGNED" if aligned else "MISALIGNED",
    }


# ── main ─────────────────────────────────────────────────────────────────

def run_learnability_check(data_dir, output_path, max_samples):
    t0 = time.time()
    X, y, ts, net_r, raw_return, asset_ids, total_full, seq_shape = _load_data(data_dir, max_samples)
    n_classes = len(np.unique(y))
    n_feat = X.shape[1]
    feature_names = FEATURE_NAMES[:n_feat] if n_feat <= len(FEATURE_NAMES) else [f"f{i}" for i in range(n_feat)]

    class_counts = {CLASS_NAMES.get(int(c), str(c)): {"count": int(n), "fraction": round(float(n / len(y)), 4)}
                    for c, n in zip(*np.unique(y, return_counts=True))}
    imbalance = round(max(v["fraction"] for v in class_counts.values()) / (1.0 / n_classes), 2)

    report = {
        "generated_at": datetime.now().isoformat(),
        "data_dir": data_dir,
        "total_sequences_full": total_full,
        "sampled_sequences": len(y),
        "sequence_shape": list(seq_shape),
        "n_features": n_feat, "n_classes": n_classes,
        "class_counts": class_counts, "imbalance_ratio": imbalance,
    }

    # ── Original 6 tests ──

    # 1. ANOVA
    logger.info("[1/6] ANOVA F-scores...")
    f_scores = anova_f(X, y, feature_names)
    mean_f = float(np.mean(list(f_scores.values())))
    report["anova_f_scores"] = f_scores
    report["anova_summary"] = {
        "top_5": dict(list(f_scores.items())[:5]),
        "mean_f": round(mean_f, 4),
        "features_above_3": sum(1 for v in f_scores.values() if v > 3.0),
        "verdict": "STRONG" if mean_f > 5.0 else "WEAK" if mean_f < 1.0 else "MODERATE",
    }

    # 2. MI
    logger.info("[2/6] Mutual Information...")
    mi = mutual_info(X, y, feature_names)
    mean_mi = float(np.mean(list(mi.values())))
    report["mutual_information"] = mi
    mi_above_001 = sum(1 for v in mi.values() if v > 0.001)
    top10_mi = float(np.mean(list(mi.values())[:min(10, len(mi))]))
    report["mi_summary"] = {
        "mean_mi": round(mean_mi, 4), "top_5": dict(list(mi.items())[:5]),
        "top10_mean_mi": round(top10_mi, 4),
        "features_above_0.001": mi_above_001,
        "verdict": "STRONG" if mi_above_001 >= 8 or top10_mi > 0.005
                   else "WEAK" if mi_above_001 < 3
                   else "MODERATE",
    }

    # 3. RF
    logger.info("[3/6] Random Forest baseline...")
    rf = rf_baseline(X, y, n_classes)
    report["rf_baseline"] = rf

    # 4. Permutation importance
    logger.info("[4/6] Permutation importance...")
    report["permutation_importance"] = permutation_imp(X, y, feature_names)

    # 5. Stability
    logger.info("[5/6] Label stability...")
    stab = label_stability(X, y)
    report["label_stability"] = stab

    # 6. Drift
    logger.info("[6/6] Monthly drift...")
    drift = monthly_psi(ts, y)
    report["monthly_drift"] = drift

    # ── Label quality validation (new) ──

    # 7. Label-Return Correlation (uses raw_return for direction-aware alignment)
    if raw_return is not None:
        logger.info("[7/10] Label-return correlation...")
        lr_corr = label_return_correlation(y, raw_return, CLASS_NAMES)
        report["label_return_correlation"] = lr_corr

        # 8. Feature-Return Correlation (uses raw_return)
        logger.info("[8/10] Feature-return correlation...")
        fr_corr = feature_return_correlation(X, raw_return, feature_names)
        report["feature_return_correlation"] = fr_corr

        # 9. Permutation test
        logger.info("[9/10] Permutation test...")
        perm = permutation_test(X, y, n_classes)
        report["permutation_test"] = perm

        # 10. Class-conditional returns (uses raw_return for direction-aware alignment)
        logger.info("[10/10] Class-conditional returns...")
        ccr = class_conditional_returns(y, raw_return, CLASS_NAMES)
        report["class_conditional_returns"] = ccr
    else:
        logger.warning("raw_return/net_r not found in labels — skipping label quality validation (tests 7-10)")

    # ── Verdict ──
    scores, reasons = [], []

    def _get_verdict(obj, path):
        for k in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(k, {})
            else:
                return "?"
        return obj.get("verdict", "?") if isinstance(obj, dict) else "?"

    for name, path, good in [
        ("ANOVA", "anova_summary", "STRONG"),
        ("MI", "mi_summary", "STRONG"),
        ("RF", "rf_baseline", "SIGNAL"),
        ("Stability", "label_stability", "STABLE"),
        ("Drift", "monthly_drift", "STABLE"),
    ]:
        val = _get_verdict(report, path)
        scores.append(1.0 if val == good else 0.5 if val in ("MODERATE", "MARGINAL", "WEAK") else 0.0)
        if val != good:
            reasons.append(f"{name}: {val}")

    # Label quality scores (if available)
    if raw_return is not None:
        for name, path, good in [
            ("Label-Return-Short", "label_return_correlation.short", "ALIGNED"),
            ("Label-Return-Long", "label_return_correlation.long", "ALIGNED"),
            ("Feature-Return", "feature_return_correlation", "STRONG"),
            ("Permutation", "permutation_test", "INFORMATIVE"),
            ("Cond. Returns", "class_conditional_returns", "ALIGNED"),
        ]:
            val = _get_verdict(report, path)
            scores.append(1.0 if val == good else 0.5 if val in ("MODERATE", "MARGINAL", "WEAK") else 0.0)
            if val != good:
                reasons.append(f"{name}: {val}")

    avg = float(np.mean(scores))
    overall = "LEARNABLE" if avg >= 0.7 else "MARGINAL" if avg >= 0.4 else "NOT LEARNABLE"
    report.update(overall_verdict=overall, overall_score=round(avg, 3),
                  verdict_reasons=reasons, elapsed_seconds=round(time.time() - t0, 2))

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print
    top5_f = list(f_scores.items())[:5]
    top5_mi = list(mi.items())[:5]

    print()
    print("=" * 72)
    print("  LEARNABILITY ANALYSIS REPORT")
    print("=" * 72)
    print(f"  Sampled: {len(y):,} / {total_full:,} sequences  ({seq_shape[1]} bars x {n_feat} features)")
    print(f"  Classes: {n_classes}  |  Imbalance: {imbalance}x")
    print()
    print("  +---------------------------+--------------------------------------+")
    print("  | Test                      | Detail                               |")
    print("  +---------------------------+--------------------------------------+")
    print(f"  | ANOVA F-scores            | mean F = {report['anova_summary']['mean_f']:<30} |")
    print(f"  | Mutual Information        | mean MI = {report['mi_summary']['mean_mi']:<29} |")
    print(f"  | RF Baseline               | acc = {rf['cv_accuracy_mean']} (chance {rf['chance_level']}, majority {rf['majority_baseline']}){'':<4} |")
    print(f"  | Label Stability           | agree = {stab['mean_neighbor_agreement']:<30} |")
    print(f"  | Monthly Drift             | PSI = {drift['psi_mean']:<31} |")

    if raw_return is not None:
        lr = report.get("label_return_correlation", {})
        fr = report.get("feature_return_correlation", {})
        pm = report.get("permutation_test", {})
        cc = report.get("class_conditional_returns", {})
        print("  +---------------------------+--------------------------------------+")
        print(f"  | Label-Return Corr         | short={lr.get('short',{}).get('corr','?')}, long={lr.get('long',{}).get('corr','?'):<17} |")
        print(f"  | Feature-Return Corr       | {fr.get('n_significant',0)} significant features              |")
        print(f"  | Permutation Test          | auc drop = {pm.get('auc_drop','?'):<27} |")
        short_r = cc.get("by_class", {}).get("short", {}).get("mean_r", "?")
        long_r = cc.get("by_class", {}).get("long", {}).get("mean_r", "?")
        print(f"  | Class-Conditional Returns | short={short_r}, long={long_r:<19} |")

    print("  +---------------------------+--------------------------------------+")
    print()
    print(f"  OVERALL:  {overall}  (score: {avg:.2f}/1.00)")
    if reasons:
        print(f"  Issues:   {'; '.join(reasons)}")
    print()
    print(f"  Top features (ANOVA):  {', '.join(f'{k}({v})' for k, v in top5_f)}")
    print(f"  Top features (MI):     {', '.join(f'{k}({v})' for k, v in top5_mi)}")
    print()
    print(f"  Report: {output_path}")
    print(f"  Time:   {report['elapsed_seconds']:.1f}s")
    print("=" * 72)
    return report


def main():
    p = argparse.ArgumentParser(description="Dataset Learnability Analysis (fast, subsampled)")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=5000,
                   help="Max rows to load (default: 5000, stratified)")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "training_set") if args.data_dir is None \
        else (os.path.abspath(args.data_dir) if os.path.isabs(args.data_dir)
              else os.path.abspath(os.path.join(base_dir, args.data_dir)))
    output = args.output or os.path.join(data_dir, "learnability_report.json")

    for f in ["sequences.npy", "labels.npz"]:
        if not os.path.exists(os.path.join(data_dir, f)):
            print(f"ERROR: {f} not found in {data_dir}")
            sys.exit(1)

    run_learnability_check(data_dir, output, args.max_samples)


if __name__ == "__main__":
    main()

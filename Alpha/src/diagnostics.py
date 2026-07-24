"""Training diagnostics for the Alpha pipeline.

Collects everything needed to debug a bad training run:
- Dataset stats: class balance per asset / per month, valid ratios, feature-label F-scores.
- Training curves: train/val loss, LR, gradient norms, per-class val metrics,
  confidence histograms per epoch.
- Holdout report: confusion matrix, per-class precision/recall vs base rate,
  accuracy by confidence bucket (calibration check).

Everything is written as small JSON/CSV files into a per-run directory, which is
zipped at the end of the pipeline (target: < 10 MB, actual size is typically KBs).
"""

import csv
import json
import logging
import os
import zipfile
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

MAX_ZIP_MB = 10.0
CLASS_NAMES = ["hold", "short", "long"]  # indices 0, 1, 2


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, np.datetime64)):
            return str(obj)
        return super().default(obj)


def compute_class_metrics(targets: np.ndarray, preds: np.ndarray, n_classes: int = 3) -> dict:
    """Confusion matrix + per-class precision/recall/F1/base-rate/edge."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    total = max(1, int(cm.sum()))

    per_class = {}
    for c in range(n_classes):
        tp = int(cm[c, c])
        precision = tp / max(1, int(cm[:, c].sum()))
        recall = tp / max(1, int(cm[c].sum()))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        base_rate = float(cm[c].sum()) / total
        per_class[CLASS_NAMES[c]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "base_rate": round(base_rate, 4),
            "edge": round(precision - base_rate, 4),  # precision above chance
            "support": int(cm[c].sum()),
            "predicted": int(cm[:, c].sum()),
        }

    return {
        "confusion_matrix": cm.tolist(),
        "accuracy": round(float(np.trace(cm)) / total, 4),
        "per_class": per_class,
    }


def confidence_histogram(max_probs: np.ndarray, bins: int = 10) -> dict:
    """Distribution of the model's top softmax probability. All-mass-near-0.33 = guessing."""
    if len(max_probs) == 0:
        return {"bin_edges": [], "counts": [], "mean_confidence": 0.0}
    hist, edges = np.histogram(max_probs, bins=bins, range=(0.0, 1.0))
    return {
        "bin_edges": [round(float(e), 3) for e in edges],
        "counts": hist.tolist(),
        "mean_confidence": round(float(np.mean(max_probs)), 4),
        "p90_confidence": round(float(np.percentile(max_probs, 90)), 4),
    }


def confidence_bucket_table(max_probs: np.ndarray, correct: np.ndarray,
                            edges=(0.33, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01)) -> list:
    """Accuracy per confidence bucket. Should be monotonically increasing if calibrated."""
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (max_probs >= lo) & (max_probs < hi)
        n = int(mask.sum())
        table.append({
            "bucket": f"[{lo:.2f}, {hi:.2f})",
            "n": n,
            "accuracy": round(float(correct[mask].mean()), 4) if n > 0 else None,
        })
    return table


def feature_label_scores(features_last_bar: np.ndarray, labels: np.ndarray,
                         feature_names: list) -> dict:
    """One-way ANOVA F-score per feature (between-class vs within-class variance).

    If no feature scores meaningfully above ~1.0, the target is not learnable
    from the current feature set — fix features, not the model.
    """
    classes = np.unique(labels)
    n = len(labels)
    if n == 0 or len(classes) < 2:
        return {}

    overall_mean = features_last_bar.mean(axis=0)
    ssb = np.zeros(features_last_bar.shape[1], dtype=np.float64)
    ssw = np.zeros(features_last_bar.shape[1], dtype=np.float64)
    for c in classes:
        grp = features_last_bar[labels == c]
        grp_mean = grp.mean(axis=0)
        ssb += len(grp) * (grp_mean - overall_mean) ** 2
        ssw += ((grp - grp_mean) ** 2).sum(axis=0)

    dfb = max(1, len(classes) - 1)
    dfw = max(1, n - len(classes))
    f_scores = (ssb / dfb) / (ssw / dfw + 1e-12)

    scores = {name: round(float(v), 4) for name, v in zip(feature_names, f_scores)}
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def _flatten(d: dict, prefix: str = "") -> dict:
    """Flattens nested dicts into scalar columns for the CSV epoch log."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten(v, f"{key}_"))
        elif isinstance(v, (int, float, str, bool)) or v is None:
            flat[key] = v
    return flat


class DiagnosticsRecorder:
    """Accumulates diagnostics and writes them to a per-run directory."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.report = {
            "created_at": datetime.now().isoformat(),
            "config": {},
            "dataset": {},
            "training": {"epochs": []},
            "holdout": {},
        }

    def set_config(self, **cfg):
        self.report["config"].update(cfg)

    def set_dataset_stats(self, stats: dict):
        self.report["dataset"] = stats

    def log_epoch(self, **row):
        self.report["training"]["epochs"].append(row)

    def set_holdout(self, holdout: dict):
        self.report["holdout"] = holdout

    def save(self) -> str:
        json_path = os.path.join(self.run_dir, "diagnostics.json")
        with open(json_path, "w") as f:
            json.dump(self.report, f, indent=2, cls=NumpyJSONEncoder)

        # Flat CSV of epoch curves for quick plotting in Excel/pandas
        epochs = self.report["training"]["epochs"]
        if epochs:
            rows = [_flatten(e) for e in epochs]
            all_keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "epoch", k))
            csv_path = os.path.join(self.run_dir, "epoch_curves.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Diagnostics written to {self.run_dir}")
        return json_path


def zip_run(run_dir: str, extra_files=None, max_mb: float = MAX_ZIP_MB) -> tuple:
    """Zips the run directory (plus optional extra files) into a single archive.

    Returns (zip_path, size_mb). Warns if the archive exceeds max_mb.
    """
    run_dir = os.path.abspath(run_dir)
    zip_path = run_dir.rstrip("\\/") + ".zip"
    base = os.path.basename(run_dir)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for root, _, files in os.walk(run_dir):
            for fn in files:
                full = os.path.join(root, fn)
                arcname = os.path.join(base, os.path.relpath(full, run_dir))
                zf.write(full, arcname)
        for extra in (extra_files or []):
            if extra and os.path.exists(extra):
                zf.write(extra, os.path.join(base, os.path.basename(extra)))

    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    if size_mb > max_mb:
        logger.warning(
            f"Diagnostics archive {zip_path} is {size_mb:.2f} MB (> {max_mb} MB limit). "
            f"Consider excluding large log files."
        )
    else:
        logger.info(f"Diagnostics archive created: {zip_path} ({size_mb:.2f} MB, limit {max_mb} MB)")
    return zip_path, size_mb

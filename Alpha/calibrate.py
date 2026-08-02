import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.run_pipeline import _date_based_split_indices
from Alpha.src.calibration import (
    apply_temperature,
    brier_score,
    collect_action_logits,
    expected_calibration_error,
    fit_temperature,
    reliability_table,
    save_calibration,
    _softmax,
)
from Alpha.src.model import AlphaSLModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _binary_cross_entropy(logits, targets, temperature):
    scaled = logits / max(float(temperature), 1e-6)
    probs = np.clip(_sigmoid(scaled), 1e-6, 1.0 - 1e-6)
    n = len(targets)
    ce = 0.0
    for i in range(n):
        t = float(targets[i])
        ce -= t * np.log(probs[i]) + (1 - t) * np.log(1 - probs[i])
    return float(ce / n)


def fit_temperature_binary(logits, targets, candidates=None):
    logits = np.asarray(logits, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    if candidates is None:
        candidates = np.linspace(0.5, 5.0, 91)
    losses = np.array([_binary_cross_entropy(logits, targets, t) for t in candidates])
    return float(candidates[int(np.argmin(losses))])


def main():
    parser = argparse.ArgumentParser(description="Fit Alpha calibration on the validation split.")
    parser.add_argument("--model-path", type=str, default="Alpha/models/alpha_model.pth")
    parser.add_argument("--dataset-dir", type=str, default="Alpha/data/training_set")
    parser.add_argument("--output", type=str, default="Alpha/models/alpha_calibration.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    dataset_dir = PROJECT_ROOT / args.dataset_dir
    sequences_path = dataset_dir / "sequences.npy"
    labels_path = dataset_dir / "labels.npz"
    model_path = PROJECT_ROOT / args.model_path
    output_path = PROJECT_ROOT / args.output

    sequences = np.load(sequences_path, mmap_mode="r")
    labels = np.load(labels_path)
    if "action_classes" not in labels or "asset_id" not in labels:
        raise RuntimeError("labels.npz does not contain action classes. Regenerate the dataset.")
    action_classes = labels["action_classes"].astype(np.int64)
    asset_ids = labels["asset_id"].astype(np.int64)
    timestamps = labels["timestamp"]
    _, val_idx, _ = _date_based_split_indices(timestamps)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    fmt_ver = checkpoint.get("format_version")
    model = AlphaSLModel(**checkpoint["model_config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits = collect_action_logits(model, sequences, asset_ids, val_idx, DEVICE, batch_size=args.batch_size)
    targets = action_classes[val_idx]

    if fmt_ver == 5:
        # V5: binary single-logit — sigmoid calibration
        logits_flat = logits[:, 0] if logits.ndim > 1 else logits.ravel()
        temperature = fit_temperature_binary(logits_flat, targets)
        scaled = _sigmoid(logits_flat / max(float(temperature), 1e-6))
        calibrated_probs = np.column_stack([1.0 - scaled, scaled])
        uncalibrated_probs = np.column_stack([1.0 - _sigmoid(logits_flat), _sigmoid(logits_flat)])
        save_calibration(output_path, temperature=temperature, threshold=0.5)
    elif fmt_ver in (3, 4):
        # V3/V4: 3-class softmax calibration
        temperature = fit_temperature(logits, targets)
        calibrated_probs = apply_temperature(logits, temperature)
        uncalibrated_probs = _softmax(logits)
        save_calibration(output_path, temperature=temperature, threshold=0.5)
    else:
        raise RuntimeError(f"Unsupported format_version: {fmt_ver}")

    # For binary, brier/ECE use 2-class probs; for 3-class, use multi-class
    if fmt_ver == 5:
        binary_targets = targets.copy()
        report = {
            "validation_samples": int(len(val_idx)),
            "format_version": fmt_ver,
            "temperature": round(float(temperature), 4),
            "uncalibrated_brier": round(float(np.mean((uncalibrated_probs[np.arange(len(binary_targets)), binary_targets] - 1.0) ** 2)), 6),
            "calibrated_brier": round(float(np.mean((calibrated_probs[np.arange(len(binary_targets)), binary_targets] - 1.0) ** 2)), 6),
        }
    else:
        report = {
            "validation_samples": int(len(val_idx)),
            "format_version": fmt_ver,
            "temperature": round(float(temperature), 4),
            "uncalibrated_brier": brier_score(uncalibrated_probs, targets),
            "calibrated_brier": brier_score(calibrated_probs, targets),
            "uncalibrated_ece": expected_calibration_error(uncalibrated_probs, targets),
            "calibrated_ece": expected_calibration_error(calibrated_probs, targets),
        }

    report_path = output_path.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Calibration saved to {output_path}")


if __name__ == "__main__":
    main()

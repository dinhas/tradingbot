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
)
from Alpha.src.model import AlphaSLModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Fit Alpha tradeability calibration on the validation split.")
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
    if "action_targets" not in labels or "asset_id" not in labels:
        raise RuntimeError("labels.npz does not contain V3 action targets. Regenerate the dataset.")
    action_targets = labels["action_targets"].astype(np.float32)
    asset_ids = labels["asset_id"].astype(np.int64)
    timestamps = labels["timestamp"]
    _, val_idx, _ = _date_based_split_indices(timestamps)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if checkpoint.get("format_version") != 3:
        raise RuntimeError("Expected a V3 action-head checkpoint.")
    model = AlphaSLModel(**checkpoint["model_config"]).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logits = collect_action_logits(model, sequences, asset_ids, val_idx, DEVICE, batch_size=args.batch_size)
    targets = action_targets[val_idx]
    temperatures = [fit_temperature(logits[:, side], targets[:, side]) for side in range(2)]
    calibrated = np.column_stack([
        apply_temperature(logits[:, side], temperatures[side]) for side in range(2)
    ])
    uncalibrated = 1.0 / (1.0 + np.exp(-logits))

    save_calibration(output_path, action_temperatures=temperatures, threshold=0.5)
    report = {
        "validation_samples": int(len(val_idx)),
        "action_temperatures": [round(float(t), 4) for t in temperatures],
        "uncalibrated_brier": [brier_score(uncalibrated[:, side], targets[:, side]) for side in range(2)],
        "calibrated_brier": [brier_score(calibrated[:, side], targets[:, side]) for side in range(2)],
        "uncalibrated_ece": [expected_calibration_error(uncalibrated[:, side], targets[:, side]) for side in range(2)],
        "calibrated_ece": [expected_calibration_error(calibrated[:, side], targets[:, side]) for side in range(2)],
        "calibrated_reliability": {
            "short": reliability_table(calibrated[:, 0], targets[:, 0]),
            "long": reliability_table(calibrated[:, 1], targets[:, 1]),
        },
    }
    report_path = output_path.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Calibration saved to {output_path}")


if __name__ == "__main__":
    main()

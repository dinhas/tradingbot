import json
from pathlib import Path

import numpy as np
import torch


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _bce(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    scaled = logits / max(float(temperature), 1e-6)
    probs = np.clip(_sigmoid(scaled), 1e-6, 1.0 - 1e-6)
    return float(-(targets * np.log(probs) + (1.0 - targets) * np.log(1.0 - probs)).mean())


def fit_temperature(logits: np.ndarray, targets: np.ndarray, candidates: np.ndarray | None = None) -> float:
    """Fits a scalar temperature for binary tradeability logits on validation data."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if candidates is None:
        candidates = np.linspace(0.5, 5.0, 91)
    losses = np.array([_bce(logits, targets, t) for t in candidates])
    return float(candidates[int(np.argmin(losses))])


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return _sigmoid(np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)).astype(np.float32)


def reliability_table(probs: np.ndarray, targets: np.ndarray, bins: int = 10) -> list[dict]:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        n = int(mask.sum())
        rows.append({
            "bucket": f"[{lo:.2f}, {hi:.2f}]" if hi == 1.0 else f"[{lo:.2f}, {hi:.2f})",
            "n": n,
            "mean_probability": round(float(probs[mask].mean()), 4) if n else None,
            "observed_rate": round(float(targets[mask].mean()), 4) if n else None,
        })
    return rows


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return round(float(np.mean((probs - targets) ** 2)), 6)


def expected_calibration_error(probs: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not mask.any():
            continue
        ece += (mask.mean()) * abs(float(probs[mask].mean()) - float(targets[mask].mean()))
    return round(float(ece), 6)


def save_calibration(path: str | Path, temperature: float | None = None, threshold: float = 0.5,
                     action_temperatures: list[float] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        payload = {"action_threshold": float(threshold)}
        if action_temperatures is not None:
            payload["action_temperatures"] = [float(v) for v in action_temperatures]
        elif temperature is not None:
            payload.update({"trade_temperature": float(temperature), "trade_threshold": float(threshold)})
        else:
            raise ValueError("Provide temperature or action_temperatures.")
        json.dump(payload, f, indent=2)
    return path


def load_calibration(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_action_logits(model, sequences, asset_ids, indices, device,
                          batch_size: int = 512) -> np.ndarray:
    logits = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            batch = np.asarray(sequences[indices[i:i + batch_size]], dtype=np.float32)
            batch_assets = torch.from_numpy(asset_ids[indices[i:i + batch_size]].astype(np.int64)).to(device)
            outputs = model(torch.from_numpy(batch).to(device), batch_assets, return_dict=True)
            logits.append(outputs["action_logits"].float().cpu().numpy())
    return np.concatenate(logits)

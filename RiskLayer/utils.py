"""Utility helpers for the RiskLayer LSTM training pipeline."""

from __future__ import annotations

import numpy as np


def floor_to_increment(values: np.ndarray | float, increment: float) -> np.ndarray | float:
    """Floor values down to the nearest fixed increment.

    Parameters
    ----------
    values:
        Scalar or numpy array to floor.
    increment:
        Positive increment step (e.g. 0.05 or 0.1).
    """
    if increment <= 0:
        raise ValueError("increment must be > 0")

    arr = np.asarray(values, dtype=np.float64)
    floored = np.floor(arr / increment) * increment
    floored = np.round(floored, 10)
    if np.isscalar(values):
        return float(floored)
    return floored


def sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    arr = np.asarray(values, dtype=np.float64)
    clipped = np.clip(arr, -60.0, 60.0)
    out = 1.0 / (1.0 + np.exp(-clipped))
    if np.isscalar(values):
        return float(out)
    return out


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Average True Range (ATR) using simple moving average of true range."""
    if period < 1:
        raise ValueError("period must be >= 1")

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)

    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, close arrays must have the same length")

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])

    atr = np.empty_like(tr)
    atr[:period] = np.mean(tr[:period])

    cumsum = np.cumsum(tr, dtype=np.float64)
    atr[period:] = (cumsum[period:] - cumsum[:-period]) / period
    return atr.astype(np.float32)

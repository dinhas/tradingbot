"""Forward label simulation for RiskLayer training.

Generates SL ATR target, TP ATR target, and trade quality target using
strict first-touch sequential simulation rules.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
from dataclasses import dataclass

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RiskLayer.utils import floor_to_increment, sigmoid
from shared_constants import DEFAULT_SPREADS

LOGGER = logging.getLogger(__name__)

SL_MIN = 1.0
SL_MAX = 3.5
TP_MIN = 2.0
TP_MAX = 6.0


@dataclass
class SimulationResult:
    sl_atr: float
    tp_atr: float
    quality: float
    outcome: int  # -1: SL first, 1: TP first, 0: neither
    event_candle: int


def _first_touch_simulation(
    entry: float,
    atr: float,
    side: int,
    future_high: np.ndarray,
    future_low: np.ndarray,
    spread: float = 0.0
) -> tuple[int, int]:
    """Return first-touch outcome and candle index with spread simulation."""
    if side not in (-1, 1):
        return 0, -1

    # entry is the spread-adjusted execution price (Ask for BUY, Bid for SELL)
    if side == 1:  # BUY
        sl_level = entry - SL_MAX * atr
        tp_level = entry + TP_MAX * atr
    else:  # SELL
        sl_level = entry + SL_MAX * atr
        tp_level = entry - TP_MAX * atr

    for idx, (high, low) in enumerate(zip(future_high, future_low), start=1):
        if side == 1:
            # Long exit side is Bid
            bid_high = high - (spread / 2.0)
            bid_low = low - (spread / 2.0)
            sl_hit = bid_low <= sl_level
            tp_hit = bid_high >= tp_level
        else:
            # Short exit side is Ask
            ask_high = high + (spread / 2.0)
            ask_low = low + (spread / 2.0)
            sl_hit = ask_high >= sl_level
            tp_hit = ask_low <= tp_level

        # In same-candle ambiguity, choose SL first (conservative risk-first rule).
        if sl_hit and tp_hit:
            return -1, idx
        if sl_hit:
            return -1, idx
        if tp_hit:
            return 1, idx

    return 0, -1


def _excursions(entry: float, side: int, highs: np.ndarray, lows: np.ndarray, spread: float = 0.0) -> tuple[float, float]:
    """Compute favorable and adverse excursions in price units with spread."""
    if side == 1:  # BUY (entry = Ask)
        # Long exits at Bid
        bid_highs = highs - (spread / 2.0)
        bid_lows = lows - (spread / 2.0)
        favorable = float(np.max(bid_highs) - entry)
        adverse = float(entry - np.min(bid_lows))
    elif side == -1:  # SELL (entry = Bid)
        # Short exits at Ask
        ask_highs = highs + (spread / 2.0)
        ask_lows = lows + (spread / 2.0)
        favorable = float(entry - np.min(ask_lows))
        adverse = float(np.max(ask_highs) - entry)
    else:
        raise ValueError(f"Invalid side {side}. Expected 1 or -1.")

    return max(favorable, 0.0), max(adverse, 0.0)


def _raw_quality(outcome: int, event_candle: int) -> float:
    if outcome == 1 and event_candle > 0:
        return 1.0 / event_candle
    if outcome == -1 and event_candle > 0:
        return -1.0 / event_candle
    return 0.0


def generate_labels(
    labels_npz_path: str,
    output_path: str,
) -> str:
    """Generate targets from dataset metadata produced by RiskLayer/generate_dataset.py."""
    data = np.load(labels_npz_path, allow_pickle=True)

    required = ["action", "future_high_path", "future_low_path", "entry_price", "signal_atr", "asset_id", "signal_step", "asset_names"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {labels_npz_path}")

    actions = data["action"].astype(np.int8)
    high_paths = data["future_high_path"].astype(np.float32)
    low_paths = data["future_low_path"].astype(np.float32)
    asset_ids = data["asset_id"].astype(np.int32)
    signal_steps = data["signal_step"].astype(np.int32)
    asset_names = data["asset_names"]

    entries = data["entry_price"].astype(np.float32)
    atr_ref = data["signal_atr"].astype(np.float32)

    sl_targets = np.zeros(len(actions), dtype=np.float32)
    tp_targets = np.zeros(len(actions), dtype=np.float32)
    q_targets = np.zeros(len(actions), dtype=np.float32)
    outcomes = np.zeros(len(actions), dtype=np.int8)
    event_candles = np.full(len(actions), -1, dtype=np.int16)

    for i in range(len(actions)):
        side = int(actions[i])
        highs = high_paths[i]
        lows = low_paths[i]
        mid_entry = float(entries[i])
        atr = max(float(atr_ref[i]), 1e-8)

        # Retrieve asset name to find spread
        asset_name = asset_names[asset_ids[i]] if asset_ids[i] < len(asset_names) else "EURUSD"
        spread = DEFAULT_SPREADS.get(asset_name, 0.0)
        
        # Adjust entry for spread (Buy at Ask, Sell at Bid)
        entry_exec = mid_entry + (side * spread / 2.0)

        outcome, event_candle = _first_touch_simulation(entry_exec, atr, side, highs, lows, spread=spread)

        # Respect first touch: any future opposite move is ignored by slicing to event.
        if event_candle > 0:
            highs_slice = highs[:event_candle]
            lows_slice = lows[:event_candle]
        else:
            highs_slice = highs
            lows_slice = lows

        favorable_move, adverse_move = _excursions(entry_exec, side, highs_slice, lows_slice, spread=spread)

        tp_atr = favorable_move / atr
        sl_atr = adverse_move / atr

        tp_atr = float(np.clip(tp_atr, TP_MIN, TP_MAX))
        sl_atr = float(np.clip(sl_atr, SL_MIN, SL_MAX))

        # Floor-to-step policy: examples require down-stepping to 0.05 increments.
        tp_atr = float(floor_to_increment(tp_atr, 0.05))
        sl_atr = float(floor_to_increment(sl_atr, 0.05))

        raw_q = _raw_quality(outcome, event_candle)
        quality = float(sigmoid(raw_q))
        quality = float(floor_to_increment(quality, 0.1))

        sl_targets[i] = sl_atr
        tp_targets[i] = tp_atr
        q_targets[i] = quality
        outcomes[i] = outcome
        event_candles[i] = event_candle

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(
        output_path,
        sl_atr=sl_targets,
        tp_atr=tp_targets,
        quality=q_targets,
        outcome=outcomes,
        event_candle=event_candles,
        action=actions,
        asset_id=asset_ids,
        signal_step=signal_steps,
        asset_names=asset_names,
    )

    LOGGER.info("Generated %d labels (Spread-Aware) → %s", len(sl_targets), output_path)
    return output_path


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Generate robust first-touch labels for RiskLayer")
    parser.add_argument("--labels-npz", required=True, help="Path to labels.npz from generate_dataset.py")
    parser.add_argument("--output", default="RiskLayer/data/training_set/risk_targets.npz")
    args = parser.parse_args()

    generate_labels(labels_npz_path=args.labels_npz, output_path=args.output)


if __name__ == "__main__":
    main()

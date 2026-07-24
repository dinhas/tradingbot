from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TradeConfig:
    tp_mult: float = 1.0
    sl_mult: float = 0.5
    max_hold_bars: int = 6
    leverage: float = 100.0
    commission_rate_per_side: float = 0.00002


@dataclass(frozen=True)
class TradeOutcome:
    entry_idx: int
    exit_idx: int
    direction: int
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    gross_return: float
    net_return: float
    net_r: float
    reason: str


class TradeSimulator:
    """Single source of truth for Alpha labeling and backtest execution economics."""

    def __init__(self, config: TradeConfig | None = None):
        self.config = config or TradeConfig()

    @staticmethod
    def entry_price(mid_price: float, direction: int, spread: float) -> float:
        return float(mid_price + direction * spread / 2.0)

    @staticmethod
    def market_exit_price(mid_price: float, direction: int, spread: float) -> float:
        return float(mid_price - direction * spread / 2.0)

    def barriers(self, entry_price: float, direction: int, atr: float) -> tuple[float, float]:
        sl_dist = self.config.sl_mult * atr
        tp_dist = self.config.tp_mult * atr
        return (
            float(entry_price - direction * sl_dist),
            float(entry_price + direction * tp_dist),
        )

    @staticmethod
    def barrier_exit(high: float, low: float, direction: int, spread: float,
                     sl: float, tp: float) -> tuple[float | None, str | None]:
        """Resolve a bar conservatively: when both barriers touch, the stop wins."""
        half_spread = spread / 2.0
        if direction == 1:
            bid_low = low - half_spread
            bid_high = high - half_spread
            if bid_low <= sl:
                return float(sl), "SL"
            if bid_high >= tp:
                return float(tp), "TP"
        else:
            ask_high = high + half_spread
            ask_low = low + half_spread
            if ask_high >= sl:
                return float(sl), "SL"
            if ask_low <= tp:
                return float(tp), "TP"
        return None, None

    def returns(self, entry_price: float, exit_price: float, direction: int,
                atr: float) -> tuple[float, float, float]:
        gross_return = ((exit_price - entry_price) / entry_price) * direction * self.config.leverage
        fees_return = 2.0 * self.config.commission_rate_per_side
        net_return = gross_return - fees_return
        risk_return = (self.config.sl_mult * atr / entry_price) * self.config.leverage
        net_r = net_return / max(risk_return, 1e-12)
        return float(gross_return), float(net_return), float(net_r)

    def simulate(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray, signal_idx: int, direction: int,
                 atr: float, spread: float) -> TradeOutcome | None:
        """Enter at the next bar open and hold to SL, TP, or the vertical barrier."""
        entry_idx = signal_idx + 1
        final_idx = entry_idx + self.config.max_hold_bars - 1
        if direction not in (-1, 1) or entry_idx >= len(closes) or final_idx >= len(closes):
            return None
        if not np.isfinite(atr) or atr <= 0:
            return None

        entry_price = self.entry_price(opens[entry_idx], direction, spread)
        sl, tp = self.barriers(entry_price, direction, atr)
        exit_price = None
        exit_idx = final_idx
        reason = "Timeout"

        for idx in range(entry_idx, final_idx + 1):
            price, bar_reason = self.barrier_exit(
                highs[idx], lows[idx], direction, spread, sl, tp
            )
            if price is not None:
                exit_price = price
                reason = bar_reason
                exit_idx = idx
                break

        if exit_price is None:
            exit_price = self.market_exit_price(closes[final_idx], direction, spread)

        gross_return, net_return, net_r = self.returns(
            entry_price, exit_price, direction, atr
        )
        return TradeOutcome(
            entry_idx=entry_idx,
            exit_idx=exit_idx,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            sl=sl,
            tp=tp,
            gross_return=gross_return,
            net_return=net_return,
            net_r=net_r,
            reason=reason,
        )

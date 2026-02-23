import numpy as np
from typing import Dict, Optional, Tuple
from .config import config

class ExecutionEngine:
    def __init__(self):
        self.commission_per_lot = config.COMMISSION_PER_LOT
        self.slippage_atr_ratio = config.SLIPPAGE_ATR_RATIO

    def calculate_trade_outcome(self,
                               side: str,
                               entry_price: float,
                               sl_price: float,
                               tp_price: float,
                               atr: float,
                               spread: float,
                               window_ohlc: np.ndarray) -> Tuple[float, bool, int]:
        """
        Simulates trade resolution over a window of OHLC data.
        window_ohlc: array of [open, high, low, close]
        Returns: (exit_price, is_closed, bars_held)
        """
        # window_ohlc is (N, 4)
        for i in range(len(window_ohlc)):
            o, h, l, c = window_ohlc[i]

            # Add spread to high/low to get bid/ask
            # Long: SL/TP triggered by Bid. Bid = Price - Spread/2?
            # Wait, usually OHLC is Mid.
            # Long Exit: Close at Bid.
            # Short Exit: Close at Ask.

            half_spread = spread / 2.0

            if side == 'long':
                # Deterministic path
                path = self._get_path(o, h, l, c)
                for p in path:
                    bid = p - half_spread
                    if bid <= sl_price:
                        # Slippage on SL
                        exit_price = sl_price - self.slippage_atr_ratio * atr
                        return exit_price, True, i + 1
                    if bid >= tp_price:
                        exit_price = tp_price
                        return exit_price, True, i + 1
            else: # Short
                path = self._get_path(o, h, l, c)
                for p in path:
                    ask = p + half_spread
                    if ask >= sl_price:
                        # Slippage on SL
                        exit_price = sl_price + self.slippage_atr_ratio * atr
                        return exit_price, True, i + 1
                    if ask <= tp_price:
                        exit_price = tp_price
                        return exit_price, True, i + 1

        return window_ohlc[-1, 3], False, len(window_ohlc)

    def _get_path(self, o: float, h: float, l: float, c: float) -> Tuple[float, float, float, float]:
        if c >= o:
            return (o, l, h, c)
        else:
            return (o, h, l, c)

    def calculate_costs(self, volume_lots: float, entry_price: float, exit_price: float, asset: str) -> float:
        """Calculates total costs (commission + spread + slippage already in exit_price)."""
        # Commission is per lot round turn
        commission = volume_lots * self.commission_per_lot
        return commission

    def get_entry_price(self, mid_price: float, spread: float, side: str) -> float:
        if side == 'long':
            return mid_price + spread / 2.0 # Ask
        else:
            return mid_price - spread / 2.0 # Bid

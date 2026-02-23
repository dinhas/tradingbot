import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict
from .config import RiskConfig

@dataclass
class TradeResult:
    pnl: float
    exit_type: str  # "TP", "SL", "Time", "Drawdown"
    exit_price: float
    commission: float
    spread_cost: float
    slippage_cost: float
    net_pnl: float

class ExecutionEngine:
    def __init__(self, config: RiskConfig):
        self.config = config

    def calculate_order_costs(self, position_size_lots: float, atr: float) -> Dict[str, float]:
        """Calculates commission and simulated slippage."""
        commission = position_size_lots * self.config.COMMISSION_PER_LOT * 2 # Round trip

        # Slippage scaled by ATR and randomized
        slippage_points = np.random.normal(0, self.config.SLIPPAGE_STD) * atr
        slippage_cost = abs(slippage_points * position_size_lots) # Simplification

        return {
            "commission": commission,
            "slippage": slippage_cost
        }

    def resolve_trade(self,
                      entry_price: float,
                      side: str,
                      sl_price: float,
                      tp_price: float,
                      df_subset: pd.DataFrame) -> TradeResult:
        """
        Resolves a trade using intrabar path logic.
        Assumes df_subset starts from the bar of entry.
        """
        spread = self.config.SPREAD_PITS * 0.00001 # Assuming 5 decimal places for now

        for timestamp, row in df_subset.iterrows():
            high = row['high']
            low = row['low']

            # Bid prices (for Long exit)
            bid_high = high - spread/2
            bid_low = low - spread/2

            # Ask prices (for Short exit)
            ask_high = high + spread/2
            ask_low = low + spread/2

            if side == "long":
                # Check if SL hit first or TP hit first in the same bar
                # Simplified: check if both hit, then use a pessimistic approach or random
                # Better: deterministic intrabar path (Open -> High -> Low -> Close or O -> L -> H -> C)
                # For safety, let's assume if both hit, SL hit first.

                hit_sl = bid_low <= sl_price
                hit_tp = bid_high >= tp_price

                if hit_sl and hit_tp:
                    # Pessimistic: hit SL
                    return self._create_result(sl_price, "SL", entry_price, spread, side)
                elif hit_sl:
                    return self._create_result(sl_price, "SL", entry_price, spread, side)
                elif hit_tp:
                    return self._create_result(tp_price, "TP", entry_price, spread, side)

            else: # short
                hit_sl = ask_high >= sl_price
                hit_tp = ask_low <= tp_price

                if hit_sl and hit_tp:
                    return self._create_result(sl_price, "SL", entry_price, spread, side)
                elif hit_sl:
                    return self._create_result(sl_price, "SL", entry_price, spread, side)
                elif hit_tp:
                    return self._create_result(tp_price, "TP", entry_price, spread, side)

        # If not hit within df_subset, force close at last price
        last_row = df_subset.iloc[-1]
        exit_price = last_row['close'] - (spread/2 if side == "long" else -spread/2)
        return self._create_result(exit_price, "Time", entry_price, spread, side)

    def _create_result(self, exit_price: float, exit_type: str, entry_price: float, spread: float, side: str = "long") -> TradeResult:
        # PnL logic: Long = Exit - Entry, Short = Entry - Exit
        pnl = (exit_price - entry_price) if side == "long" else (entry_price - exit_price)

        return TradeResult(
            pnl=pnl,
            exit_type=exit_type,
            exit_price=exit_price,
            commission=0.0, # Filled by environment
            spread_cost=spread,
            slippage_cost=0.0, # Filled by environment
            net_pnl=0.0 # Filled by environment
        )

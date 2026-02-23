import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Tuple, Optional
from Risklayer.config import config

@dataclass
class TradeResult:
    entry_price: float
    exit_price: float
    pnl_raw: float
    pnl_net: float
    exit_type: str  # 'SL', 'TP', or 'Force'
    exit_step: int
    spread_cost: float
    commission: float
    slippage: float

class ExecutionEngine:
    def __init__(self):
        self.spread_pips = config.SPREAD_PIPS
        self.commission_per_lot = config.COMMISSION_PER_LOT
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_costs(self, position_size_lots: float, entry_price: float, atr: float) -> Tuple[float, float]:
        """Calculates commission and simulated slippage."""
        # Commission: $7 per 100k (1 lot)
        commission = position_size_lots * self.commission_per_lot

        # Slippage: Random based on ATR
        slippage_points = np.abs(np.random.normal(0, config.SLIPPAGE_STD * atr))

        return commission, slippage_points

    def execute_trade(self, df: pd.DataFrame, start_idx: int, direction: int,
                      sl_price: float, tp_price: float, position_size_units: float) -> TradeResult:
        """
        Simulates trade execution with realistic Bid/Ask and intrabar logic.
        direction: 1 for Long, -1 for Short
        """
        # Note: units for EURUSD is 100,000 per lot
        lots = position_size_units / 100000.0
        pip_value = 0.0001
        spread_cost_price = (self.spread_pips * pip_value)

        initial_row = df.iloc[start_idx]
        mid_entry = initial_row['open']
        atr = initial_row['atr_14']

        commission, slippage_points = self.calculate_costs(lots, mid_entry, atr)

        if direction == 1:  # LONG
            entry_price = mid_entry + (spread_cost_price / 2) + slippage_points
        else:  # SHORT
            entry_price = mid_entry - (spread_cost_price / 2) - slippage_points

        spread_cost_total = (spread_cost_price * position_size_units)

        # Simulation loop
        for i in range(start_idx, len(df)):
            row = df.iloc[i]
            o, h, l, c = row['open'], row['high'], row['low'], row['close']

            # Convert Mid OHLC to Bid/Ask
            if direction == 1: # Checking for LONG exit (on Bid)
                bid_o, bid_h, bid_l, bid_c = o - spread_cost_price/2, h - spread_cost_price/2, l - spread_cost_price/2, c - spread_cost_price/2

                # Deterministic Path: Open -> Low -> High -> Close (Pessimistic for Long)
                # Check SL first
                if bid_l <= sl_price:
                    exit_price = sl_price
                    return self._finalize_result(entry_price, exit_price, direction, position_size_units, 'SL', i, spread_cost_total, commission, slippage_points)
                if bid_h >= tp_price:
                    exit_price = tp_price
                    return self._finalize_result(entry_price, exit_price, direction, position_size_units, 'TP', i, spread_cost_total, commission, slippage_points)

            else: # Checking for SHORT exit (on Ask)
                ask_o, ask_h, ask_l, ask_c = o + spread_cost_price/2, h + spread_cost_price/2, l + spread_cost_price/2, c + spread_cost_price/2

                # Deterministic Path: Open -> High -> Low -> Close (Pessimistic for Short)
                # Check SL first
                if ask_h >= sl_price:
                    exit_price = sl_price
                    return self._finalize_result(entry_price, exit_price, direction, position_size_units, 'SL', i, spread_cost_total, commission, slippage_points)
                if ask_l <= tp_price:
                    exit_price = tp_price
                    return self._finalize_result(entry_price, exit_price, direction, position_size_units, 'TP', i, spread_cost_total, commission, slippage_points)

        # If we reach here, force close at last price
        last_row = df.iloc[-1]
        exit_price = last_row['close']
        return self._finalize_result(entry_price, exit_price, direction, position_size_units, 'Force', len(df)-1, spread_cost_total, commission, slippage_points)

    def _finalize_result(self, entry, exit, direction, units, exit_type, step, spread_cost, commission, slippage) -> TradeResult:
        pnl_raw = (exit - entry) * units * direction
        pnl_net = pnl_raw - commission

        return TradeResult(
            entry_price=entry,
            exit_price=exit,
            pnl_raw=pnl_raw,
            pnl_net=pnl_net,
            exit_type=exit_type,
            exit_step=step,
            spread_cost=spread_cost,
            commission=commission,
            slippage=slippage
        )

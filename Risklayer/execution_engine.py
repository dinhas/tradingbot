import numpy as np
import logging
from typing import Dict, Optional, Tuple
from Risklayer.config import config

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self):
        self.spreads = config.SPREADS
        self.commissions = config.COMMISSION_PER_LOT
        self.contract_sizes = config.CONTRACT_SIZES
        self.slippage_std = config.SLIPPAGE_STD

    def get_entry_price(self, asset: str, mid_price: float, side: str, atr: float) -> float:
        """Calculates entry price with spread and slippage."""
        spread = self.spreads.get(asset, 0.0)
        slippage = np.random.normal(0, self.slippage_std * atr)
        slippage = max(0, slippage) # Slippage is usually against us

        if side == 'long':
            return mid_price + (spread / 2) + slippage
        else:
            return mid_price - (spread / 2) - slippage

    def check_exit(
        self,
        asset: str,
        side: str,
        open_price: float,
        sl_price: float,
        tp_price: float,
        high: float,
        low: float,
        close: float,
        atr: float
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Checks if SL or TP was hit in the current bar.
        Returns: (exit_price, reason)
        """
        spread = self.spreads.get(asset, 0.0)

        if side == 'long':
            # Long exit rules (Exit at Bid)
            bid_high = high - (spread / 2)
            bid_low = low - (spread / 2)

            # Check if both hit (Conservative: SL first)
            if bid_low <= sl_price and bid_high >= tp_price:
                return sl_price, 'sl'

            if bid_low <= sl_price:
                return sl_price, 'sl'

            if bid_high >= tp_price:
                return tp_price, 'tp'

        else:
            # Short exit rules (Exit at Ask)
            ask_high = high + (spread / 2)
            ask_low = low + (spread / 2)

            # Check if both hit (Conservative: SL first)
            if ask_high >= sl_price and ask_low <= tp_price:
                return sl_price, 'sl'

            if ask_high >= sl_price:
                return sl_price, 'sl'

            if ask_low <= tp_price:
                return tp_price, 'tp'

        return None, None

    def calculate_pnl(
        self,
        asset: str,
        side: str,
        entry_price: float,
        exit_price: float,
        volume: float
    ) -> float:
        """Calculates net profit after commission."""
        contract_size = self.contract_sizes.get(asset, 100000)

        if side == 'long':
            gross_pnl = (exit_price - entry_price) * volume * contract_size
        else:
            gross_pnl = (entry_price - exit_price) * volume * contract_size

        # Commission (per lot = 100,000 units)
        lots = volume # Assuming volume is already in lots?
        # Wait, usually volume is units. If volume is units:
        lots = volume / contract_size # Actually wait, if contract_size is 100,000, 1 lot = 100,000.
        # Let's define volume as "lots".

        commission = self.commissions * lots * 2 # Round turn

        return gross_pnl - commission

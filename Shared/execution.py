import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class TradeConfig:
    """
    Centralized Configuration for Trading Logic.
    Values taken from RiskLayer/src/risk_env.py (Ground Truth).
    """
    # Risk Management
    MAX_RISK_PER_TRADE: float = 0.40      # Hard cap 40% (Very Aggressive)
    DEFAULT_RISK_PCT: float = 0.25        # Fixed 25% risk per trade
    MAX_LEVERAGE: float = 400.0
    MAX_MARGIN_PER_TRADE_PCT: float = 0.80 # Max margin utilization
    
    # Action Scaling (Model Output -> Real Values)
    # SL: 0.2 ATR to 2.0 ATR
    SL_MULT_MIN: float = 0.2
    SL_MULT_MAX: float = 2.0
    
    # TP: 0.5 ATR to 4.0 ATR
    TP_MULT_MIN: float = 0.5
    TP_MULT_MAX: float = 4.0
    
    # Execution constraints
    MIN_LOTS: float = 0.01
    MAX_LOTS: float = 100.0
    CONTRACT_SIZE: int = 100000
    
    # Market Simulation (Friction)
    SPREAD_MIN_PIPS: float = 0.5
    SPREAD_ATR_FACTOR: float = 0.05
    SLIPPAGE_MIN_PIPS: float = 0.5
    SLIPPAGE_MAX_PIPS: float = 1.5
    TRADING_COST_PCT: float = 0.0  # Fees (if any, separate from spread)

class ExecutionEngine:
    """
    Shared Logic for executing trades with ADAPTIVE risk (25%-50%) based on drawdown.
    Centralized source of truth for all trade math and price simulation.
    """
    def __init__(self, config: TradeConfig = TradeConfig()):
        self.config = config

    def get_spread(self, mid_price: float, atr: float) -> float:
        """Calculates dynamic spread in price units."""
        return (self.config.SPREAD_MIN_PIPS * 0.0001 * mid_price) + (self.config.SPREAD_ATR_FACTOR * atr)

    def get_slippage(self, mid_price: float) -> float:
        """Generates random adverse slippage in price units."""
        pips = np.random.uniform(self.config.SLIPPAGE_MIN_PIPS, self.config.SLIPPAGE_MAX_PIPS)
        return pips * 0.0001 * mid_price

    def get_entry_price(self, mid_price: float, direction: int, atr: float, enable_slippage: bool = True) -> float:
        """
        Calculates execution price including spread and optional slippage.
        LONG: Buy at Ask (Mid + Spread + Slippage)
        SHORT: Sell at Bid (Mid - Slippage)
        """
        spread = self.get_spread(mid_price, atr)
        slippage = 0.0
        
        if direction == 1: # LONG
            return mid_price + spread + slippage
        return mid_price - slippage # SHORT

    def get_close_price(self, mid_price: float, direction: int, atr: float, enable_slippage: bool = True) -> float:
        """
        Calculates exit price for a position including spread and optional slippage.
        Close LONG: Sell at BID (Mid - Slippage)
        Close SHORT: Buy at ASK (Mid + Spread + Slippage)
        """
        spread = self.get_spread(mid_price, atr)
        slippage = 0.0

        if direction == 1: # Close LONG
            return mid_price - slippage
        return mid_price + spread + slippage # Close SHORT

    def calculate_pnl(self, 
                      entry_price: float, 
                      exit_price: float, 
                      lots: float, 
                      direction: int, 
                      is_usd_quote: bool = True,
                      is_usd_base: bool = False) -> float:
        """
        Calculates Net P&L in USD including currency conversion.
        """
        price_change = exit_price - entry_price
        # Quote P&L
        gross_pnl_quote = price_change * lots * self.config.CONTRACT_SIZE * direction
        
        # Currency Conversion to USD
        if is_usd_quote:
             return gross_pnl_quote
        elif is_usd_base:
             # USDJPY: PnL / exit_price
             return gross_pnl_quote / exit_price 
        else:
             # Fallback (Cross pairs approx as quote)
             return gross_pnl_quote

    def decode_action(self, action: np.ndarray) -> Tuple[float, float]:
        """
        Decodes model output [-1, 1] into SL/TP multipliers.
        Risk percentage is fixed at config.DEFAULT_RISK_PCT (25%).
        Returns: (sl_mult, tp_mult)
        """
        # SL Scaling: 0.2 ATR to 2.0 ATR
        sl_span = self.config.SL_MULT_MAX - self.config.SL_MULT_MIN
        sl_mult = np.clip(
            (action[0] + 1) / 2 * sl_span + self.config.SL_MULT_MIN,
            self.config.SL_MULT_MIN,
            self.config.SL_MULT_MAX
        )
        
        # TP Scaling: 0.5 ATR to 4.0 ATR
        tp_span = self.config.TP_MULT_MAX - self.config.TP_MULT_MIN
        tp_mult = np.clip(
            (action[1] + 1) / 2 * tp_span + self.config.TP_MULT_MIN,
            self.config.TP_MULT_MIN,
            self.config.TP_MULT_MAX
        )
        
        return float(sl_mult), float(tp_mult)

    def calculate_position_size(self, 
                              equity: float,
                              initial_equity: float,
                              entry_price: float, 
                              sl_dist_price: float, 
                              atr: float,
                              is_usd_quote: bool = True,
                              is_usd_base: bool = False) -> float:
        """
        Calculates lot size using ADAPTIVE risk (25%-50%) to maintain signal strength.
        
        Adaptive Risk Scaling:
        - At 100%+ equity: Risk 25% (normal)
        - At 50% equity: Risk 50% (aggressive recovery)
        - Below 30% equity: Risk capped at 50% (prevent overleverage)
        
        This prevents weak reward signals during training when equity is low.
        
        Args:
            equity: Current account equity
            initial_equity: Starting equity (for adaptive scaling calculation)
            entry_price: Trade entry price
            sl_dist_price: Stop loss distance in price units
            atr: Average True Range
            is_usd_quote: True if USD is quote currency (e.g., EURUSD)
            is_usd_base: True if USD is base currency (e.g., USDJPY)
            
        Returns:
            Lot size (0.0 if insufficient equity/too risky)
        """
        # ADAPTIVE RISK SCALING: Increase risk when equity drops to maintain signal strength
        # Formula: risk_multiplier = initial / max(current, initial * 0.3)
        # This ensures:
        #   - $10 account -> 25% risk ($2.50)
        #   - $5 account -> 50% risk ($2.50) - same absolute impact
        #   - $3 account -> 50% risk ($1.50) - capped to prevent blow-up
        
        min_equity_for_scaling = initial_equity * 0.3  # Don't scale below 30% of initial
        risk_multiplier = initial_equity / max(equity, min_equity_for_scaling)
        effective_risk_pct = min(self.config.DEFAULT_RISK_PCT * risk_multiplier, 0.50)
        
        risk_amount_cash = equity * effective_risk_pct
        
        # Safety Check: Ensure SL isn't too tight (min 0.2 ATR or 0.01% of price)
        min_sl_dist = max(0.0001 * entry_price, 0.2 * atr)
        effective_sl_dist = max(sl_dist_price, min_sl_dist)
        
        # Raw Lot Calculation
        lots = 0.0
        if effective_sl_dist > 1e-9:
            if is_usd_quote:
                # EURUSD: Risk / (SL * 100000)
                lots = risk_amount_cash / (effective_sl_dist * self.config.CONTRACT_SIZE)
            elif is_usd_base:
                # USDJPY: (Risk * Price) / (SL * 100000)
                lots = (risk_amount_cash * entry_price) / (effective_sl_dist * self.config.CONTRACT_SIZE)
            else:
                # Cross pairs (approximate as USD Quote)
                lots = risk_amount_cash / (effective_sl_dist * self.config.CONTRACT_SIZE)
        
        # Leverage Clamping
        if is_usd_quote:
            lot_value_usd = self.config.CONTRACT_SIZE * entry_price
        else:
            lot_value_usd = self.config.CONTRACT_SIZE
            
        max_position_value = (equity * self.config.MAX_MARGIN_PER_TRADE_PCT) * self.config.MAX_LEVERAGE
        max_lots_leverage = max_position_value / max(lot_value_usd, 1e-9)
        
        lots = min(lots, max_lots_leverage)
        
        # Final Clip: ensure minimum lots so trades are never skipped
        if lots < self.config.MIN_LOTS:
            lots = float(self.config.MIN_LOTS)
            
        return float(np.clip(lots, self.config.MIN_LOTS, self.config.MAX_LOTS))

    def get_exit_prices(self, entry_price: float, direction: int, sl_mult: float, tp_mult: float, atr: float, digits: int = 5) -> Tuple[float, float, float, float]:
        """
        Returns (sl_price, tp_price, sl_dist, tp_dist)
        """
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        
        # Round distances to pip precision
        pip_unit = 10 ** -digits
        sl_dist = round(sl_dist / pip_unit) * pip_unit
        tp_dist = round(tp_dist / pip_unit) * pip_unit
        
        sl_price = round(entry_price - (direction * sl_dist), digits)
        tp_price = round(entry_price + (direction * tp_dist), digits)
        
        return sl_price, tp_price, sl_dist, tp_dist

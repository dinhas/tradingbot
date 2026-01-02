import logging
import numpy as np

class PositionSizer:
    """
    Calculates position sizes based on risk parameters and account state.
    Matches the logic in RiskLayer/src/risk_env.py.
    """
    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        
        # Configuration matches RiskEnv
        self.MAX_RISK_PER_TRADE = 0.40      # Max Risk Cap (Absolute ceiling)
        self.BASE_RISK_PER_TRADE = 0.02     # Fixed 2.0% Risk
        self.MAX_MARGIN_PER_TRADE_PCT = 0.80
        self.MAX_LEVERAGE = 400.0
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000
        
    def calculate_risk_lots(self, 
                          account_state, 
                          asset_name, 
                          entry_price, 
                          sl_price, 
                          atr):
        """
        Calculates the lot size for a trade.
        
        Args:
            account_state (dict): Contains 'equity', 'peak_equity', 'balance'.
            asset_name (str): Symbol name (e.g., 'EURUSD').
            entry_price (float): Current price.
            sl_price (float): Stop Loss price.
            atr (float): Current ATR (for min SL checks).
            
        Returns:
            float: Lot size (volume in units / 100,000), rounded to 2 decimals.
        """
        equity = account_state.get('equity', 0.0)
        peak_equity = max(account_state.get('peak_equity', equity), 1e-9)
        
        # 1. Calculate Drawdown-Adjusted Risk
        drawdown = 1.0 - (equity / peak_equity)
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        actual_risk_pct = self.BASE_RISK_PER_TRADE * risk_cap_mult
        
        # Cap absolute risk (safety net)
        actual_risk_pct = min(actual_risk_pct, self.MAX_RISK_PER_TRADE)
        
        risk_amount_cash = equity * actual_risk_pct
        
        # 2. Calculate Stop Loss Distance
        sl_dist_price = abs(entry_price - sl_price)
        
        # Safety: Ensure valid SL distance
        min_sl_dist = max(0.0001 * entry_price, 0.2 * atr)
        if sl_dist_price < min_sl_dist:
            # self.logger.warning(f"SL distance {sl_dist_price:.5f} too small. Adjusted to min {min_sl_dist:.5f}")
            sl_dist_price = min_sl_dist
            
        # 3. Calculate Raw Lots
        # Simplified for USD-based pairs (EURUSD, GBPUSD, etc.) and XAUUSD
        # Logic matches RiskEnv: lots = risk_cash / (sl_dist * contract_size)
        if sl_dist_price <= 0:
            return 0.0
            
        lots = risk_amount_cash / (sl_dist_price * self.CONTRACT_SIZE)
        
        # 4. Leverage Clamping
        # Max Position Value = Equity * Max_Margin_Pct * Max_Leverage
        # Lot Value (approx) = Lots * Contract_Size * Price (for base USD) or just Lots * Contract_Size (for quote USD)
        # Assuming Quote USD (EURUSD, XAUUSD) -> 1 Lot = 100k Units of Base.
        # Position Value in USD = Lots * 100k * Price
        
        # Determine if USD is Quote or Base (Simple heuristic for now)
        is_usd_quote = asset_name.endswith('USD')
        
        if is_usd_quote:
            lot_value_usd = self.CONTRACT_SIZE * entry_price
        else:
            # Fallback for USDJPY etc (USD is Base) -> Value is just 100k USD
            lot_value_usd = self.CONTRACT_SIZE
            
        max_position_value = (equity * self.MAX_MARGIN_PER_TRADE_PCT) * self.MAX_LEVERAGE
        max_lots_leverage = max_position_value / lot_value_usd
        
        lots = min(lots, max_lots_leverage)
        
        # 5. Final Constraints
        if lots < self.MIN_LOTS:
            return 0.0
            
        lots = np.clip(lots, self.MIN_LOTS, 100.0)
        
        return round(lots, 2)

import logging
import numpy as np
from .config import load_config

class PositionSizer:
    """
    Calculates position sizes based on margin usage.
    
    Logic:
    - If balance < $30: Use 0.01 lots (Minimum).
    - If balance >= $30: Use 25% margin usage per trade.
    """
    def __init__(self, leverage=None):
        self.logger = logging.getLogger("LiveExecution")
        
        # Load leverage from config if not provided
        if leverage is None:
            try:
                config = load_config()
                self.MAX_LEVERAGE = config.get("LEVERAGE", 400.0)
            except Exception:
                self.MAX_LEVERAGE = 400.0
        else:
            self.MAX_LEVERAGE = leverage
            
        self.TARGET_MARGIN_PCT = 0.25       # 25% Margin target
        self.MIN_LOTS = 0.01
        
        # Contract sizes: 100 for Gold, 100,000 for Forex
        self.CONTRACT_SIZES = {
            'XAUUSD': 100,
            'EURUSD': 100000,
            'GBPUSD': 100000,
            'USDJPY': 100000,
            'USDCHF': 100000
        }
        
    def calculate_risk_lots(self, 
                          account_state, 
                          asset_name, 
                          entry_price, 
                          sl_price, 
                          atr):
        """
        Calculates the lot size for a trade based on margin requirements.
        """
        balance = account_state.get('balance', 0.0)
        equity = account_state.get('equity', balance)
        
        # 1. Small Account Override
        if balance < 30:
            # self.logger.info(f"Balance ${balance:.2f} < $30. Using default minimum: 0.01 lots")
            return 0.01
            
        # 2. Margin-Based Sizing (Target 25% Margin)
        target_margin_usd = equity * self.TARGET_MARGIN_PCT
        
        contract_size = self.CONTRACT_SIZES.get(asset_name, 100000)
        
        # Margin formula: (Lots * Contract Size * BasePrice) / Leverage
        # To get Lots: (Target Margin * Leverage) / (Contract Size * BasePrice)
        
        # In cTrader, for EURUSD, base is EUR. For USDJPY, base is USD.
        # If USD is Quote (EURUSD, GBPUSD, XAUUSD): BasePrice = entry_price
        # If USD is Base (USDJPY, USDCHF): BasePrice = 1.0
        
        is_usd_quote = asset_name.endswith('USD')
        base_price = entry_price if is_usd_quote else 1.0
        
        # Margin per 1.00 Lot
        margin_per_lot = (contract_size * base_price) / self.MAX_LEVERAGE
        
        if margin_per_lot <= 0:
            return self.MIN_LOTS
            
        lots = target_margin_usd / margin_per_lot
        
        # 3. Round and Clamp
        lots = np.clip(lots, self.MIN_LOTS, 100.0)
        
        # Round to 2 decimals (0.01 step)
        final_lots = round(float(lots), 2)
        
        # self.logger.info(f"Sizing for {asset_name}: Balance ${balance:.2f}, Target Margin ${target_margin_usd:.2f} -> {final_lots} lots")
        
        return final_lots
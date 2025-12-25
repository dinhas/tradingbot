import pandas as pd
import numpy as np
import logging
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine
from TradeGuard.src.feature_calculator import TradeGuardFeatureCalculator

class FeatureManager:
    """
    Coordinates feature calculation for Alpha, Risk, and TradeGuard models.
    """
    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.alpha_fe = AlphaFeatureEngine()
        
        # History buffers for each asset (need at least 200 bars for some indicators like MA200)
        self.history = {asset: pd.DataFrame() for asset in self.assets}
        self.max_history = 300 # Keep 300 candles

    def push_candle(self, asset, candle_data):
        """
        Pushes a new candle to the history buffer.
        candle_data: dict with [open, high, low, close, volume] and 'timestamp' index
        """
        # Create a single row DataFrame
        ts = candle_data.pop('timestamp')
        new_row = pd.DataFrame([candle_data], index=[ts])
        
        # Append and trim
        self.history[asset] = pd.concat([self.history[asset], new_row])
        if len(self.history[asset]) > self.max_history:
            self.history[asset] = self.history[asset].iloc[-self.max_history:]
            
        self.logger.debug(f"Pushed candle for {asset}. Buffer size: {len(self.history[asset])}")

    def get_alpha_risk_observation(self, portfolio_state):
        """
        Calculates the 140-feature vector for Alpha/Risk models.
        """
        # 1. Prepare data_dict for Alpha FE
        # We need to ensure we have data for all assets
        data_dict = {asset: df for asset, df in self.history.items() if not df.empty}
        
        if len(data_dict) < len(self.assets):
            self.logger.warning("Not all assets have data yet. Observations might be incomplete.")
        
        # 2. Preprocess data
        # AlphaFE returns (raw_df, normalized_df)
        _, normalized_df = self.alpha_fe.preprocess_data(data_dict)
        
        # 3. Get the last row (latest features)
        latest_features = normalized_df.iloc[-1].to_dict()
        
        # 4. Construct observation
        return self.alpha_fe.get_observation(latest_features, portfolio_state)

    def get_tradeguard_observation(self, trade_infos, portfolio_state):
        """
        Calculates the 105-feature vector for the TradeGuard model.
        """
        # TradeGuardFeatureCalculator expects a df_dict in __init__
        data_dict = {asset: df for asset, df in self.history.items() if not df.empty}
        
        # Since TradeGuardFeatureCalculator precomputes everything, we instantiate it for the current state
        # Note: This might be slightly inefficient but follows the "logic reuse" requirement.
        calculator = TradeGuardFeatureCalculator(data_dict)
        
        # Step is -1 (latest)
        return calculator.get_multi_asset_obs(-1, trade_infos, portfolio_state)

    def is_ready(self):
        """Checks if enough history is collected for all assets."""
        min_required = 200 # For MA200
        for asset in self.assets:
            if len(self.history[asset]) < min_required:
                return False
        return True

import pandas as pd
import numpy as np
import logging
from collections import deque
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
        
        # History buffers for each asset
        self.history = {asset: pd.DataFrame() for asset in self.assets}
        self.max_history = 300 
        
        # Risk-specific history (Last 5 trades per asset)
        self.risk_pnl_history = {asset: deque([0.0]*5, maxlen=5) for asset in self.assets}
        self.risk_action_history = {asset: deque([np.zeros(3) for _ in range(5)], maxlen=5) for asset in self.assets}

    def push_candle(self, asset, candle_data):
        """
        Pushes a new candle to the history buffer.
        candle_data: dict with [open, high, low, close, volume] and 'timestamp' index
        """
        # Create a single row DataFrame
        ts = candle_data.pop('timestamp')
        # Floor to minutes to ensure alignment across assets even if arrival varies by seconds
        ts = ts.floor('min')
        new_row = pd.DataFrame([candle_data], index=[ts])
        
        # Append and trim
        self.history[asset] = pd.concat([self.history[asset], new_row])
        if len(self.history[asset]) > self.max_history:
            self.history[asset] = self.history[asset].iloc[-self.max_history:]
            
        self.logger.debug(f"Pushed candle for {asset}. Buffer size: {len(self.history[asset])}")

    def record_risk_trade(self, asset, pnl_pct, actions):
        """
        Records trade outcome for Risk model's historical features.
        actions: [SL_Mult, TP_Mult, Risk_Pct]
        """
        self.risk_pnl_history[asset].append(pnl_pct)
        self.risk_action_history[asset].append(actions)

    def get_alpha_observation(self, portfolio_state):
        """Calculates the 140-feature vector for Alpha model."""
        data_dict = {asset: df for asset, df in self.history.items() if not df.empty}
        _, normalized_df = self.alpha_fe.preprocess_data(data_dict)
        latest_features = normalized_df.iloc[-1].to_dict()
        return self.alpha_fe.get_observation(latest_features, portfolio_state)

    def get_risk_observation(self, asset, alpha_obs, portfolio_state):
        """
        Constructs the 165-feature vector for Risk model.
        """
        # 1. Market State (140) - alpha_obs passed in
        
        # 2. Account State (5)
        equity = portfolio_state.get('equity', 10.0)
        initial_equity = portfolio_state.get('initial_equity', 10.0)
        peak_equity = portfolio_state.get('peak_equity', initial_equity)
        
        drawdown = 1.0 - (equity / peak_equity) if peak_equity > 0 else 0
        equity_norm = equity / initial_equity if initial_equity > 0 else 1.0
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        account_obs = np.array([
            equity_norm,
            drawdown,
            0.0, # Leverage placeholder
            risk_cap_mult,
            0.0  # Padding
        ], dtype=np.float32)
        
        # 3. History (25)
        hist_pnl = np.array(self.risk_pnl_history[asset], dtype=np.float32)
        hist_acts = np.array(self.risk_action_history[asset], dtype=np.float32).flatten()
        
        return np.concatenate([alpha_obs, account_obs, hist_pnl, hist_acts])

    def get_tradeguard_observation(self, trade_infos, portfolio_state):
        """Calculates the 105-feature vector for the TradeGuard model."""
        data_dict = {asset: df for asset, df in self.history.items() if not df.empty}
        calculator = TradeGuardFeatureCalculator(data_dict)
        return calculator.get_multi_asset_obs(-1, trade_infos, portfolio_state)

    def is_ready(self):
        """Checks if enough history is collected for all assets."""
        min_required = 200 # For MA200
        for asset in self.assets:
            if len(self.history[asset]) < min_required:
                return False
        return True
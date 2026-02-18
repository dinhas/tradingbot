import pandas as pd
import numpy as np
import logging
from collections import deque
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine
from RiskLayer.src.feature_engine import FeatureEngine as RiskFeatureEngine

class FeatureManager:
    """
    Coordinates feature calculation for Alpha, Risk, and TradeGuard models.
    """
    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.alpha_fe = AlphaFeatureEngine()
        self.risk_fe = RiskFeatureEngine()
        
        # History buffers for each asset
        self.history = {asset: pd.DataFrame() for asset in self.assets}
        self.max_history = 300 
        
        # Risk-specific history (Last 5 trades per asset)
        self.risk_pnl_history = {asset: deque([0.0]*5, maxlen=5) for asset in self.assets}
        self.risk_action_history = {asset: deque([np.zeros(3) for _ in range(5)], maxlen=5) for asset in self.assets}
        
        # Real-time spread tracking
        self.live_spreads = {asset: {'bid': 0.0, 'ask': 0.0, 'spread': 0.0} for asset in self.assets}

    def push_candle(self, asset, candle_data):
        """
        Pushes a new candle to the history buffer.
        If the timestamp exists, updates the row. If not, appends.
        """
        ts = candle_data.pop('timestamp')
        ts = pd.Timestamp(ts)
        # Floor to minutes to ensure alignment across assets
        ts = ts.floor('min')
        
        new_row = pd.Series(candle_data, name=ts)
        
        if ts in self.history[asset].index:
            # Update existing row
            self.history[asset].loc[ts] = new_row
        else:
            # Append new row
            self.history[asset] = pd.concat([self.history[asset], pd.DataFrame([new_row])])
            # Ensure index is sorted after append
            self.history[asset] = self.history[asset].sort_index()

        # Trim to max history
        if len(self.history[asset]) > self.max_history:
            self.history[asset] = self.history[asset].iloc[-self.max_history:]
            
        self.logger.debug(f"Pushed candle for {asset}. Buffer size: {len(self.history[asset])}")

    def update_data(self, symbol_id, ohlcv_res):
        """
        Updates internal history from cTrader trendbars response.
        """
        from datetime import datetime
        
        asset = self._get_asset_name_from_id(symbol_id)
        if not asset:
             self.logger.error(f"Unknown symbol_id: {symbol_id}")
             return

        # trendbars are often in chronological order but let's be sure
        new_rows = []
        for bar in ohlcv_res.trendbar:
             # cTrader Price = value / 100,000 (Consistent with training data fetcher)
             divisor = 100000.0
             
             low = bar.low / divisor
             new_rows.append({
                 'timestamp': datetime.fromtimestamp(bar.utcTimestampInMinutes * 60),
                 'open': low + (bar.deltaOpen / divisor),
                 'high': low + (bar.deltaHigh / divisor),
                 'low': low,
                 'close': low + (bar.deltaClose / divisor),
                 'volume': bar.volume
             })
        
        if new_rows:
             # Sort by timestamp
             new_rows.sort(key=lambda x: x['timestamp'])
             # For each row, push
             for row in new_rows:
                  self.push_candle(asset, row)

    def update_from_trendbar(self, asset, bar):
        """
        Updates history from a single Protobuf Trendbar object.
        """
        from datetime import datetime
        
        # cTrader Price = value / 100,000
        divisor = 100000.0
        
        low = bar.low / divisor
        row = {
             'timestamp': datetime.fromtimestamp(bar.utcTimestampInMinutes * 60),
             'open': low + (bar.deltaOpen / divisor),
             'high': low + (bar.deltaHigh / divisor),
             'low': low,
             'close': low + (bar.deltaClose / divisor),
             'volume': bar.volume
        }
        self.push_candle(asset, row)

    def _get_asset_name_from_id(self, symbol_id):
         # Standard mapping from Orchestrator/Client
         mapping = {1: 'EURUSD', 2: 'GBPUSD', 41: 'XAUUSD', 6: 'USDCHF', 4: 'USDJPY'}
         return mapping.get(symbol_id)

    def get_atr(self, asset):
        """Calculates the current 14-period ATR for the given asset."""
        if len(self.history[asset]) < 15:
            return 0.0
        
        from ta.volatility import AverageTrueRange
        df = self.history[asset]
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        return float(atr.iloc[-1])

    def record_risk_trade(self, asset, pnl_pct, actions):
        """
        Records trade outcome for Risk model's historical features.
        actions: [SL_Mult, TP_Mult, Risk_Pct]
        """
        self.risk_pnl_history[asset].append(pnl_pct)
        self.risk_action_history[asset].append(actions)

    def get_alpha_observation(self, asset, portfolio_state):
        """Calculates the 40-feature vector for Alpha model for a specific asset."""
        data_dict = {a: df for a, df in self.history.items() if not df.empty}
        _, normalized_df = self.alpha_fe.preprocess_data(data_dict)
        latest_features = normalized_df.iloc[-1].to_dict()

        # Alpha model needs 40 features: 25 asset-specific + 15 global
        return self.alpha_fe.get_observation(latest_features, portfolio_state, asset)

    def push_spot(self, asset, bid, ask):
        """Updates the latest real-time spread, bid, and ask for an asset."""
        if asset not in self.live_spreads:
            return
        self.live_spreads[asset] = {
            'bid': bid,
            'ask': ask,
            'spread': ask - bid
        }

    def get_risk_observation(self, asset, alpha_outputs, portfolio_state):
        """
        Constructs the 40-feature vector for Risk model using RiskFeatureEngine.
        alpha_outputs: dict with [direction, quality, meta]
        """
        data_dict = {a: df for a, df in self.history.items() if not df.empty}
        
        # Inject current live spreads and alpha outputs into the data for preprocessing
        # We'll create a temporary copy to avoid polluting historical data with partial rows
        processed_data = {}
        for a, df in data_dict.items():
            temp_df = df.copy()
            # Add spread column for RiskFeatureEngine to use
            spread_val = self.live_spreads[a]['spread']
            if spread_val <= 0:
                 spread_val = 0.3 if a == 'XAUUSD' else 0.00002
            temp_df[f"{a}_spread"] = spread_val
            
            # Add alpha outputs as columns
            if a == asset:
                temp_df[f"{a}_alpha_direction"] = alpha_outputs.get('direction', 0)
                temp_df[f"{a}_alpha_quality"] = alpha_outputs.get('quality', 0)
                temp_df[f"{a}_alpha_meta"] = alpha_outputs.get('meta', 0)
            else:
                temp_df[f"{a}_alpha_direction"] = 0
                temp_df[f"{a}_alpha_quality"] = 0
                temp_df[f"{a}_alpha_meta"] = 0
                
            processed_data[a] = temp_df

        # Preprocess using Risk Feature Engine
        _, normalized_df = self.risk_fe.preprocess_data(processed_data)
        latest_features = normalized_df.iloc[-1].to_dict()
        
        # Get 40-dim observation
        return self.risk_fe.get_observation(latest_features, portfolio_state, asset)


    def is_ready(self):
        """Checks if enough history is collected for all assets."""
        min_required = 200 # For MA200
        for asset in self.assets:
            if len(self.history[asset]) < min_required:
                return False
        return True
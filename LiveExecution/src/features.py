import pandas as pd
import numpy as np
import logging
from collections import deque
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine
from RiskLayer.src.feature_engine import RiskFeatureEngine

class FeatureManager:
    """
    Coordinates feature calculation for Alpha and Risk models.
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

        # Cache for performance
        self.last_preprocessed_ts = None
        self.cached_normalized_df = None
        self.cached_risk_df = None

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
        
        # Invalidate cache on new high-timestamp candle
        if self.last_preprocessed_ts is not None and ts > self.last_preprocessed_ts:
            self.logger.debug(f"Invalidating feature cache due to newer candle: {ts}")
            self.last_preprocessed_ts = None
            self.cached_normalized_df = None

    def update_data(self, symbol_id, ohlcv_res):
        """
        Updates internal history from cTrader trendbars response.
        """
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbar
        from datetime import datetime
        
        asset = self._get_asset_name_from_id(symbol_id)
        if not asset:
             self.logger.error(f"Unknown symbol_id: {symbol_id}")
             return

        # trendbars are often in chronological order but let's be sure
        new_rows = []
        for bar in ohlcv_res.trendbar:
             divisor = 100.0 if asset == 'XAUUSD' else 100000.0
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

    def update_single_bar(self, symbol_id, bar):
        """Processes a single ProtoOATrendbar from a SpotEvent."""
        from datetime import datetime
        asset = self._get_asset_name_from_id(symbol_id)
        if not asset: return
        
        divisor = 100.0 if asset == 'XAUUSD' else 100000.0
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

    def record_risk_trade(self, asset, pnl_pct, actions):
        """
        Records trade outcome for Risk model's historical features.
        actions: [SL_Mult, TP_Mult, Risk_Pct]
        """
        self.risk_pnl_history[asset].append(pnl_pct)
        self.risk_action_history[asset].append(actions)

    def get_alpha_observation(self, asset, portfolio_state):
        """Calculates the 40-feature vector for Alpha model for a specific asset."""
        # Use cached preprocessed dataframe if the latest timestamp matches
        latest_ts = max(df.index.max() for df in self.history.values() if not df.empty)
        
        if self.last_preprocessed_ts == latest_ts and self.cached_normalized_df is not None:
             self.logger.debug(f"Using cached features for TS: {latest_ts}")
             normalized_df = self.cached_normalized_df
        else:
             self.logger.info(f"Recalculating features for TS: {latest_ts}...")
             data_dict = {a: df for a, df in self.history.items() if not df.empty}
             _, normalized_df = self.alpha_fe.preprocess_data(data_dict)
             self.last_preprocessed_ts = latest_ts
             self.cached_normalized_df = normalized_df
        
        latest_features = normalized_df.iloc[-1].to_dict()
        return self.alpha_fe.get_observation(latest_features, portfolio_state, asset)

    def get_risk_observation(self, asset, alpha_signal, portfolio_state):
        """
        Constructs the 86-feature vector for Risk model.
        (81 Market features + 5 Account features)
        """
        # 1. Preprocess data using RiskFeatureEngine
        latest_ts = max(df.index.max() for df in self.history.values() if not df.empty)
        
        if self.last_preprocessed_ts == latest_ts and self.cached_risk_df is not None:
             risk_df = self.cached_risk_df
        else:
             data_dict = {a: df.copy() for a, df in self.history.items() if not df.empty}
             # Inject the current alpha signal into the dataframe for the engine
             for a in self.assets:
                  if a == asset:
                       data_dict[a]['alpha_signal'] = alpha_signal
                       data_dict[a]['alpha_conf'] = 1.0 # Placeholder
                  else:
                       data_dict[a]['alpha_signal'] = 0.0
                       data_dict[a]['alpha_conf'] = 0.0
             
             risk_df = self.risk_fe.preprocess_data(data_dict)
             self.cached_risk_df = risk_df
             
        # 2. Extract features for target asset
        # Get only columns for this asset
        asset_cols = [c for c in risk_df.columns if c.startswith(f"{asset}_")]
        # RiskTradingEnv uses a specific order, which RiskFeatureEngine.preprocess_data now preserves
        # But let's be safe and just take them in order they appear in the df for that asset
        market_obs = risk_df.iloc[-1][asset_cols].values.astype(np.float32)
        
        # 3. Account State (5)
        equity = portfolio_state.get('equity', 10000.0)
        initial_equity = portfolio_state.get('initial_equity', 10000.0)
        peak_equity = portfolio_state.get('peak_equity', initial_equity)
        
        drawdown = np.clip(1.0 - (equity / peak_equity) if peak_equity > 0 else 0, 0.0, 1.0)
        equity_norm = np.clip(equity / initial_equity if initial_equity > 0 else 1.0, 0.0, 6.0)
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        spread_val = 0.0001 # TODO: Get real spread
        asset_id = float(self.assets.index(asset))
        
        account_obs = np.array([
            spread_val,
            asset_id,
            equity_norm,
            drawdown,
            risk_cap_mult
        ], dtype=np.float32)
        
        # Total 81 (market) + 5 (account) = 86
        return np.concatenate([market_obs, account_obs])

    def is_ready(self):
        """Checks if enough history is collected for all assets."""
        min_required = 200 # For MA200
        for asset in self.assets:
            if len(self.history[asset]) < min_required:
                return False
        return True

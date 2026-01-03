import pandas as pd
import numpy as np
import logging
from collections import deque
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine
from Alpha.src.feature_normalizer import FrozenFeatureNormalizer
from TradeGuard.src.feature_calculator import TradeGuardFeatureCalculator

class FeatureManager:
    """
    Coordinates feature calculation for Alpha, Risk, and TradeGuard models.
    """
    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.alpha_fe = AlphaFeatureEngine()
        
        # Load Frozen Normalizer for Alpha/Risk models
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent
        norm_path = project_root / "models" / "checkpoints" / "alpha" / "feature_normalizer.pkl"
        
        self.normalizer = FrozenFeatureNormalizer()
        if norm_path.exists():
            self.logger.info(f"Loading frozen feature normalizer from {norm_path}")
            self.normalizer.load(str(norm_path))
        else:
            self.logger.warning(f"Frozen feature normalizer NOT found at {norm_path}. Predictions may be unreliable.")
        
        # History buffers for each asset
        self.history = {asset: pd.DataFrame() for asset in self.assets}
        self.max_history = 300 

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
        from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbar
        from datetime import datetime
        
        asset = self._get_asset_name_from_id(symbol_id)
        if not asset:
             self.logger.error(f"Unknown symbol_id: {symbol_id}")
             return

        # trendbars are often in chronological order but let's be sure
        new_rows = []
        for bar in ohlcv_res.trendbar:
             # cTrader uses low, deltaOpen, deltaHigh, deltaClose relative to low?
             # Actually ProtoOATrendbar: low, deltaOpen, deltaHigh, deltaClose
             # open = low + deltaOpen
             # high = low + deltaHigh
             # close = low + deltaClose
             # divisor is usually 100,000 for prices (pips) or 100 for Gold
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

    def get_alpha_observation(self, asset, portfolio_state):
        """
        Calculates the 40-feature vector for Alpha model for a specific asset.
        """
        data_dict = {a: df for a, df in self.history.items() if not df.empty}
        # Get RAW (aligned but not rolling-normalized) dataframe
        raw_df, _ = self.alpha_fe.preprocess_data(data_dict)
        
        if raw_df.empty:
             return np.zeros(40, dtype=np.float32)
             
        # Apply frozen normalization to the latest row
        latest_row = raw_df.iloc[-1:].copy()
        if self.normalizer.is_fitted:
            latest_row = self.normalizer.transform_df(latest_row)
        
        latest_features = latest_row.iloc[0].to_dict()
        
        # Extract features for specific asset
        return self.alpha_fe.get_observation(latest_features, portfolio_state, asset)

    def get_risk_observation(self, alpha_obs, portfolio_state):
        """
        Constructs the 45-feature vector for Risk model.
        [0..39] Alpha Features
        [40..44] Account State
        """
        # 1. Market State (40) - alpha_obs passed in
        
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
        
        return np.concatenate([alpha_obs, account_obs])

    def get_tradeguard_observation(self, asset, trade_info, portfolio_state, norm_stats=None):
        """
        Calculates the normalized 25-feature vector for the TradeGuard model for a specific asset.
        """
        data_dict = {a: df for a, df in self.history.items() if not df.empty}
        calculator = TradeGuardFeatureCalculator(data_dict)
        
        # asset_portfolio_state should contain action_raw, signal_persistence, etc.
        # global_state should contain total_drawdown, total_exposure.
        asset_state = portfolio_state.get(asset, {})
        
        # Get raw 25-feature vector
        obs = calculator.get_single_asset_obs(
            asset, 
            -1, # Latest step
            trade_info, 
            asset_state, 
            portfolio_state
        )
        
        # Apply normalization if stats are provided
        if norm_stats:
            mean = norm_stats.get('feature_mean')
            std = norm_stats.get('feature_std')
            if mean is not None and std is not None:
                obs = (obs - mean) / std
                obs = np.clip(obs, -10, 10) # Clip extreme values as in environment
                obs = np.nan_to_num(obs, nan=0.0)
                
        return obs.astype(np.float32)

    def is_ready(self):
        """Checks if enough history is collected for all assets."""
        min_required = 200 # For MA200
        for asset in self.assets:
            if len(self.history[asset]) < min_required:
                return False
        return True
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from .feature_engine import FeatureEngine

class TradingEnv(gym.Env):
    """
    Trading environment for RL agent in Alpha Layer.
    
    Curriculum Stages:
        Stage 1: Direction only (5 outputs)
        Stage 2: Direction + Position sizing (10 outputs)
        Stage 3: Direction + Position sizing + SL/TP (20 outputs)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir='data', stage=3, is_training=True):
        super(TradingEnv, self).__init__()
        
        self.data_dir = data_dir
        self.stage = stage
        self.is_training = is_training
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Configuration Constants
        self.MAX_POS_SIZE_PCT = 0.50
        self.MAX_TOTAL_EXPOSURE = 0.60
        self.DRAWDOWN_LIMIT = 0.25
        self.MIN_POSITION_SIZE = 10  
        self.MIN_ATR_MULTIPLIER = 0.0001
        self.REWARD_LOG_INTERVAL = 5000

        # Load Data
        self.data = self._load_data()
        self.feature_engine = FeatureEngine()
        self.raw_data, self.processed_data = self.feature_engine.preprocess_data(self.data)
        
        # Cache for performance
        self._cache_data_arrays()
        
        # Master Obs Matrix for Combined Backtest Speed
        self._create_master_obs_matrix()
        
        # Define Action Space
        if self.stage == 1:
            self.action_dim = 5   
        elif self.stage == 2:
            self.action_dim = 10  
        else:
            self.action_dim = 20  
            
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        
        # State Variables
        self.current_step = 0
        self.max_steps = len(self.processed_data) - 1
        self.equity = 10000.0
        self.leverage = 100
        self.positions = {asset: None for asset in self.assets}
        
        self.completed_trades = []
        self.start_equity = self.equity
        self.peak_equity = self.equity
        
    def _create_master_obs_matrix(self):
        """Creates a single matrix containing all asset observations for batch inference."""
        n_steps = len(self.processed_data)
        self.master_obs_matrix = np.zeros((n_steps, len(self.assets) * 40), dtype=np.float32)
        
        # Use vectorized extraction for each asset
        for i, asset in enumerate(self.assets):
            obs_matrix = self.feature_engine.get_observation_vectorized(self.processed_data, asset)
            self.master_obs_matrix[:, i*40:(i+1)*40] = obs_matrix

    def _cache_data_arrays(self):
        self.close_arrays = {a: self.raw_data[f"{a}_close"].values.astype(np.float32) for a in self.assets}
        self.low_arrays = {a: self.raw_data[f"{a}_low"].values.astype(np.float32) for a in self.assets}
        self.high_arrays = {a: self.raw_data[f"{a}_high"].values.astype(np.float32) for a in self.assets}
        self.atr_arrays = {a: self.raw_data[f"{a}_atr_14"].values.astype(np.float32) for a in self.assets}

    def _load_data(self):
        data = {}
        for asset in self.assets:
            # Check multiple possible paths
            paths = [
                f"{self.data_dir}/{asset}_5m.parquet",
                f"{self.data_dir}/{asset}_5m_2025.parquet",
                f"data/{asset}_5m.parquet"
            ]
            
            df = None
            for p in paths:
                try:
                    df = pd.read_parquet(p)
                    logging.info(f"Loaded {asset} from {p}")
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                logging.error(f"Data for {asset} not found.")
                # Create dummy data to prevent crash
                dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
                df = pd.DataFrame(index=dates)
                for col in ['open', 'high', 'low', 'close']: df[f"{asset}_{col}"] = 1.0
                df[f"{asset}_volume"] = 100
                df[f"{asset}_atr_14"] = 0.001
            
            data[asset] = df
        return data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.equity = 10000.0
        self.current_step = 500 if not self.is_training else np.random.randint(500, self.max_steps - 288)
        self.positions = {asset: None for asset in self.assets}
        self.completed_trades = []
        return self._get_observation(), {}

    def _get_observation(self):
        current_row = self.processed_data.iloc[self.current_step]
        # In multi-asset, we default to the first asset or a specific one set via helper
        asset = getattr(self, 'current_asset', self.assets[0])
        return self.feature_engine.get_observation(current_row, {}, asset)

    def set_asset(self, asset):
        self.current_asset = asset

    def _get_current_timestamp(self):
        return self.processed_data.index[self.current_step]

    def _update_positions(self):
        prices = {a: self.close_arrays[a][self.current_step] for a in self.assets}
        for asset, pos in list(self.positions.items()):
            if pos is None: continue
            price = prices[asset]
            if pos['direction'] == 1:
                if price <= pos['sl'] or price >= pos['tp']: self._close_position(asset, price)
            else:
                if price >= pos['sl'] or price <= pos['tp']: self._close_position(asset, price)

    def _close_position(self, asset, price):
        pos = self.positions[asset]
        pnl = (price - pos['entry_price']) * pos['direction'] * (pos['size'] * self.leverage / pos['entry_price'])
        self.equity += pnl
        self.completed_trades.append({'asset': asset, 'pnl': pnl, 'timestamp': self._get_current_timestamp()})
        self.positions[asset] = None

    def _open_position(self, asset, direction, act, price, atr):
        size = act['size'] * self.MAX_POS_SIZE_PCT * self.equity
        sl = price - (direction * act['sl_mult'] * atr)
        tp = price + (direction * act['tp_mult'] * atr)
        self.positions[asset] = {'direction': direction, 'entry_price': price, 'size': size, 'sl': sl, 'tp': tp, 'entry_step': self.current_step}

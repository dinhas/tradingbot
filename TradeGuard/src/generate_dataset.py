import logging
import os
import sys
import multiprocessing
import time
from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import ta
from ta.volatility import AverageTrueRange

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import Downloader for automatic data fetching
try:
    from TradeGuard.src.download_data import DataFetcherTraining
except ImportError:
    DataFetcherTraining = None

# Import TradingEnv safely
try:
    from Alpha.src.trading_env import TradingEnv
except ImportError:
    TradingEnv = object

class DatasetGenerationEnv(TradingEnv):
    """
    Extended TradingEnv that captures trade signals and their outcomes.
    """
    def __init__(self, df_dict, feature_engine, precomputed_features=None, data_dir='data', stage=3, data=None):
        super().__init__(data_dir=data_dir, is_training=False, stage=stage, data=data)
        self.signals = []
        self.df_dict = df_dict
        self.guard_feature_engine = feature_engine
        self.precomputed_features = precomputed_features
        self.last_action = None
        # Track signal history for features 3 & 4
        self.asset_signal_history = {asset: [] for asset in self.assets}

    def _get_current_timestamp(self):
        asset = self.assets[0]
        return self.df_dict[asset].index[self.current_step]

    def set_last_action(self, action):
        self.last_action = action

    def _execute_trades(self, actions):
        """Capture signals for every direction != 0, regardless of current position state."""
        current_prices = self._get_current_prices()
        atrs = self._get_current_atrs()
        
        for asset, act in actions.items():
            direction = act['direction']
            # Update signal history for features 3 & 4
            self.asset_signal_history[asset].append(direction)
            if len(self.asset_signal_history[asset]) > 20:
                self.asset_signal_history[asset].pop(0)

            if direction != 0:
                self._record_signal(asset, direction, act, current_prices[asset], atrs[asset])
        
        # Execute actual trades in the base environment (manages real positions/equity)
        super()._execute_trades(actions)

    def _calculate_persistence_reversal(self, asset, current_direction):
        history = self.asset_signal_history[asset]
        if len(history) < 2:
            return 1.0, 0.0
            
        # Persistence: Count consecutive signals of same direction backwards
        count = 0
        for sig in reversed(history[:-1]): # Exclude current
            if sig == current_direction:
                count += 1
            else:
                break
        persistence = count + 1.0
        
        # Reversal: Did we flip from -1 to 1 or 1 to -1?
        # Check last non-zero signal
        prev_signal = 0
        for sig in reversed(history[:-1]):
            if sig != 0:
                prev_signal = sig
                break
        
        reversal = 1.0 if prev_signal != 0 and prev_signal != current_direction else 0.0
        return persistence, reversal

    def _record_signal(self, asset, direction, act, price, atr):
        """Records a signal for TradeGuard dataset by simulating a virtual trade outcome."""
        # Save current position to restore it after simulation
        real_pos = self.positions[asset]
        
        # Calculate SL/TP for virtual labeling
        atr_val = max(atr, price * self.MIN_ATR_MULTIPLIER)
        sl_dist = act['sl_mult'] * atr_val
        tp_dist = act['tp_mult'] * atr_val
        
        # Create virtual position for simulation
        self.positions[asset] = {
            'direction': direction,
            'entry_price': price,
            'size': 1.0, # Unit size for labeling
            'sl': price - (direction * sl_dist),
            'tp': price + (direction * tp_dist),
            'entry_step': self.current_step,
            'sl_dist': sl_dist,
            'tp_dist': tp_dist
        }
        
        try:
            # Simulate outcome using peek-ahead
            outcome = super()._simulate_trade_outcome_with_timing(asset)
            
            # --- Feature Calculation ---
            asset_idx = self.assets.index(asset)
            if self.stage == 1:
                base_idx = asset_idx
            elif self.stage == 2:
                base_idx = asset_idx * 2
            else:
                base_idx = asset_idx * 4
            
            persistence, reversal = self._calculate_persistence_reversal(asset, direction)
            
            recent_trades = [{'pnl': t['net_pnl']} for t in self.all_trades[-10:]]
            total_exposure = sum(pos['size'] for pos in self.positions.values() if pos is not None)
            
            # Calculate dynamic position value (Group E)
            pos_size_val = self.equity * act['size'] * 0.5
            
            portfolio_state = {
                'equity': self.equity,
                'peak_equity': self.peak_equity,
                'total_exposure': total_exposure,
                'open_positions_count': sum(1 for p in self.positions.values() if p is not None),
                'recent_trades': recent_trades,
                'asset_action_raw': self.last_action[base_idx] if self.last_action is not None else 0,
                'asset_recent_actions': [self.last_action[base_idx]] * 5 if self.last_action is not None else [0]*5,
                'asset_signal_persistence': persistence,
                'asset_signal_reversal': reversal,
                'position_value': pos_size_val if pos_size_val > 0 else 0
            }
            
            trade_info = {
                'entry_price': price,
                'sl': self.positions[asset]['sl'],
                'tp': self.positions[asset]['tp'],
                'direction': direction
            }
            
            timestamp = self._get_current_timestamp()
            
            # 1. Calculate Dynamic Features (Groups A, E)
            f_a = self.guard_feature_engine.calculate_alpha_confidence(None, portfolio_state)
            
            # 2. Retrieve Precomputed Features (Groups B, C, D, F)
            if self.precomputed_features and asset in self.precomputed_features:
                f_market = self.precomputed_features[asset][self.current_step].tolist()
            else:
                f_market = [0.0] * 40 # Should not happen with current runner

            f_e = self.guard_feature_engine.calculate_execution_stats(None, trade_info, portfolio_state, current_atr=atr)
            
            # Combine all features (Order: A, B, C, D, E, F)
            all_features = f_a + f_market[:30] + f_e + f_market[30:]
            
            self.signals.append({
                'timestamp': timestamp,
                'asset': asset,
                'direction': direction,
                'label': 1 if outcome['pnl'] > 0 else 0,
                'features': all_features,
                'outcome_pnl': outcome['pnl']
            })
        finally:
            # Restore the actual environment position
            self.positions[asset] = real_pos

    def _open_position(self, asset, direction, act, price, atr):
        # Base class handles real position opening logic
        super()._open_position(asset, direction, act, price, atr)


class LightweightDatasetEnv:
    """
    Lightweight environment for dataset generation that skips expensive TradingEnv preprocessing.
    Only implements the minimum needed to run the PPO model and record trade signals.
    """
    def __init__(self, df_dict, feature_engine, precomputed_features, stage=3):
        self.df_dict = df_dict
        self.guard_feature_engine = feature_engine
        self.precomputed_features = precomputed_features
        self.stage = stage
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Constants
        self.MIN_ATR_MULTIPLIER = 0.0001
        self.MAX_POS_SIZE_PCT = 0.50
        self.MAX_TOTAL_EXPOSURE = 0.60
        self.MIN_POSITION_SIZE = 0.1
        
        # Cache numpy arrays for fast access
        self._cache_data_arrays()
        
        # State tracking
        self.signals = []
        self.last_action = None
        self.asset_signal_history = {asset: [] for asset in self.assets}
        
        # Determine action dimensions
        if self.stage == 1:
            self.action_dim = 5
        elif self.stage == 2:
            self.action_dim = 10
        else:
            self.action_dim = 20
            
    def _cache_data_arrays(self):
        """Cache DataFrame columns as numpy arrays for performance."""
        # Find minimum length across all assets to prevent IndexErrors due to misalignment
        min_len = min(len(df) for df in self.df_dict.values())
        self.max_steps = min_len  # We can go up to the full length (handled by bounds checks)
        
        self.close_arrays = {}
        self.low_arrays = {}
        self.high_arrays = {}
        self.open_arrays = {}
        self.volume_arrays = {}
        
        for asset in self.assets:
            df = self.df_dict[asset]
            # Slice to min_len to ensure all arrays are perfectly aligned
            self.close_arrays[asset] = df['close'].values[:min_len].astype(np.float32)
            self.low_arrays[asset] = df['low'].values[:min_len].astype(np.float32)
            self.high_arrays[asset] = df['high'].values[:min_len].astype(np.float32)
            self.open_arrays[asset] = df['open'].values[:min_len].astype(np.float32)
            self.volume_arrays[asset] = df['volume'].values[:min_len].astype(np.float32)
            
        # Precompute ATR arrays efficiently
        self.atr_arrays = {}
        for asset in self.assets:
            high = self.high_arrays[asset]
            low = self.low_arrays[asset]
            close = self.close_arrays[asset]
            
            # Calculate ATR (14-period)
            tr = np.maximum(high[1:] - low[1:], 
                           np.maximum(np.abs(high[1:] - close[:-1]),
                                    np.abs(low[1:] - close[:-1]))) # Fixed np.maximum for tr
            atr = np.zeros(min_len)
            if min_len > 14:
                atr[14:] = pd.Series(tr).rolling(14).mean().values[13:]
                atr[:14] = atr[14]
            else:
                atr[:] = 0.0001
                
            self.atr_arrays[asset] = atr.astype(np.float32)
            
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.current_step = 200  # Start after indicator warmup
        self.equity = 10000.0
        self.start_equity = self.equity
        self.peak_equity = self.equity
        self.leverage = 100
        self.positions = {asset: None for asset in self.assets}
        self.all_trades = []
        self.completed_trades = []
        self.asset_signal_history = {asset: [] for asset in self.assets}
        
        return self._get_observation(), {}
    
    def set_last_action(self, action):
        self.last_action = action
        
    def step(self, action):
        """Execute one step."""
        self.completed_trades = []
        
        # Parse and execute trades
        parsed_actions = self._parse_action(action)
        self._execute_trades(parsed_actions)
        
        # Advance time
        self.current_step += 1
        
        # Check termination
        truncated = self.current_step >= self.max_steps
        terminated = False
        
        # Update peak equity
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # If truncated, return zeros observation (episode is over, obs won't be used)
        if truncated:
            return np.zeros(140, dtype=np.float32), 0.0, terminated, truncated, {}
        
        return self._get_observation(), 0.0, terminated, truncated, {}
    
    def _get_observation(self):
        """Build a simple observation vector for the model."""
        # Build a 140-feature observation (simplified version)
        obs = np.zeros(140, dtype=np.float32)
        
        # Bounds check
        arr_len = len(self.close_arrays[self.assets[0]])
        if self.current_step >= arr_len:
            return obs  # Return zeros if out of bounds
        
        for i, asset in enumerate(self.assets):
            base_idx = i * 25
            price = self.close_arrays[asset][self.current_step]
            atr = self.atr_arrays[asset][self.current_step]
            
            # Basic price features
            if self.current_step > 0:
                prev_price = self.close_arrays[asset][self.current_step - 1]
                obs[base_idx] = price
                obs[base_idx + 1] = (price - prev_price) / prev_price if prev_price != 0 else 0
            
            obs[base_idx + 3] = atr
            
            # Position state
            pos = self.positions[asset]
            if pos:
                obs[base_idx + 13] = 1  # has_position
                obs[base_idx + 14] = pos['size'] / self.equity if self.equity > 0 else 0
                
        # Global features
        obs[125] = self.equity / self.start_equity
        obs[126] = 1 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0
        
        return obs
    
    def _parse_action(self, action):
        """Parse action array into trading decisions."""
        parsed = {}
        for i, asset in enumerate(self.assets):
            if self.stage == 1:
                direction_raw = action[i]
                size_raw = 0.5
                sl_raw = 1.5
                tp_raw = 3.0
            elif self.stage == 2:
                base_idx = i * 2
                direction_raw = action[base_idx]
                size_raw = np.clip((action[base_idx + 1] + 1) / 2, 0, 1)
                sl_raw = 1.5
                tp_raw = 3.0
            else:
                base_idx = i * 4
                direction_raw = action[base_idx]
                size_raw = np.clip((action[base_idx + 1] + 1) / 2, 0, 1)
                sl_raw = np.clip((action[base_idx + 2] + 1) / 2 * 2.5 + 0.5, 0.5, 3.0)
                tp_raw = np.clip((action[base_idx + 3] + 1) / 2 * 3.5 + 1.5, 1.5, 5.0)
                
            parsed[asset] = {
                'direction': 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0),
                'size': size_raw,
                'sl_mult': sl_raw,
                'tp_mult': tp_raw
            }
        return parsed
    
    def _execute_trades(self, actions):
        """Execute trading decisions and record signals."""
        for asset, act in actions.items():
            direction = act['direction']
            
            # Update signal history
            self.asset_signal_history[asset].append(direction)
            if len(self.asset_signal_history[asset]) > 20:
                self.asset_signal_history[asset].pop(0)
            
            # Record any non-zero signal
            if direction != 0:
                price = self.close_arrays[asset][self.current_step]
                atr = self.atr_arrays[asset][self.current_step]
                self._record_signal(asset, direction, act, price, atr)
                
    def _calculate_persistence_reversal(self, asset, current_direction):
        history = self.asset_signal_history[asset]
        if len(history) < 2:
            return 1.0, 0.0
        
        count = 0
        for sig in reversed(history[:-1]):
            if sig == current_direction:
                count += 1
            else:
                break
        persistence = count + 1.0
        
        prev_signal = 0
        for sig in reversed(history[:-1]):
            if sig != 0:
                prev_signal = sig
                break
        
        reversal = 1.0 if prev_signal != 0 and prev_signal != current_direction else 0.0
        return persistence, reversal
    
    def _record_signal(self, asset, direction, act, price, atr):
        """Record a signal with features and outcome label."""
        # Calculate SL/TP
        atr_val = max(atr, price * self.MIN_ATR_MULTIPLIER)
        sl_dist = act['sl_mult'] * atr_val
        tp_dist = act['tp_mult'] * atr_val
        sl = price - (direction * sl_dist)
        tp = price + (direction * tp_dist)
        
        # Simulate outcome
        outcome = self._simulate_outcome(asset, direction, sl, tp)
        
        # Calculate features
        asset_idx = self.assets.index(asset)
        if self.stage == 1:
            base_idx = asset_idx
        elif self.stage == 2:
            base_idx = asset_idx * 2
        else:
            base_idx = asset_idx * 4
            
        persistence, reversal = self._calculate_persistence_reversal(asset, direction)
        
        # Group A: Alpha confidence features
        f_a = [
            self.last_action[base_idx] if self.last_action is not None else 0,
            abs(self.last_action[base_idx]) if self.last_action is not None else 0,
            0.0,  # std placeholder
            persistence,
            reversal,
            1 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0,
            sum(1 for p in self.positions.values() if p is not None),
            0.0,  # exposure placeholder
            0.5,  # win rate placeholder
            0.0   # pnl sum placeholder
        ]
        
        # Get precomputed market features (Groups B, C, D, F) with bounds check
        if self.precomputed_features and asset in self.precomputed_features:
            arr = self.precomputed_features[asset]
            if self.current_step < len(arr):
                f_market = arr[self.current_step].tolist()
            else:
                f_market = [0.0] * 40
        else:
            f_market = [0.0] * 40
            
        # Group E: Execution stats
        f_e = [
            0.0,  # entry_dist
            sl_dist / (atr_val + 1e-6),
            tp_dist / (atr_val + 1e-6),
            tp_dist / (sl_dist + 1e-6),
            0.0,  # position_value
            1 - (self.equity / self.peak_equity) if self.peak_equity > 0 else 0,
            0.0, 0.5, 0.5, 0.0  # placeholders
        ]
        
        # Combine features
        all_features = f_a + f_market[:30] + f_e + f_market[30:]
        
        # Get timestamp with bounds check
        df = self.df_dict[asset]
        if self.current_step < len(df):
            timestamp = df.index[self.current_step]
        else:
            timestamp = df.index[-1]  # Use last available timestamp
        
        self.signals.append({
            'timestamp': timestamp,
            'asset': asset,
            'direction': direction,
            'label': 1 if outcome['pnl'] > 0 else 0,
            'features': all_features,
            'outcome_pnl': outcome['pnl']
        })
        
    def _simulate_outcome(self, asset, direction, sl, tp):
        """Look ahead to determine trade outcome."""
        start_idx = self.current_step + 1
        end_idx = min(start_idx + 1000, len(self.low_arrays[asset]))
        
        if start_idx >= end_idx:
            return {'pnl': 0.0}
            
        lows = self.low_arrays[asset][start_idx:end_idx]
        highs = self.high_arrays[asset][start_idx:end_idx]
        entry_price = self.close_arrays[asset][self.current_step]
        
        if direction == 1:  # Long
            sl_hit_mask = lows <= sl
            tp_hit_mask = highs >= tp
        else:  # Short
            sl_hit_mask = highs >= sl
            tp_hit_mask = lows <= tp
            
        sl_hit = sl_hit_mask.any()
        tp_hit = tp_hit_mask.any()
        
        if sl_hit and tp_hit:
            first_sl = np.argmax(sl_hit_mask)
            first_tp = np.argmax(tp_hit_mask)
            exit_price = sl if first_sl <= first_tp else tp
        elif sl_hit:
            exit_price = sl
        elif tp_hit:
            exit_price = tp
        else:
            exit_price = self.close_arrays[asset][end_idx - 1]
            
        # Calculate PnL
        price_change = (exit_price - entry_price) * direction
        pnl = price_change / entry_price if entry_price != 0 else 0
        
        return {'pnl': pnl}


class DatasetGenerator:
    def __init__(self, data_dir="TradeGuard/data", n_jobs=-1):
        self.logger = self.setup_logging()
        self.data_dir = Path(data_dir) 
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.logger.info(f"DatasetGenerator initialized with {self.n_jobs} cores")

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger("TradeGuard.DatasetGenerator")

    def load_data(self):
        start_time = time.time()
        data = {}
        for asset in self.assets:
            for suffix in ["_5m.parquet", "_5m_2025.parquet"]:
                path = self.data_dir / f"{asset}{suffix}"
                if path.exists():
                    self.logger.info(f"Loading {asset} from {path}...")
                    data[asset] = pd.read_parquet(path)
                    break
        
        if data:
            self.logger.info(f"Step 1: Data loading complete in {time.time() - start_time:.2f}s")
        return data

    @staticmethod
    def precompute_market_features(df_dict, logger=None):
        """Precomputes Groups B, C, D, F for all assets using FAST vectorized operations."""
        import warnings
        warnings.filterwarnings('ignore')  # Suppress division warnings
        
        start_time = time.time()
        precomputed = {}
        
        for asset, df in df_dict.items():
            n = len(df)
            
            # Common components as numpy arrays for speed
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            open_p = df['open'].values.astype(np.float64)
            vol = df['volume'].values.astype(np.float64)
            
            range_val = high - low
            body = np.abs(close - open_p)
            
            # Fast ATR calculation (fully vectorized with pandas)
            tr = np.zeros(n)
            tr[1:] = np.maximum(high[1:] - low[1:], 
                               np.maximum(np.abs(high[1:] - close[:-1]),
                                         np.abs(low[1:] - close[:-1])))
            atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
            atr_14 = np.nan_to_num(atr_14, nan=0.0001)
            
            # Group B: News Proxies (fast rolling using cumsum trick)
            def fast_rolling_mean(arr, window):
                result = np.zeros(len(arr))
                cumsum = np.cumsum(arr)
                result[window-1:] = (cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])) / window
                result[:window-1] = result[window-1]
                return result
            
            def fast_rolling_std(arr, window):
                mean = fast_rolling_mean(arr, window)
                sq_mean = fast_rolling_mean(arr**2, window)
                result = np.sqrt(np.maximum(sq_mean - mean**2, 0))
                return result
            
            vol_avg_50 = fast_rolling_mean(vol, 50)
            vol_std_50 = fast_rolling_std(vol, 50)
            
            f11 = vol / (vol_avg_50 + 1e-6)
            f12 = (vol - vol_avg_50) / (vol_std_50 + 1e-6)
            f13 = range_val / (atr_14 + 1e-6)
            f14 = np.zeros(n)  # Placeholder for range quantile (skip slow quantile)
            f15 = body / (range_val + 1e-6)
            
            upper_wick = high - np.maximum(open_p, close)
            lower_wick = np.minimum(open_p, close) - low
            f16 = (upper_wick + lower_wick) / (body + 1e-6)
            
            f17 = np.zeros(n)
            f17[1:] = np.abs(open_p[1:] - close[:-1]) / (atr_14[1:] + 1e-6)
            
            vol_diff = np.zeros(n)
            vol_diff[1:] = vol[1:] - vol[:-1]
            vol_ma_10 = fast_rolling_mean(vol, 10)
            f18 = vol_diff / (vol_ma_10 + 1e-6)
            
            f19 = atr_14 / (close + 1e-6)
            f20 = np.zeros(n)  # Placeholder for ATR quantile
            
            # Group C: Market Regime (fast approximations)
            # ADX approximation using simple directional movement
            dm_plus = np.zeros(n)
            dm_minus = np.zeros(n)
            dm_plus[1:] = np.maximum(high[1:] - high[:-1], 0)
            dm_minus[1:] = np.maximum(low[:-1] - low[1:], 0)
            
            # Simple smoothed DI
            di_plus = fast_rolling_mean(dm_plus, 14) / (atr_14 + 1e-6) * 100
            di_minus = fast_rolling_mean(dm_minus, 14) / (atr_14 + 1e-6) * 100
            dx = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-6) * 100
            f21 = fast_rolling_mean(dx, 14)  # ADX approximation
            f22 = di_plus  # +DI
            f23 = di_minus  # -DI
            
            # Aroon approximation - fully vectorized using pandas
            rolling_high_25 = pd.Series(high).rolling(25, min_periods=1).max().values
            rolling_low_25 = pd.Series(low).rolling(25, min_periods=1).min().values
            
            f24 = (high - rolling_low_25) / (rolling_high_25 - rolling_low_25 + 1e-6) * 100  # Aroon up proxy
            f25 = (rolling_high_25 - low) / (rolling_high_25 - rolling_low_25 + 1e-6) * 100  # Aroon down proxy
            f26 = np.full(n, 0.5)  # Hurst placeholder
            
            net_change_10 = np.zeros(n)
            net_change_10[10:] = np.abs(close[10:] - close[:-10])
            close_diff = np.zeros(n)
            close_diff[1:] = np.abs(close[1:] - close[:-1])
            abs_sum_10 = fast_rolling_mean(close_diff, 10) * 10
            f27 = net_change_10 / (abs_sum_10 + 1e-6)
            
            # Fast slope approximation (linear regression using simple difference)
            f28 = np.zeros(n)
            f28[20:] = (close[20:] - close[:-20]) / 20 / (close[20:] + 1e-6) * 10000
            
            sma_200 = fast_rolling_mean(close, min(200, n))
            f29 = close / (sma_200 + 1e-6)
            
            # Bollinger Band Width approximation
            sma_20 = fast_rolling_mean(close, 20)
            std_20 = fast_rolling_std(close, 20)
            f30 = (std_20 * 4) / (sma_20 + 1e-6)  # BB width approximation
            
            # Group D: Session Edge (already fast - just need proper indexing)
            try:
                hours = df.index.hour.values
                minutes = df.index.minute.values
                dows = df.index.dayofweek.values
            except AttributeError:
                # Fallback for non-datetime index
                hours = np.zeros(n)
                minutes = np.zeros(n)
                dows = np.zeros(n)
            
            f31 = np.sin(2 * np.pi * hours / 24)
            f32 = np.cos(2 * np.pi * hours / 24)
            f33 = np.sin(2 * np.pi * dows / 7)
            f34 = np.cos(2 * np.pi * dows / 7)
            f35 = ((hours >= 7) & (hours < 16)).astype(np.float64)
            f36 = ((hours >= 12) & (hours < 21)).astype(np.float64)
            f37 = ((hours >= 0) & (hours < 9)).astype(np.float64)
            f38 = minutes / 60.0
            f39 = (f35 * f36)
            f40 = (hours * 60 + minutes) / 1440.0
            
            # Group F: Price Action
            f51 = (close > open_p).astype(np.float64)
            f52 = body / (atr_14 + 1e-6)
            f53 = upper_wick / (body + 1e-6)
            f54 = lower_wick / (body + 1e-6)
            f55 = np.full(n, 5.0)  # Placeholder
            
            # Rolling max/min for 20 periods - fully vectorized using pandas
            rolling_high_20 = pd.Series(high).rolling(20, min_periods=1).max().values
            rolling_low_20 = pd.Series(low).rolling(20, min_periods=1).min().values
            
            f56 = (rolling_high_20 - close) / (atr_14 + 1e-6)
            f57 = (close - rolling_low_20) / (atr_14 + 1e-6)
            f58 = np.zeros(n)
            f59 = np.zeros(n)
            f60 = np.zeros(n)
            
            # Combine all 40 features into numpy array
            combined = np.column_stack([
                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,  # B
                f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,  # C
                f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,  # D
                f51, f52, f53, f54, f55, f56, f57, f58, f59, f60   # F
            ])
            
            # Handle NaN/Inf
            combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
            
            precomputed[asset] = combined.astype(np.float32)
            
        print(f"Features precomputed for all assets in {time.time() - start_time:.2f}s")
        return precomputed

    @staticmethod
    def run_inference_chunk(data_dir, assets, model_path, start_idx, end_idx, stage):
        """Worker function for multiprocessing. Optimized to load only necessary data."""
        import os
        chunk_start = time.time()
        worker_id = os.getpid()
        print(f"[Worker {worker_id}] Starting: processing bars {start_idx} to {end_idx}")
        
        # Load data locally in the worker - only the chunk we need plus lookback
        data_path = Path(data_dir)
        LOOKBACK = 500  # Buffer for indicators that need history
        actual_start = max(0, start_idx - LOOKBACK)
        
        df_dict = {}
        for asset in assets:
            for suffix in ["_5m.parquet", "_5m_2025.parquet"]:
                p = data_path / f"{asset}{suffix}"
                if p.exists():
                    # Read only the rows we need
                    full_df = pd.read_parquet(p)
                    df_dict[asset] = full_df.iloc[actual_start:end_idx].copy()
                    break
        
        if not df_dict:
            print(f"[Worker {worker_id}] ERROR: No data found.")
            return []

        print(f"[Worker {worker_id}] Data loaded: {len(next(iter(df_dict.values())))} rows")

        # Precompute market features for just this chunk (with lookback)
        precomputed = DatasetGenerator.precompute_market_features(df_dict, logger=None)
        
        # Slice off the lookback portion - we only want start_idx:end_idx
        offset = start_idx - actual_start
        chunk_df_dict = {asset: df.iloc[offset:].copy() for asset, df in df_dict.items()}
        chunk_precomputed = {asset: arr[offset:] for asset, arr in precomputed.items()}
        
        print(f"[Worker {worker_id}] Features precomputed, loading model...")
        
        model = PPO.load(model_path, device='cpu')
        
        print(f"[Worker {worker_id}] Creating environment...")
        
        # Use the lightweight environment that skips heavy preprocessing
        env = LightweightDatasetEnv(
            df_dict=chunk_df_dict,
            feature_engine=FeatureEngine(),
            precomputed_features=chunk_precomputed,
            stage=stage
        )
        
        print(f"[Worker {worker_id}] Environment ready, starting inference...")
        
        obs, _ = env.reset()
        done = False
        steps = 0
        total_steps = len(next(iter(chunk_df_dict.values()))) - 1
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            env.set_last_action(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if steps % 20000 == 0:
                print(f"[Worker {worker_id}] Progress: {steps}/{total_steps} bars ({(steps/total_steps)*100:.1f}%)")
            
        print(f"[Worker {worker_id}] FINISHED: {total_steps} bars in {time.time() - chunk_start:.2f}s, collected {len(env.signals)} signals")
        return env.signals

    def run(self, model_path, output_path):
        run_start = time.time()
        # Initial load to get metadata
        df_dict = self.load_data()
        if not df_dict:
            if DataFetcherTraining:
                self.logger.info(f"No data found in {self.data_dir}. Attempting automatic download...")
                fetcher = DataFetcherTraining(output_dir=str(self.data_dir))
                fetcher.start()
                # Reload data after download
                df_dict = self.load_data()
            
            if not df_dict:
                self.logger.error(f"No data files found in {self.data_dir} after download attempt. Ensure assets match: {self.assets}")
                return

        # Detect stage
        self.logger.info("Step 3: Loading model and detecting stage...")
        model = PPO.load(model_path, device='cpu')
        action_dim = model.action_space.shape[0]
        stage = 1 if action_dim == 5 else (2 if action_dim == 10 else 3)
        del model # Free memory
        
        total_len = len(next(iter(df_dict.values())))
        inference_start = time.time()
        
        # SINGLE-THREADED MODE (for Kaggle or when n_jobs=1)
        if self.n_jobs == 1:
            self.logger.info(f"Step 4: Processing {total_len} bars in SINGLE-THREADED mode...")
            
            # Run the worker function directly without multiprocessing
            all_signals = DatasetGenerator.run_inference_chunk(
                str(self.data_dir), self.assets, model_path, 0, total_len, stage
            )
        else:
            # MULTI-THREADED MODE
            chunk_size = total_len // self.n_jobs
            chunks = [(str(self.data_dir), self.assets, model_path, i * chunk_size, (i + 1) * chunk_size if i < self.n_jobs - 1 else total_len, stage) 
                      for i in range(self.n_jobs)]
            
            self.logger.info(f"Step 4: Spawning {self.n_jobs} workers to process {total_len} bars...")
            
            # Use starmap to pass arguments
            with multiprocessing.Pool(self.n_jobs) as pool:
                results = pool.starmap(DatasetGenerator.run_inference_chunk, chunks)
                
            # Combine results from all workers
            all_signals = [sig for chunk_signals in results for sig in chunk_signals]
            
        self.logger.info(f"Step 5: Inference and Labeling complete in {time.time() - inference_start:.2f}s")
        self.logger.info(f"Step 6: Generated {len(all_signals)} total signals.")
        
        self.save_dataset(all_signals, output_path)
        self.logger.info(f"TOTAL RUNTIME: {time.time() - run_start:.2f}s")

    def save_dataset(self, signals, output_path):
        if not signals: return
        
        # Ensure output directory exists
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for s in signals:
            row = {'timestamp': s['timestamp'], 'asset': s['asset'], 'label': s['label']}
            for i, val in enumerate(s['features']): row[f'feature_{i}'] = val
            data.append(row)
        pd.DataFrame(data).to_parquet(output_path)
        self.logger.info(f"Saved dataset to {output_path}")

class FeatureEngine:
    """Refined FeatureEngine for precomputation and dynamic calculation."""
    def calculate_alpha_confidence(self, market_row, portfolio_state):
        f = [
            portfolio_state['asset_action_raw'],
            abs(portfolio_state['asset_action_raw']),
            np.std(portfolio_state['asset_recent_actions']),
            portfolio_state['asset_signal_persistence'],
            portfolio_state['asset_signal_reversal'],
            1 - (portfolio_state['equity'] / portfolio_state['peak_equity']) if portfolio_state['peak_equity'] > 0 else 0,
            portfolio_state['open_positions_count'],
            portfolio_state['total_exposure'] / portfolio_state['equity'] if portfolio_state['equity'] > 0 else 0
        ]
        # Win rate (10 bars)
        rt = portfolio_state['recent_trades']
        f.append(sum(1 for t in rt if t['pnl'] > 0) / len(rt) if rt else 0.5)
        f.append(sum(t['pnl'] for t in rt) if rt else 0)
        return f

    def calculate_risk_output(self, risk_params):
        """Calculates features from Risk Model output."""
        return [
            risk_params.get('sl_mult', 1.0),
            risk_params.get('tp_mult', 1.0),
            risk_params.get('risk_factor', 1.0)
        ]

    def calculate_news_proxies(self, df):
        idx = -1
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[idx]
        vol_avg = df['volume'].iloc[-50:].mean()
        candle_range = df['high'].iloc[idx] - df['low'].iloc[idx]
        body = abs(df['close'].iloc[idx] - df['open'].iloc[idx])
        
        return [
            df['volume'].iloc[idx] / vol_avg if vol_avg > 0 else 1.0,
            (df['volume'].iloc[idx] - vol_avg) / (df['volume'].iloc[-50:].std() + 1e-6),
            candle_range / (atr + 1e-6),
            1.0 if candle_range < (df['high'] - df['low']).iloc[-100:].quantile(0.1) else 0.0,
            body / (candle_range + 1e-6),
            (df['high'].iloc[idx] - max(df['open'].iloc[idx], df['close'].iloc[idx]) + 
             min(df['open'].iloc[idx], df['close'].iloc[idx]) - df['low'].iloc[idx]) / (body + 1e-6),
            abs(df['open'].iloc[idx] - df['close'].iloc[idx-1]) / (atr + 1e-6) if len(df)>1 else 0,
            (df['volume'].diff().iloc[idx]) / (df['volume'].iloc[-10:].mean() + 1e-6),
            atr / (df['close'].iloc[idx] + 1e-6), # Simplified volatility regime
            1.0 if atr < (AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()).iloc[-100:].quantile(0.05) else 0.0
        ]

    def calculate_market_regime(self, df):
        idx = -1
        adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        aroon_ind = ta.trend.AroonIndicator(df['high'], df['low'], window=25)
        
        y = df['close'].iloc[-20:].values
        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
        
        return [
            adx_ind.adx().iloc[idx], adx_ind.adx_pos().iloc[idx], adx_ind.adx_neg().iloc[idx],
            aroon_ind.aroon_up().iloc[idx], aroon_ind.aroon_down().iloc[idx],
            0.5, # Placeholder for Hurst (too slow)
            abs(df['close'].iloc[idx] - df['close'].iloc[idx-10]) / (df['close'].diff().abs().iloc[-10:].sum() + 1e-6),
            (slope / df['close'].iloc[idx]) * 10000,
            df['close'].iloc[idx] / (df['close'].rolling(200).mean().iloc[idx] + 1e-6),
            ta.volatility.BollingerBands(df['close']).bollinger_wband().iloc[idx]
        ]

    def calculate_session_edge(self, ts):
        h = ts.hour
        is_london, is_ny, is_asian = (7 <= h < 16), (12 <= h < 21), (0 <= h < 9)
        return [
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * ts.dayofweek / 7), np.cos(2 * np.pi * ts.dayofweek / 7),
            float(is_london), float(is_ny), float(is_asian), ts.minute / 60.0,
            1.0 if is_london and is_ny else 0.0, (h * 60 + ts.minute) / 1440.0
        ]

    def calculate_execution_stats(self, df, trade_info, portfolio_state, current_atr=None):
        if current_atr is None:
            # Calculate ATR if not provided (fallback for tests)
            from ta.volatility import AverageTrueRange
            current_atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[-1]
            
        sl_dist = abs(trade_info['entry_price'] - trade_info['sl'])
        tp_dist = abs(trade_info['entry_price'] - trade_info['tp'])
        
        entry_dist = 0.0
        if df is not None:
            entry_dist = abs(trade_info['entry_price'] - df['close'].iloc[-1])
        
        return [
            entry_dist / (current_atr + 1e-6),
            sl_dist / (current_atr + 1e-6),
            tp_dist / (current_atr + 1e-6),
            tp_dist / (sl_dist + 1e-6),
            portfolio_state.get('position_value', 0) / portfolio_state['equity'],
            1 - (portfolio_state['equity'] / portfolio_state['peak_equity']) if portfolio_state['peak_equity'] > 0 else 0,
            0.0, 0.5, 0.5, 0.0 # Placeholders for RSI, BB, MACD
        ]

    def calculate_price_action_context(self, df):
        idx = -1
        o, h, l, c = df['open'].iloc[idx], df['high'].iloc[idx], df['low'].iloc[idx], df['close'].iloc[idx]
        atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range().iloc[idx]
        body = abs(c - o)
        return [
            1.0 if c > o else 0.0, body / (atr + 1e-6),
            (h - max(o, c)) / (body + 1e-6), (min(o, c) - l) / (body + 1e-6),
            5.0, # consecutive_direction placeholder
            (df['high'].iloc[-20:].max() - c) / (atr + 1e-6),
            (c - df['low'].iloc[-20:].min()) / (atr + 1e-6),
            0.0, 0.0, 0.0 # EMA, Velocity, VPT placeholders
        ]

if __name__ == "__main__":
    import argparse
    import platform
    
    # Use 'fork' on Linux (Kaggle/Colab), 'spawn' on Windows
    # 'fork' works better in containerized environments like Kaggle
    if platform.system() != 'Windows':
        try:
            multiprocessing.set_start_method('fork', force=True)
        except RuntimeError:
            pass
    else:
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    parser = argparse.ArgumentParser(description="TradeGuard Dataset Generator")
    parser.add_argument("--model", type=str, default="Alpha/models/checkpoints/8.03.zip", help="Path to Alpha PPO model")
    parser.add_argument("--data", type=str, default="TradeGuard/data", help="Directory containing asset parquet files")
    parser.add_argument("--output", type=str, default="TradeGuard/data/guard_dataset.parquet", help="Output Parquet file")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs (-1 for all cores, 1 for single-threaded)")
    parser.add_argument("--single", action="store_true", help="Force single-threaded mode (use this for Kaggle)")
    
    args = parser.parse_args()
    
    # Force single-threaded if --single is specified
    n_jobs = 1 if args.single else args.jobs
    
    generator = DatasetGenerator(data_dir=args.data, n_jobs=n_jobs)
    generator.run(args.model, args.output)
    
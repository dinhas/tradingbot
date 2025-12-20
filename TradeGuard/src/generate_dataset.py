import logging
import os
import sys
import multiprocessing
from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import ta
from ta.volatility import AverageTrueRange

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

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
            if direction != 0:
                self._record_signal(asset, direction, act, current_prices[asset], atrs[asset])
        
        # Execute actual trades in the base environment (manages real positions/equity)
        super()._execute_trades(actions)

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
            
            recent_trades = [{'pnl': t['net_pnl']} for t in self.all_trades[-10:]]
            total_exposure = sum(pos['size'] for pos in self.positions.values() if pos is not None)
            
            portfolio_state = {
                'equity': self.equity,
                'peak_equity': self.peak_equity,
                'total_exposure': total_exposure,
                'open_positions_count': sum(1 for p in self.positions.values() if p is not None),
                'recent_trades': recent_trades,
                'asset_action_raw': self.last_action[base_idx] if self.last_action is not None else 0,
                'asset_recent_actions': [self.last_action[base_idx]] * 5 if self.last_action is not None else [0]*5,
                'asset_signal_persistence': 1.0,
                'asset_signal_reversal': 0.0,
                'position_value': position_size if (position_size := (self.equity * act['size'] * 0.5)) > 0 else 0
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


class DatasetGenerator:
    def __init__(self, n_jobs=-1):
        self.logger = self.setup_logging()
        self.data_dir = Path("data") 
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.n_jobs = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
        self.logger.info(f"DatasetGenerator initialized with {self.n_jobs} cores")

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger("TradeGuard.DatasetGenerator")

    def load_data(self):
        data = {}
        for asset in self.assets:
            for suffix in ["_5m.parquet", "_5m_2025.parquet"]:
                path = self.data_dir / f"{asset}{suffix}"
                if path.exists():
                    self.logger.info(f"Loading {asset} from {path}")
                    data[asset] = pd.read_parquet(path)
                    break
        return data

    def precompute_market_features(self, df_dict):
        """Precomputes Groups B, C, D, F for all assets using vectorized operations."""
        self.logger.info("Precomputing market features (B, C, D, F) for all assets...")
        precomputed = {}
        
        for asset, df in df_dict.items():
            self.logger.info(f"Vectorizing {asset} features...")
            
            # Common components
            high, low, close, vol = df['high'], df['low'], df['close'], df['volume']
            open_p = df['open']
            range_val = high - low
            body = (close - open_p).abs()
            
            # Indicators (vectorized)
            atr_14 = AverageTrueRange(high, low, close, window=14).average_true_range()
            adx_ind = ta.trend.ADXIndicator(high, low, close, window=14)
            aroon_ind = ta.trend.AroonIndicator(high, low, window=25)
            bb = ta.volatility.BollingerBands(close)
            sma_200 = close.rolling(200).mean()
            
            # Group B: News Proxies
            vol_avg_50 = vol.rolling(50).mean()
            vol_std_50 = vol.rolling(50).std()
            range_q10_100 = range_val.rolling(100).quantile(0.1)
            atr_q05_100 = atr_14.rolling(100).quantile(0.05)
            
            f11 = vol / (vol_avg_50 + 1e-6)
            f12 = (vol - vol_avg_50) / (vol_std_50 + 1e-6)
            f13 = range_val / (atr_14 + 1e-6)
            f14 = (range_val < range_q10_100).astype(float)
            f15 = body / (range_val + 1e-6)
            
            upper_wick = high - df[['open', 'close']].max(axis=1)
            lower_wick = df[['open', 'close']].min(axis=1) - low
            f16 = (upper_wick + lower_wick) / (body + 1e-6)
            
            f17 = (open_p - close.shift(1)).abs() / (atr_14 + 1e-6)
            f18 = vol.diff() / (vol.rolling(10).mean() + 1e-6)
            f19 = atr_14 / (close + 1e-6)
            f20 = (atr_14 < atr_q05_100).astype(float)
            
            # Group C: Market Regime
            f21 = adx_ind.adx()
            f22 = adx_ind.adx_pos()
            f23 = adx_ind.adx_neg()
            f24 = aroon_ind.aroon_up()
            f25 = aroon_ind.aroon_down()
            f26 = pd.Series(0.5, index=df.index) # Hurst Placeholder
            
            net_change_10 = (close - close.shift(10)).abs()
            abs_change_sum_10 = close.diff().abs().rolling(10).sum()
            f27 = net_change_10 / (abs_change_sum_10 + 1e-6)
            
            # Optimized Slope calculation (Rolling 20)
            def get_slope(y):
                if len(y) < 20: return 0.0
                return np.polyfit(np.arange(20), y, 1)[0]
            f28 = (close.rolling(20).apply(get_slope, raw=True) / (close + 1e-6)) * 10000
            
            f29 = close / (sma_200 + 1e-6)
            f30 = bb.bollinger_wband()
            
            # Group D: Session Edge (Vectorized Time Features)
            hours = df.index.hour
            minutes = df.index.minute
            dows = df.index.dayofweek
            
            f31 = np.sin(2 * np.pi * hours / 24)
            f32 = np.cos(2 * np.pi * hours / 24)
            f33 = np.sin(2 * np.pi * dows / 7)
            f34 = np.cos(2 * np.pi * dows / 7)
            f35 = ((hours >= 7) & (hours < 16)).astype(float)
            f36 = ((hours >= 12) & (hours < 21)).astype(float)
            f37 = ((hours >= 0) & (hours < 9)).astype(float)
            f38 = minutes / 60.0
            f39 = (f35 * f36).astype(float)
            f40 = (hours * 60 + minutes) / 1440.0
            
            # Group F: Price Action
            f51 = (close > open_p).astype(float)
            f52 = body / (atr_14 + 1e-6)
            f53 = upper_wick / (body + 1e-6)
            f54 = lower_wick / (body + 1e-6)
            f55 = pd.Series(5.0, index=df.index) # Placeholder
            f56 = (high.rolling(20).max() - close) / (atr_14 + 1e-6)
            f57 = (close - low.rolling(20).min()) / (atr_14 + 1e-6)
            f58 = pd.Series(0.0, index=df.index)
            f59 = pd.Series(0.0, index=df.index)
            f60 = pd.Series(0.0, index=df.index)
            
            # Combine all 40 precomputed features (B=10, C=10, D=10, F=10)
            combined = pd.concat([
                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, # B
                f21, f22, f23, f24, f25, f26, f27, f28, f29, f30, # C
                pd.Series(f31, index=df.index), pd.Series(f32, index=df.index), 
                pd.Series(f33, index=df.index), pd.Series(f34, index=df.index),
                pd.Series(f35, index=df.index), pd.Series(f36, index=df.index),
                pd.Series(f37, index=df.index), pd.Series(f38, index=df.index),
                pd.Series(f39, index=df.index), pd.Series(f40, index=df.index), # D
                f51, f52, f53, f54, f55, f56, f57, f58, f59, f60  # F
            ], axis=1).fillna(0).values
            
            precomputed[asset] = combined
            
        return precomputed

    def run_inference_chunk(self, model_path, df_dict, precomputed, start_idx, end_idx, stage):
        """Worker function for multiprocessing."""
        # Subset the data
        chunk_df_dict = {asset: df.iloc[start_idx:end_idx] for asset, df in df_dict.items()}
        chunk_precomputed = {asset: arr[start_idx:end_idx] for asset, arr in precomputed.items()}
        
        model = PPO.load(model_path, device='cpu')
        env = DatasetGenerationEnv(
            df_dict=chunk_df_dict,
            feature_engine=FeatureEngine(),
            precomputed_features=chunk_precomputed,
            data_dir=str(self.data_dir),
            stage=stage,
            data=chunk_df_dict
        )
        
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            env.set_last_action(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        return env.signals

    def run(self, model_path, output_path):
        df_dict = self.load_data()
        precomputed = self.precompute_market_features(df_dict)
        
        # Detect stage
        model = PPO.load(model_path, device='cpu')
        action_dim = model.action_space.shape[0]
        stage = 1 if action_dim == 5 else (2 if action_dim == 10 else 3)
        del model # Free memory
        
        # Split into chunks for multiprocessing
        total_len = len(next(iter(df_dict.values())))
        chunk_size = total_len // self.n_jobs
        chunks = [(i * chunk_size, (i + 1) * chunk_size if i < self.n_jobs - 1 else total_len) 
                  for i in range(self.n_jobs)]
        
        self.logger.info(f"Spawning {self.n_jobs} workers to process {total_len} bars...")
        
        with multiprocessing.Pool(self.n_jobs) as pool:
            worker_func = partial(self.run_inference_chunk, model_path, df_dict, precomputed, stage=stage)
            results = pool.starmap(worker_func, chunks)
            
        # Combine results
        all_signals = [sig for chunk_signals in results for sig in chunk_signals]
        self.logger.info(f"Generated {len(all_signals)} total signals.")
        
        self.save_dataset(all_signals, output_path)

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

    def calculate_execution_stats(self, df_ignored, trade_info, portfolio_state, current_atr):
        sl_dist = abs(trade_info['entry_price'] - trade_info['sl'])
        tp_dist = abs(trade_info['entry_price'] - trade_info['tp'])
        return [
            0.0, # entry_atr_distance placeholder
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
    # Use spawn for CUDA compatibility and cross-platform consistency
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="TradeGuard Dataset Generator")
    parser.add_argument("--model", type=str, default="Alpha/models/checkpoints/8.03.zip", help="Path to Alpha PPO model")
    parser.add_argument("--output", type=str, default="TradeGuard/data/guard_dataset.parquet", help="Output Parquet file")
    parser.add_argument("--jobs", type=int, default=-1, help="Number of parallel jobs")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(n_jobs=args.jobs)
    generator.run(args.model, args.output)
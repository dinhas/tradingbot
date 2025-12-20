import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
import ta
from ta.volatility import AverageTrueRange

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import TradingEnv safely - might fail in some test envs if deps missing
try:
    from Alpha.src.trading_env import TradingEnv
except ImportError:
    # Fallback or mock for testing if needed, though usually deps are present
    TradingEnv = object

def label_trade(entry_price: float, direction: int, sl: float, tp: float, 
                future_highs: list, future_lows: list, future_closes: list) -> int:
    """
    Simulates trade outcome using lookahead data.
    Returns: 1 (Win) or 0 (Loss)
    """
    for i in range(len(future_highs)):
        if direction == 1:  # Long
            if future_lows[i] <= sl:
                return 0  # Loss
            if future_highs[i] >= tp:
                return 1  # Win
        else:  # Short
            if future_highs[i] >= sl:
                return 0  # Loss
            if future_lows[i] <= tp:
                return 1  # Win
    
    # Timed out - check final P&L
    if not future_closes:
        return 0
    final_price = future_closes[-1]
    pnl = (final_price - entry_price) * direction
    return 1 if pnl > 0 else 0

class DatasetGenerationEnv(TradingEnv):
    """
    Extended TradingEnv that captures trade signals and their outcomes.
    """
    def __init__(self, df_dict, feature_engine, data_dir='data'):
        # Force is_training=False for deterministic behavior
        super().__init__(data_dir=data_dir, is_training=False)
        self.signals = []
        self.df_dict = df_dict
        self.guard_feature_engine = feature_engine
        self.last_action = None

    def _get_current_timestamp(self):
        """
        Returns the current timestamp from the data.
        """
        # Assuming all assets have same index or at least the first one is representative
        asset = self.assets[0]
        return self.df_dict[asset].index[self.current_step]

    def _simulate_trade_outcome_with_timing(self, asset: str) -> dict:
        """
        Looks ahead in data to determine trade outcome.
        """
        pos = self.positions[asset]
        if pos is None:
            return {'pnl': 0, 'bars_held': 0, 'exit_reason': 'none'}
            
        entry_price = pos['entry_price']
        direction = pos['direction']
        sl = pos['sl']
        tp = pos['tp']
        
        # Max hold time e.g., 4 hours = 48 bars of 5m
        max_bars = 48
        
        # Get future data
        start_idx = self.current_step + 1
        end_idx = min(start_idx + max_bars, len(self.df_dict[asset]))
        
        future_df = self.df_dict[asset].iloc[start_idx:end_idx]
        
        if future_df.empty:
            return {'pnl': 0, 'bars_held': 0, 'exit_reason': 'end_of_data'}
            
        highs = future_df['high'].values
        lows = future_df['low'].values
        closes = future_df['close'].values
        
        for i in range(len(highs)):
            if direction == 1: # Long
                if lows[i] <= sl:
                    return {'pnl': (sl - entry_price), 'bars_held': i+1, 'exit_reason': 'sl'}
                if highs[i] >= tp:
                    return {'pnl': (tp - entry_price), 'bars_held': i+1, 'exit_reason': 'tp'}
            else: # Short
                if highs[i] >= sl:
                    return {'pnl': (entry_price - sl), 'bars_held': i+1, 'exit_reason': 'sl'}
                if lows[i] <= tp:
                    return {'pnl': (entry_price - tp), 'bars_held': i+1, 'exit_reason': 'tp'}
                    
        # Timeout
        final_price = closes[-1]
        pnl = (final_price - entry_price) * direction
        return {'pnl': pnl, 'bars_held': len(closes), 'exit_reason': 'timeout'}

    def set_last_action(self, action):
        """Stores the raw action from the model for feature calculation."""
        self.last_action = action

    def _open_position(self, asset, direction, act, price, atr):
        """
        Override to capture signal details when a position is opened.
        """
        super()._open_position(asset, direction, act, price, atr)
        
        # Check if position was successfully opened
        if self.positions[asset] is not None:
            # Calculate Future Outcome (Label)
            outcome = self._simulate_trade_outcome_with_timing(asset)
            
            # --- Feature Calculation ---
            # 1. Prepare inputs
            # Group A & E need portfolio state
            total_exposure = sum(pos['size'] for pos in self.positions.values() if pos is not None)
            
            # Find the index of the asset in the original list
            asset_idx = self.assets.index(asset)
            
            # Stage 3 logic: 4 elements per asset
            base_idx = asset_idx * 4
            
            # Extract recent trades for win rate calculation
            recent_trades = [{'pnl': t['net_pnl']} for t in self.all_trades[-10:]]
            
            portfolio_state = {
                'equity': self.equity,
                'peak_equity': self.peak_equity,
                'total_exposure': total_exposure,
                'open_positions_count': sum(1 for p in self.positions.values() if p is not None),
                'recent_trades': recent_trades,
                'asset_action_raw': self.last_action[base_idx] if self.last_action is not None else 0,
                'asset_recent_actions': [self.last_action[base_idx]] * 5 if self.last_action is not None else [0]*5,
                'asset_signal_persistence': 1.0, # Simplified
                'asset_signal_reversal': 0.0,    # Simplified
                'position_value': self.positions[asset]['size']
            }
            
            trade_info = {
                'entry_price': price,
                'sl': self.positions[asset]['sl'],
                'tp': self.positions[asset]['tp'],
                'direction': direction
            }
            
            # Historical DF for Group B, C, F
            # Use 300 bars context
            hist_df = self.df_dict[asset].iloc[max(0, self.current_step-300):self.current_step+1]
            
            timestamp = self._get_current_timestamp()
            
            # Calculate groups
            f_a = self.guard_feature_engine.calculate_alpha_confidence(None, portfolio_state)
            f_b = self.guard_feature_engine.calculate_news_proxies(hist_df)
            f_c = self.guard_feature_engine.calculate_market_regime(hist_df)
            f_d = self.guard_feature_edge = self.guard_feature_engine.calculate_session_edge(timestamp)
            f_e = self.guard_feature_engine.calculate_execution_stats(hist_df, trade_info, portfolio_state)
            f_f = self.guard_feature_engine.calculate_price_action_context(hist_df)
            
            # Combine all features (Total 60)
            all_features = f_a + f_b + f_c + f_d + f_e + f_f
            
            # Capture Signal Info
            signal_data = {
                'timestamp': timestamp,
                'asset': asset,
                'direction': direction,
                'entry_price': price,
                'sl': self.positions[asset]['sl'],
                'tp': self.positions[asset]['tp'],
                'outcome_pnl': outcome['pnl'],
                'bars_held': outcome['bars_held'],
                'exit_reason': outcome['exit_reason'],
                'label': 1 if outcome['pnl'] > 0 else 0,
                'features': all_features,
                'step': self.current_step
            }
            self.signals.append(signal_data)

class DatasetGenerator:
    """
    Generates training dataset for TradeGuard LightGBM model.
    """
    def __init__(self):
        self.logger = self.setup_logging()
        self.data_dir = Path("data") 
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.logger.info("DatasetGenerator initialized")

    def setup_logging(self):
        """
        Sets up logging for the DatasetGenerator.
        """
        logger = logging.getLogger("TradeGuard.DatasetGenerator")
        # Only add handler if not already present to avoid duplicate logs
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_data(self):
        """
        Loads 5-minute OHLCV data for all defined assets.
        
        Returns:
            dict: Dictionary mapping asset names to pandas DataFrames.
        """
        data = {}
        self.logger.info("Loading historical data...")
        
        for asset in self.assets:
            file_path = self.data_dir / f"{asset}_5m.parquet"
            file_path_2025 = self.data_dir / f"{asset}_5m_2025.parquet"
            
            if file_path.exists():
                self.logger.info(f"Loading {asset} from {file_path}")
                try:
                    df = pd.read_parquet(file_path)
                    data[asset] = df
                except Exception as e:
                    self.logger.error(f"Failed to load {asset}: {e}")
            elif file_path_2025.exists():
                self.logger.info(f"Loading {asset} from {file_path_2025}")
                try:
                    df = pd.read_parquet(file_path_2025)
                    data[asset] = df
                except Exception as e:
                    self.logger.error(f"Failed to load {asset}: {e}")
            else:
                self.logger.warning(f"File not found: {file_path} or {file_path_2025}")
        
        return data

    def load_model(self, model_path):
        """
        Loads the trained Alpha PPO model.
        
        Args:
            model_path (str): Path to the model zip file.
            
        Returns:
            PPO: Loaded model or None if failed.
        """
        self.logger.info(f"Loading Alpha model from {model_path}...")
        path = Path(model_path)
        if not path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            model = PPO.load(path)
            self.logger.info("Alpha model loaded successfully.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None

    def generate_signals(self, model_path):
        """
        Runs the Alpha model inference loop to generate trade signals.
        
        Args:
            model_path (str): Path to the model.
            
        Returns:
            list: List of signal dictionaries.
        """
        model = self.load_model(model_path)
        if not model:
            return []
            
        self.logger.info("Starting inference loop...")
        
        # Load full DataFrames for feature calculation context
        df_dict = self.load_data()
        
        # Initialize custom environment
        env = DatasetGenerationEnv(
            df_dict=df_dict, 
            feature_engine=FeatureEngine(),
            data_dir=str(self.data_dir)
        )
        
        obs, _ = env.reset()
        done = False
        
        # Iterate until done
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            env.set_last_action(action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        self.logger.info(f"Inference complete. Generated {len(env.signals)} signals.")
        return env.signals

    def save_dataset(self, signals, output_path):
        """
        Saves the generated signals and features to a Parquet file.
        """
        if not signals:
            self.logger.warning("No signals to save.")
            return
            
        self.logger.info(f"Saving {len(signals)} signals to {output_path}...")
        
        # Convert list of dicts to DataFrame
        data = []
        for s in signals:
            row = {
                'timestamp': s['timestamp'],
                'asset': s['asset'],
                'direction': s['direction'],
                'label': s['label']
            }
            # Add features
            if 'features' in s:
                for i, val in enumerate(s['features']):
                    row[f'feature_{i}'] = val
            data.append(row)
            
        df = pd.DataFrame(data)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to Parquet
        df.to_parquet(output_path)
        self.logger.info("Dataset saved successfully.")

    def run(self, model_path, output_path):
        """
        Full pipeline: generate signals and save dataset.
        """
        signals = self.generate_signals(model_path)
        self.save_dataset(signals, output_path)
        self.logger.info("Pipeline complete.")

class FeatureEngine:
    """
    Calculates features for the TradeGuard model.
    """
    def __init__(self):
        pass

    def calculate_alpha_confidence(self, market_row, portfolio_state):
        """
        Calculates Alpha Model Confidence features (1-10).
        """
        features = []
        
        # Feature 1: alpha_action_raw
        features.append(portfolio_state['asset_action_raw'])
        
        # Feature 2: alpha_action_abs
        features.append(abs(portfolio_state['asset_action_raw']))
        
        # Feature 3: alpha_action_std
        features.append(np.std(portfolio_state['asset_recent_actions']))
        
        # Feature 4: alpha_signal_persistence
        features.append(portfolio_state['asset_signal_persistence'])
        
        # Feature 5: alpha_signal_reversal
        features.append(portfolio_state['asset_signal_reversal'])
        
        # Feature 6: alpha_portfolio_drawdown
        equity = portfolio_state['equity']
        peak_equity = portfolio_state['peak_equity']
        drawdown = 1 - (equity / peak_equity) if peak_equity > 0 else 0
        features.append(drawdown)
        
        # Feature 7: alpha_open_positions
        features.append(portfolio_state['open_positions_count'])
        
        # Feature 8: alpha_margin_usage
        margin_usage = portfolio_state['total_exposure'] / equity if equity > 0 else 0
        features.append(margin_usage)
        
        # Feature 9: alpha_recent_win_rate
        recent_trades = portfolio_state['recent_trades'][-10:]
        if not recent_trades:
            win_rate = 0.5 # Default if no trades
        else:
            wins = sum(1 for t in recent_trades if t['pnl'] > 0)
            win_rate = wins / len(recent_trades)
        features.append(win_rate)
        
        # Feature 10: alpha_recent_pnl
        pnl_sum = sum(t['pnl'] for t in recent_trades)
        features.append(pnl_sum)
        
        return features

    def calculate_risk_output(self, risk_params):
        """
        Calculates Risk Model Output features.
        """
        features = []
        features.append(risk_params.get('sl_mult', 1.0))
        features.append(risk_params.get('tp_mult', 1.0))
        features.append(risk_params.get('risk_factor', 1.0))
        return features

    def calculate_news_proxies(self, df):
        """
        Calculates Synthetic News Proxies features (11-20).
        Expects a DataFrame with at least 50 bars of history.
        """
        features = []
        # We assume the calculation is for the LAST bar in df
        idx = -1
        
        # Calculate ATR for range-based features
        atr_series = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        current_atr = atr_series.iloc[idx]
        
        # Feature 11: volume_ratio
        vol_50_avg = df['volume'].iloc[-50:].mean()
        features.append(df['volume'].iloc[idx] / vol_50_avg if vol_50_avg > 0 else 1.0)
        
        # Feature 12: volume_zscore
        vol_50 = df['volume'].iloc[-50:]
        vol_std = vol_50.std()
        features.append((df['volume'].iloc[idx] - vol_50_avg) / vol_std if vol_std > 0 else 0.0)
        
        # Feature 13: range_ratio
        candle_range = df['high'].iloc[idx] - df['low'].iloc[idx]
        features.append(candle_range / current_atr if current_atr > 0 else 1.0)
        
        # Feature 14: range_compression
        ranges = df['high'] - df['low']
        p10 = ranges.iloc[-100:].quantile(0.1)
        features.append(1.0 if candle_range < p10 else 0.0)
        
        # Feature 15: body_to_range
        body = abs(df['close'].iloc[idx] - df['open'].iloc[idx])
        features.append(body / candle_range if candle_range > 0 else 0.0)
        
        # Feature 16: wick_ratio
        upper_wick = df['high'].iloc[idx] - max(df['open'].iloc[idx], df['close'].iloc[idx])
        lower_wick = min(df['open'].iloc[idx], df['close'].iloc[idx]) - df['low'].iloc[idx]
        # Use a small epsilon to avoid division by zero if body is 0
        features.append((upper_wick + lower_wick) / (body + 0.0001))
        
        # Feature 17: gap_size
        if len(df) > 1:
            gap = abs(df['open'].iloc[idx] - df['close'].iloc[idx-1])
            features.append(gap / current_atr if current_atr > 0 else 0.0)
        else:
            features.append(0.0)
            
        # Feature 18: tick_surge
        vol_diff = df['volume'].diff().iloc[idx]
        vol_10_avg = df['volume'].iloc[-10:].mean()
        features.append(vol_diff / vol_10_avg if vol_10_avg > 0 else 0.0)
        
        # Feature 19: volatility_regime
        # ATR percentile rank (rolling 200)
        if len(atr_series) >= 200:
            rank = atr_series.iloc[-200:].rank(pct=True).iloc[idx]
            features.append(rank)
        else:
            features.append(0.5)
            
        # Feature 20: quiet_market
        q05 = atr_series.iloc[-100:].quantile(0.05)
        features.append(1.0 if current_atr < q05 else 0.0)
        
        return features

    def calculate_market_regime(self, df):
        """
        Calculates Market Regime features (21-30).
        Includes ADX, Aroon, Hurst, Efficiency Ratio, and Trend stats.
        """
        features = []
        idx = -1
        
        # Pre-calculate indicators
        
        # ADX (14)
        adx_ind = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        
        # Feature 21: ADX
        features.append(adx_ind.adx().iloc[idx])
        
        # Feature 22: DI+
        features.append(adx_ind.adx_pos().iloc[idx])
        
        # Feature 23: DI-
        features.append(adx_ind.adx_neg().iloc[idx])
        
        # Aroon (25)
        aroon_ind = ta.trend.AroonIndicator(high=df['high'], low=df['low'], window=25)
        
        # Feature 24: Aroon Up
        features.append(aroon_ind.aroon_up().iloc[idx])
        
        # Feature 25: Aroon Down
        features.append(aroon_ind.aroon_down().iloc[idx])
        
        # Feature 26: Hurst Exponent (Approximate over last 100 bars)
        # Simplified R/S analysis to avoid heavy computation
        try:
            ts = df['close'].iloc[-100:].values
            if len(ts) >= 100:
                lags = range(2, 20)
                # Calculate standard deviation of differences for each lag
                tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
                # Log-log plot
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = poly[0]
                # Clamp to 0-1 range
                features.append(max(0.0, min(1.0, hurst)))
            else:
                features.append(0.5)
        except:
            features.append(0.5)
            
        # Feature 27: Efficiency Ratio (Kaufman) - 10 period
        changes = df['close'].diff().abs().iloc[-10:]
        total_change = changes.sum()
        net_change = abs(df['close'].iloc[idx] - df['close'].iloc[idx-10])
        features.append(net_change / total_change if total_change > 0 else 0.0)
        
        # Feature 28: Linear Regression Slope (normalized) - 20 period
        y = df['close'].iloc[-20:].values
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        # Normalize slope by price to make it comparable
        features.append((slope / df['close'].iloc[idx]) * 10000)
        
        # Feature 29: Trend Bias (Price vs SMA 200)
        sma_200 = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
        if not pd.isna(sma_200.iloc[idx]):
            features.append(df['close'].iloc[idx] / sma_200.iloc[idx])
        else:
            features.append(1.0)
            
        # Feature 30: Bollinger Band Width
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        features.append(bb.bollinger_wband().iloc[idx])
        
        return features

    def calculate_session_edge(self, timestamp):
        """
        Calculates Session Edge features (31-40).
        Time-based cyclical features and session flags (UTC).
        """
        features = []
        
        # Ensure timestamp is pandas Timestamp
        ts = pd.Timestamp(timestamp)
        
        # Feature 31: hour_sin
        features.append(np.sin(2 * np.pi * ts.hour / 24))
        
        # Feature 32: hour_cos
        features.append(np.cos(2 * np.pi * ts.hour / 24))
        
        # Feature 33: dow_sin
        features.append(np.sin(2 * np.pi * ts.dayofweek / 7))
        
        # Feature 34: dow_cos
        features.append(np.cos(2 * np.pi * ts.dayofweek / 7))
        
        # Define Sessions (UTC) - Approximate
        # London: 07:00 - 16:00
        # NY: 12:00 - 21:00
        # Asian (Tokyo): 00:00 - 09:00
        
        h = ts.hour
        
        is_london = 1.0 if 7 <= h < 16 else 0.0
        is_ny = 1.0 if 12 <= h < 21 else 0.0
        is_asian = 1.0 if 0 <= h < 9 else 0.0
        
        # Feature 35: is_london_open
        features.append(is_london)
        
        # Feature 36: is_ny_open
        features.append(is_ny)
        
        # Feature 37: is_asian_open
        features.append(is_asian)
        
        # Feature 38: minute_progress
        features.append(ts.minute / 60.0)
        
        # Feature 39: market_overlap (London & NY)
        features.append(1.0 if is_london and is_ny else 0.0)
        
        # Feature 40: time_since_day_start (fractional day)
        features.append((h * 60 + ts.minute) / 1440.0)
        
        return features

    def calculate_execution_stats(self, df, trade_info, portfolio_state):
        """
        Calculates Execution Statistics features (41-50).
        """
        features = []
        idx = -1
        
        # Calculate ATR
        atr_series = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        current_atr = atr_series.iloc[idx]
        if current_atr == 0: current_atr = 0.0001
        
        # Feature 41: entry_atr_distance
        # How far has price moved in the last 12 bars?
        if len(df) >= 12:
            dist = abs(df['close'].iloc[idx] - df['open'].iloc[idx-11])
            features.append(dist / current_atr)
        else:
            features.append(0.0)
            
        # Feature 42: sl_distance_atr
        sl_dist = abs(trade_info['entry_price'] - trade_info['sl'])
        features.append(sl_dist / current_atr)
        
        # Feature 43: tp_distance_atr
        tp_dist = abs(trade_info['entry_price'] - trade_info['tp'])
        features.append(tp_dist / current_atr)
        
        # Feature 44: risk_reward_ratio
        features.append(tp_dist / sl_dist if sl_dist > 0 else 2.0)
        
        # Feature 45: position_size_pct
        equity = portfolio_state['equity']
        pos_value = portfolio_state.get('position_value', 0)
        features.append(pos_value / equity if equity > 0 else 0.0)
        
        # Feature 46: current_drawdown
        peak_equity = portfolio_state['peak_equity']
        drawdown = 1 - (equity / peak_equity) if peak_equity > 0 else 0
        features.append(drawdown)
        
        # Feature 47: spread_estimate (High-Low relative to price)
        price = df['close'].iloc[idx]
        range_val = df['high'].iloc[idx] - df['low'].iloc[idx]
        features.append(range_val / price if price > 0 else 0.0)
        
        # Feature 48: momentum_at_entry (RSI-14)
        rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi().iloc[idx]
        features.append(rsi / 100.0 if not pd.isna(rsi) else 0.5)
        
        # Feature 49: bb_position
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        bb_h = bb.bollinger_hband().iloc[idx]
        bb_l = bb.bollinger_lband().iloc[idx]
        if bb_h != bb_l:
            bb_pos = (price - bb_l) / (bb_h - bb_l)
            features.append(max(0.0, min(1.0, bb_pos)))
        else:
            features.append(0.5)
            
        # Feature 50: macd_histogram
        macd = ta.trend.MACD(df['close']).macd_diff().iloc[idx]
        features.append(macd / current_atr if not pd.isna(macd) else 0.0)
        
        return features

    def calculate_price_action_context(self, df):
        """
        Calculates Price Action Context features (51-60).
        """
        features = []
        idx = -1
        
        atr_series = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        current_atr = atr_series.iloc[idx]
        if current_atr == 0: current_atr = 0.0001
        
        o, h, l, c = df['open'].iloc[idx], df['high'].iloc[idx], df['low'].iloc[idx], df['close'].iloc[idx]
        body = abs(c - o)
        
        # Feature 51: candle_direction
        features.append(1.0 if c > o else 0.0)
        
        # Feature 52: candle_body_size
        features.append(body / current_atr)
        
        # Feature 53: upper_wick_size
        upper_wick = h - max(o, c)
        features.append(upper_wick / (body + 0.0001))
        
        # Feature 54: lower_wick_size
        lower_wick = min(o, c) - l
        features.append(lower_wick / (body + 0.0001))
        
        # Feature 55: consecutive_direction
        directions = (df['close'] > df['open']).astype(int)
        last_dir = directions.iloc[idx]
        count = 0
        for i in range(len(directions)-1, -1, -1):
            if directions.iloc[i] == last_dir:
                count += 1
            else:
                break
        features.append(float(count))
        
        # Feature 56: distance_from_high_20
        h20 = df['high'].iloc[-20:].max()
        features.append((h20 - c) / current_atr)
        
        # Feature 57: distance_from_low_20
        l20 = df['low'].iloc[-20:].min()
        features.append((c - l20) / current_atr)
        
        # Feature 58: ema_alignment (EMA9 vs EMA21)
        ema9 = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator().iloc[idx]
        ema21 = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator().iloc[idx]
        if not pd.isna(ema9) and not pd.isna(ema21):
            features.append((ema9 - ema21) / ema21)
        else:
            features.append(0.0)
            
        # Feature 59: price_velocity (last 5 bars)
        if len(df) >= 6:
            features.append((c - df['close'].iloc[idx-5]) / current_atr)
        else:
            features.append(0.0)
            
        # Feature 60: volume_price_trend
        vpt = ta.volume.VolumePriceTrendIndicator(df['close'], df['volume']).volume_price_trend().iloc[idx]
        # Normalize VPT if possible, otherwise raw (it's hard to normalize as it's cumulative)
        # We'll just use raw for now, LightGBM can handle it.
        features.append(vpt if not pd.isna(vpt) else 0.0)
        
        return features

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TradeGuard Dataset Generator")
    parser.add_argument("--model", type=str, default="Alpha/models/checkpoints/8.03.zip", help="Path to Alpha PPO model")
    parser.add_argument("--output", type=str, default="TradeGuard/data/guard_dataset.parquet", help="Output Parquet file")
    
    args = parser.parse_args()
    
    generator = DatasetGenerator()
    generator.run(args.model, args.output)
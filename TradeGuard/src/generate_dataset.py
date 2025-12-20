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

class DatasetGenerationEnv(TradingEnv):
    """
    Extended TradingEnv that captures trade signals and their outcomes.
    """
    def __init__(self, data_dir='data', stage=3):
        # Force is_training=False for deterministic behavior
        super().__init__(data_dir=data_dir, stage=stage, is_training=False)
        self.signals = []

    def _open_position(self, asset, direction, act, price, atr):
        """
        Override to capture signal details when a position is opened.
        """
        super()._open_position(asset, direction, act, price, atr)
        
        # Check if position was successfully opened
        if self.positions[asset] is not None:
            # Calculate Future Outcome (Label)
            outcome = self._simulate_trade_outcome_with_timing(asset)
            
            # Capture Signal Info
            signal_data = {
                'timestamp': self._get_current_timestamp(),
                'asset': asset,
                'direction': direction,
                'entry_price': price,
                'sl': self.positions[asset]['sl'],
                'tp': self.positions[asset]['tp'],
                'outcome_pnl': outcome['pnl'],
                'bars_held': outcome['bars_held'],
                'exit_reason': outcome['exit_reason'],
                'label': 1 if outcome['pnl'] > 0 else 0,
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

    def generate_signals(self, model_path, stage=3):
        """
        Runs the Alpha model inference loop to generate trade signals.
        
        Args:
            model_path (str): Path to the model.
            stage (int): Curriculum stage.
            
        Returns:
            list: List of signal dictionaries.
        """
        model = self.load_model(model_path)
        if not model:
            return []
            
        self.logger.info("Starting inference loop...")
        
        # Initialize custom environment
        # We pass str(self.data_dir) because TradingEnv expects a path string or object
        env = DatasetGenerationEnv(data_dir=str(self.data_dir), stage=stage)
        
        obs, _ = env.reset()
        done = False
        
        # Iterate until done
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            
        self.logger.info(f"Inference complete. Generated {len(env.signals)} signals.")
        return env.signals

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

if __name__ == "__main__":
    generator = DatasetGenerator()

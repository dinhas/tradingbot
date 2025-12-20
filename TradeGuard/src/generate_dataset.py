import logging
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

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

if __name__ == "__main__":
    generator = DatasetGenerator()

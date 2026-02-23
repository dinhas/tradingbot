import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from .feature_engine import FeatureEngine
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from RiskLayer.src.risk_config import RiskConfig, DEFAULT_RISK_CONFIG


class TradingEnv(gym.Env):
    """
    Trading environment for RL agent in Alpha Layer.

    Curriculum Stages:
        Stage 1: Direction only (5 outputs)
        Stage 2: Direction + Position sizing (10 outputs)
        Stage 3: Direction + Position sizing + SL/TP (20 outputs)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self, data_dir="data", stage=3, is_training=True, risk_config: RiskConfig = None
    ):
        super(TradingEnv, self).__init__()

        self.data_dir = data_dir
        self.stage = stage
        self.is_training = is_training
        self.assets = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD"]

        # Load Risk Configuration
        self.risk_config = risk_config or DEFAULT_RISK_CONFIG

        # Configuration Constants
        self.MAX_POS_SIZE_PCT = 0.50
        self.MAX_TOTAL_EXPOSURE = 0.60
        self.DRAWDOWN_LIMIT = 0.25
        self.MIN_POSITION_SIZE = 0.01
        self.MIN_ATR_MULTIPLIER = 0.0001
        self.REWARD_LOG_INTERVAL = 5000

        # Load Data
        self.data = self._load_data()
        self.feature_engine = FeatureEngine()
        self.raw_data, self.processed_data = self.feature_engine.preprocess_data(
            self.data
        )

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

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.feature_engine.observation_dim,), dtype=np.float32
        )

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
        self.master_obs_matrix = np.zeros(
            (n_steps, len(self.assets) * self.feature_engine.observation_dim), dtype=np.float32
        )

        # Use vectorized extraction for each asset
        for i, asset in enumerate(self.assets):
            obs_matrix = self.feature_engine.get_observation_vectorized(
                self.processed_data, asset
            )
            start = i * self.feature_engine.observation_dim
            end = (i + 1) * self.feature_engine.observation_dim
            self.master_obs_matrix[:, start:end] = obs_matrix

    def _cache_data_arrays(self):
        self.close_arrays = {
            a: self.raw_data[f"{a}_close"].values.astype(np.float32)
            for a in self.assets
        }
        self.low_arrays = {
            a: self.raw_data[f"{a}_low"].values.astype(np.float32) for a in self.assets
        }
        self.high_arrays = {
            a: self.raw_data[f"{a}_high"].values.astype(np.float32) for a in self.assets
        }
        self.atr_arrays = {
            a: self.raw_data[f"{a}_atr_14"].values.astype(np.float32)
            for a in self.assets
        }

    def _load_data(self):
        data = {}
        for asset in self.assets:
            # Check multiple possible paths
            paths = [
                f"{self.data_dir}/{asset}_5m.parquet",
                f"{self.data_dir}/{asset}_5m_2025.parquet",
                f"data/{asset}_5m.parquet",
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
                dates = pd.date_range(start="2024-01-01", periods=1000, freq="5min")
                df = pd.DataFrame(index=dates)
                for col in ["open", "high", "low", "close"]:
                    df[f"{asset}_{col}"] = 1.0
                df[f"{asset}_volume"] = 100
                df[f"{asset}_atr_14"] = 0.001

            data[asset] = df
        return data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.equity = 10000.0
        self.current_step = (
            500
            if not self.is_training
            else np.random.randint(500, self.max_steps - 288)
        )
        self.positions = {asset: None for asset in self.assets}
        self.completed_trades = []
        return self._get_observation(), {}

    def _get_observation(self):
        current_row = self.processed_data.iloc[self.current_step]
        # In multi-asset, we default to the first asset or a specific one set via helper
        asset = getattr(self, "current_asset", self.assets[0])
        return self.feature_engine.get_observation(current_row, {}, asset)

    def set_asset(self, asset):
        self.current_asset = asset

    def _get_current_timestamp(self):
        """Get timestamp for current step."""
        try:
            return self.processed_data.index[self.current_step]
        except (IndexError, KeyError):
            from datetime import datetime, timedelta

            base_time = datetime(2025, 1, 1)
            return base_time + timedelta(minutes=self.current_step * 5)

    def _get_current_spread(self, asset: str, price: float, atr: float) -> float:
        """Calculate realistic spread based on session and volatility."""
        timestamp = self._get_current_timestamp()
        hour_utc = timestamp.hour
        is_weekend = timestamp.weekday() >= 5

        return self.risk_config.get_spread(asset, price, atr, hour_utc, is_weekend)

    def _execute_trades(self, actions):
        """Execute trading decisions for all assets."""
        current_prices = self._get_current_prices()
        atrs = self._get_current_atrs()

        for asset, act in actions.items():
            direction = act["direction"]
            current_pos = self.positions[asset]
            price = current_prices[asset]
            atr = atrs[asset]

            if current_pos is None:
                if direction != 0:
                    self._open_position(asset, direction, act, price, atr)
            elif current_pos["direction"] == direction:
                pass
            elif direction != 0 and current_pos["direction"] != direction:
                self._close_position(asset, price)
                self._open_position(asset, direction, act, price, atr)
            elif direction == 0 and current_pos is not None:
                self._close_position(asset, price)

    def _open_position(self, asset, direction, act, price, atr):
        """Open a new position with realistic spread and SL buffer."""
        size = act["size"] * self.MAX_POS_SIZE_PCT * self.equity

        # Minimum position check
        if size < self.MIN_POSITION_SIZE:
            logging.warning(f"[{asset}] Trade rejected: size {size:.2f} < min {self.MIN_POSITION_SIZE}")
            return

        # Maximum position check
        size = min(size, self.equity * 0.5)

        # Global exposure check
        if not self._check_global_exposure(size):
            logging.warning(f"[{asset}] Trade rejected: global exposure limit reached")
            return

        # Handle zero ATR
        atr = max(atr, price * self.MIN_ATR_MULTIPLIER)

        # Get realistic spread
        spread = self._get_current_spread(asset, price, atr)

        # Apply spread to entry price (pay the spread - worse fill)
        # For BUY: we pay the ask (higher price), for SELL: we pay the bid (lower price)
        entry_price = price + (direction * spread / 2)

        # Calculate SL/TP distances
        sl_dist = act["sl_mult"] * atr
        tp_dist = act["tp_mult"] * atr

        # Add breathing room buffer to SL
        sl_buffer = self.risk_config.breathing_room.calculate_sl_buffer(spread, atr)
        sl_dist_with_buffer = sl_dist + sl_buffer

        # Calculate SL/TP levels
        sl = entry_price - (direction * sl_dist_with_buffer)
        tp = entry_price + (direction * tp_dist)

        self.positions[asset] = {
            "direction": direction,
            "entry_price": entry_price,
            "size": size,
            "sl": sl,
            "tp": tp,
            "entry_step": self.current_step,
            "sl_dist": sl_dist_with_buffer,
            "tp_dist": tp_dist,
            "spread_paid": spread,
        }

    def _close_position(self, asset, price):
        """Close position with exit spread."""
        pos = self.positions[asset]
        if pos is None:
            return

        # Get exit spread
        atr = self._get_current_atrs()[asset]
        spread = self._get_current_spread(asset, price, atr)

        # Apply spread to exit price (pay the spread - worse fill)
        # For closing BUY: we sell at bid (lower price), for closing SELL: we buy at ask (higher price)
        exit_price = price - (pos["direction"] * spread / 2)

        # Leveraged P&L calculation
        price_change_pct = (
            (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["direction"]
        )
        pnl = price_change_pct * (pos["size"] * self.leverage)

        self.equity += pnl

        # Total spread cost for reporting (spread is already in price adjustments)
        entry_spread = pos.get("spread_paid", spread)
        exit_spread = spread / 2
        notional_value = pos["size"] * self.leverage
        spread_cost = notional_value * (entry_spread + exit_spread) / pos["entry_price"]

        self.completed_trades.append(
            {
                "timestamp": self._get_current_timestamp(),
                "asset": asset,
                "pnl": pnl,
                "net_pnl": pnl - spread_cost,
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "size": pos["size"],
                "spread_entry": entry_spread,
                "spread_exit": exit_spread,
            }
        )
        self.positions[asset] = None

    def _get_current_prices(self):
        return {
            asset: self.close_arrays[asset][self.current_step] for asset in self.assets
        }

    def _get_current_atrs(self):
        return {
            asset: self.atr_arrays[asset][self.current_step] for asset in self.assets
        }

    def _check_global_exposure(self, new_position_size):
        """Check if adding position would exceed the exposure limit."""
        current_exposure = sum(
            pos["size"] for pos in self.positions.values() if pos is not None
        )
        total_allocated = current_exposure + new_position_size
        return total_allocated <= (self.equity * self.MAX_TOTAL_EXPOSURE)

    def _update_positions(self):
        """Check SL/TP with spread-adjusted exit prices."""
        current_prices = self._get_current_prices()
        current_atrs = self._get_current_atrs()

        for asset, pos in list(self.positions.items()):
            if pos is None:
                continue

            price = current_prices[asset]
            atr = current_atrs[asset]
            spread = self._get_current_spread(asset, price, atr)

            if pos["direction"] == 1:  # Long
                bid_price = price - spread / 2
                if bid_price <= pos["sl"]:
                    self._close_position(asset, pos["sl"])
                elif price >= pos["tp"]:
                    self._close_position(asset, pos["tp"])
            else:  # Short
                ask_price = price + spread / 2
                if ask_price >= pos["sl"]:
                    self._close_position(asset, pos["sl"])
                elif price <= pos["tp"]:
                    self._close_position(asset, pos["tp"])

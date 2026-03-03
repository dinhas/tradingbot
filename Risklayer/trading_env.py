import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import logging

from Risklayer.config import config
from Risklayer.execution_engine import ExecutionEngine
from Risklayer.reward_engine import RewardEngine
from Risklayer.feature_engineering import FeatureEngine

logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, asset: Optional[str] = 'EURUSD', signals_df: Optional[pd.DataFrame] = None):
        super().__init__()
        self.df = df
        self.asset = asset
        self.signals_df = signals_df
        self.execution_engine = ExecutionEngine()
        self.reward_engine = RewardEngine()
        self.feature_engine = FeatureEngine()

        # State: 30 features + ATR + Vol% + Equity% + DD% + Margin + PosState = 36
        self.observation_space = spaces.Box(low=-10, high=10, shape=(config.STATE_DIM,), dtype=np.float32)

        # Action: SL Multiplier, RR Ratio, Risk Percent
        self.action_space = spaces.Box(low=-1, high=1, shape=(config.ACTION_DIM,), dtype=np.float32)

        if self.asset:
            self.feature_cols = self.feature_engine.get_observation_cols(self.asset)

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.equity = config.INITIAL_EQUITY
        self.initial_equity = config.INITIAL_EQUITY
        self.max_equity = self.equity
        self.drawdown = 0.0
        self.position = None

        if self.signals_df is not None:
            # Pick a random signal row
            signal_idx = np.random.randint(0, len(self.signals_df))
            signal_row = self.signals_df.iloc[signal_idx]
            
            timestamp = signal_row.name
            try:
                # Find the location in the full history
                self.current_index = self.df.index.get_loc(timestamp)
                if isinstance(self.current_index, slice):
                    self.current_index = self.current_index.start
            except (KeyError, ValueError):
                self.current_index = np.random.randint(0, len(self.df) - 500)
            
            self.current_asset = signal_row['asset_name']
            self.feature_cols = self.feature_engine.get_observation_cols(self.current_asset)
        else:
            self.current_index = np.random.randint(500, max(501, len(self.df) - 5000))
            self.current_asset = self.asset or 'EURUSD'
            if not hasattr(self, 'feature_cols'):
                self.feature_cols = self.feature_engine.get_observation_cols(self.current_asset)

        self._ensure_signals_exist()
        return self._get_observation(), {}

    def _ensure_signals_exist(self):
        """Fetches meta/quality/direction for the current index."""
        row = self.df.iloc[self.current_index]
        
        if self.signals_df is not None:
            timestamp = self.df.index[self.current_index]
            mask = (self.signals_df.index == timestamp) & (self.signals_df['asset_name'] == self.current_asset)
            matches = self.signals_df[mask]
            if len(matches) > 0:
                self.current_meta = matches.iloc[0]['meta_score']
                self.current_qual = matches.iloc[0]['quality_score']
                self.current_pred_dir = matches.iloc[0]['pred_direction']
            else:
                self.current_meta = 0.0
                self.current_qual = 0.0
                self.current_pred_dir = 0
        elif 'meta_score' in self.df.columns:
            self.current_meta = row['meta_score']
            self.current_qual = row['quality_score']
            self.current_pred_dir = row.get('pred_direction', 1 if self.current_meta > 0.5 else -1)
        else:
            # Synthetic fallback
            self.current_meta = 0.5 + 0.1 * np.random.randn()
            self.current_qual = 0.3 + 0.05 * np.random.randn()
            self.current_pred_dir = 1 if self.current_meta > 0.5 else -1

    def _seek_next_signal(self):
        """Advances current_index until a signal is found."""
        if self.signals_df is not None:
            # Jumping logic handled in reset and step
            pass
        else:
            while self.current_index < len(self.df) - 1:
                self._ensure_signals_exist()
                if self.current_meta > config.META_SCORE_THRESHOLD and \
                   self.current_qual > config.QUALITY_SCORE_THRESHOLD:
                    break
                self.current_index += 1

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_index]
        alpha_features = row[self.feature_cols].values.astype(np.float32)

        atr_col = f"{self.current_asset}_atr_14" if f"{self.current_asset}_atr_14" in row else f"{self.current_asset}_atr"
        atr = row[atr_col] if atr_col in row else 0.0001
        vol_pct = row.get(f"{self.current_asset}_vol_percentile", 0.5)

        equity_pct = self.equity / self.initial_equity
        drawdown_pct = self.drawdown
        margin_usage = 0.0
        if self.position:
            contract_size = config.CONTRACT_SIZES.get(self.current_asset, 100000)
            notional = self.position['volume'] * contract_size * row[f"{self.current_asset}_close"]
            margin_usage = (notional / 100) / (self.equity + 1e-8)

        pos_state = 1.0 if self.position else 0.0
        obs = np.concatenate([alpha_features, [atr, vol_pct, equity_pct, drawdown_pct, margin_usage, pos_state]])
        return np.pad(obs, (0, max(0, config.STATE_DIM - len(obs))))[:config.STATE_DIM]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Action
        sl_mult = self._denormalize(action[0], config.SL_MULTIPLIER_MIN, config.SL_MULTIPLIER_MAX)
        rr_ratio = self._denormalize(action[1], config.RR_RATIO_MIN, config.RR_RATIO_MAX)
        risk_pct = self._denormalize(action[2], config.RISK_PERCENT_MIN, config.RISK_PERCENT_MAX)

        # 2. Entry
        row = self.df.iloc[self.current_index]
        atr_col = f"{self.current_asset}_atr_14" if f"{self.current_asset}_atr_14" in row else f"{self.current_asset}_atr"
        atr = row[atr_col] if atr_col in row else 0.0001
        self._ensure_signals_exist()
        side = 'long' if self.current_pred_dir > 0 else 'short'
        
        entry_price = self.execution_engine.get_entry_price(self.current_asset, row[f"{self.current_asset}_close"], side, atr)
        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr_ratio
        
        sl_price = entry_price - sl_dist if side == 'long' else entry_price + sl_dist
        tp_price = entry_price + tp_dist if side == 'long' else entry_price - tp_dist

        volume = (self.equity * risk_pct) / (sl_dist * config.CONTRACT_SIZES.get(self.current_asset, 100000) + 1e-8)
        self.position = {'side': side, 'entry_price': entry_price, 'sl': sl_price, 'tp': tp_price, 'volume': volume}

        reward = 0
        peak_col, valley_col = f"{self.current_asset}_peak_dist", f"{self.current_asset}_valley_dist"
        if peak_col in row:
            reward = self.reward_engine.calculate_structural_reward(row[peak_col], row[valley_col], tp_dist, sl_dist)

        # 3. Simulation
        trade_closed = False
        while not trade_closed and self.current_index < len(self.df) - 1:
            self.current_index += 1
            row = self.df.iloc[self.current_index]
            exit_price, _ = self.execution_engine.check_exit(
                self.current_asset, self.position['side'], self.position['entry_price'],
                self.position['sl'], self.position['tp'],
                row[f"{self.current_asset}_high"], row[f"{self.current_asset}_low"], row[f"{self.current_asset}_close"],
                row[atr_col] if atr_col in row else 0.0001
            )
            if exit_price:
                pnl = self.execution_engine.calculate_pnl(self.current_asset, self.position['side'], self.position['entry_price'], exit_price, self.position['volume'])
                self.equity += pnl
                self.max_equity = max(self.max_equity, self.equity)
                self.drawdown = (self.max_equity - self.equity) / (self.max_equity + 1e-8)
                margin_usage = (self.position['volume'] * config.CONTRACT_SIZES.get(self.current_asset, 100000) * exit_price / 100) / (self.equity + 1e-8)
                reward += self.reward_engine.calculate_trade_close_reward(pnl, self.initial_equity, self.drawdown, margin_usage)
                self.position = None
                trade_closed = True

        terminated = self.equity <= 0.02 * self.initial_equity or self.current_index >= len(self.df) - 50
        if terminated and self.equity <= 0.02 * self.initial_equity:
            reward += self.reward_engine.get_termination_penalty()

        if not terminated:
            if self.signals_df is not None:
                signal_idx = np.random.randint(0, len(self.signals_df))
                signal_row = self.signals_df.iloc[signal_idx]
                try:
                    self.current_index = self.df.index.get_loc(signal_row.name)
                    if isinstance(self.current_index, slice): self.current_index = self.current_index.start
                    self.current_asset = signal_row['asset_name']
                    self.feature_cols = self.feature_engine.get_observation_cols(self.current_asset)
                    self._ensure_signals_exist()
                except: terminated = True
            else:
                self._seek_next_signal()
                if self.current_index >= len(self.df) - 50: terminated = True

        return self._get_observation(), reward, terminated, False, {'equity': self.equity, 'drawdown': self.drawdown}

    def _denormalize(self, val: float, min_val: float, max_val: float) -> float:
        return (val + 1) * (max_val - min_val) / 2 + min_val

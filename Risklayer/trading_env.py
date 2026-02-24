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
    def __init__(self, df: pd.DataFrame, asset: str = 'EURUSD'):
        super().__init__()
        self.df = df
        self.asset = asset
        self.execution_engine = ExecutionEngine()
        self.reward_engine = RewardEngine()
        self.feature_engine = FeatureEngine()

        # State: 30 features + ATR + Vol% + Equity% + DD% + Margin + PosState = 36
        self.observation_space = spaces.Box(low=-10, high=10, shape=(config.STATE_DIM,), dtype=np.float32)

        # Action: SL Multiplier, RR Ratio, Risk Percent
        # Normalized to [-1, 1] for SAC
        self.action_space = spaces.Box(low=-1, high=1, shape=(config.ACTION_DIM,), dtype=np.float32)

        self.feature_cols = self.feature_engine.get_observation_cols(asset)

        # Initial State
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Handle small DataFrames for testing
        low_idx = min(500, len(self.df) // 10)
        high_idx = max(low_idx + 1, len(self.df) - (len(self.df) // 10))

        if len(self.df) > 5500:
            self.current_index = np.random.randint(500, len(self.df) - 5000)
        else:
            self.current_index = np.random.randint(low_idx, high_idx)

        self.equity = config.INITIAL_EQUITY
        self.initial_equity = config.INITIAL_EQUITY
        self.max_equity = self.equity
        self.drawdown = 0.0

        self.position = None # None or {'side': 'long', 'entry_price': ..., 'sl': ..., 'tp': ..., 'volume': ...}

        # Generate synthetic meta/quality scores if not in DF
        if 'meta_score' not in self.df.columns:
            # Correlate slightly with future 12-bar return
            future_ret = self.df[f"{self.asset}_close"].pct_change(12).shift(-12).fillna(0)
            self.df['meta_score'] = 0.5 + 0.1 * np.sign(future_ret) + 0.1 * np.random.randn(len(self.df))
            self.df['quality_score'] = 0.3 + 0.05 * np.random.randn(len(self.df))

        # Find first signal
        self._seek_next_signal()

        return self._get_observation(), {}

    def _seek_next_signal(self):
        """Advances current_index until a signal is found or end of data."""
        while self.current_index < len(self.df) - 1:
            row = self.df.iloc[self.current_index]
            if row['meta_score'] > config.META_SCORE_THRESHOLD and \
               row['quality_score'] > config.QUALITY_SCORE_THRESHOLD:
                break
            self.current_index += 1

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_index]

        # 1. Alpha features (30)
        alpha_features = row[self.feature_cols].values.astype(np.float32)

        # 2. ATR & Vol% (2)
        atr = row[f"{self.asset}_atr"]
        vol_pct = row[f"{self.asset}_vol_percentile"]

        # 3. Account metrics (4)
        equity_pct = self.equity / self.initial_equity
        drawdown_pct = self.drawdown
        margin_usage = 0.0 # Simplified
        if self.position:
            # Assume 1:100 leverage
            contract_size = config.CONTRACT_SIZES.get(self.asset, 100000)
            notional = self.position['volume'] * contract_size * row[f"{self.asset}_close"]
            margin_usage = (notional / 100) / self.equity

        pos_state = 1.0 if self.position else 0.0

        # Combine
        obs = np.concatenate([
            alpha_features,
            [atr, vol_pct, equity_pct, drawdown_pct, margin_usage, pos_state]
        ])

        # Ensure correct dimension
        if len(obs) > config.STATE_DIM:
            obs = obs[:config.STATE_DIM]
        elif len(obs) < config.STATE_DIM:
            obs = np.pad(obs, (0, config.STATE_DIM - len(obs)))

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Denormalize Action
        sl_mult = self._denormalize(action[0], config.SL_MULTIPLIER_MIN, config.SL_MULTIPLIER_MAX)
        rr_ratio = self._denormalize(action[1], config.RR_RATIO_MIN, config.RR_RATIO_MAX)
        risk_pct = self._denormalize(action[2], config.RISK_PERCENT_MIN, config.RISK_PERCENT_MAX)

        # 2. Enter Trade
        row = self.df.iloc[self.current_index]
        atr = row[f"{self.asset}_atr"]
        mid_price = row[f"{self.asset}_close"]

        # Determine side based on some logic (e.g. meta_score > 0.6 = long, else short? or just long for now)
        # To make it more interesting, let's use the meta_score direction
        side = 'long' if row['meta_score'] > 0.5 else 'short'

        entry_price = self.execution_engine.get_entry_price(self.asset, mid_price, side, atr)
        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr_ratio

        if side == 'long':
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # Position Sizing
        risk_amount = self.equity * risk_pct
        # sl_dist is in price units. Risk = sl_dist * volume * contract_size
        volume = risk_amount / (sl_dist * config.CONTRACT_SIZES.get(self.asset, 100000))

        self.position = {
            'side': side,
            'entry_price': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'volume': volume,
            'sl_dist': sl_dist,
            'tp_dist': tp_dist
        }

        # Structural Reward
        peak_dist = row[f"{self.asset}_peak_dist"]
        valley_dist = row[f"{self.asset}_valley_dist"]
        reward = self.reward_engine.calculate_structural_reward(peak_dist, valley_dist, tp_dist, sl_dist)

        # 3. Simulate Trade until exit
        trade_closed = False
        while not trade_closed and self.current_index < len(self.df) - 1:
            self.current_index += 1
            row = self.df.iloc[self.current_index]

            exit_price, reason = self.execution_engine.check_exit(
                self.asset, self.position['side'], self.position['entry_price'],
                self.position['sl'], self.position['tp'],
                row[f"{self.asset}_high"], row[f"{self.asset}_low"], row[f"{self.asset}_close"],
                row[f"{self.asset}_atr"]
            )

            if exit_price:
                pnl = self.execution_engine.calculate_pnl(
                    self.asset, self.position['side'], self.position['entry_price'],
                    exit_price, self.position['volume']
                )
                self.equity += pnl

                # Update Drawdown
                self.max_equity = max(self.max_equity, self.equity)
                self.drawdown = (self.max_equity - self.equity) / self.max_equity

                # Trade Reward
                margin_usage = (self.position['volume'] * config.CONTRACT_SIZES.get(self.asset, 100000) * exit_price / 100) / self.equity
                reward += self.reward_engine.calculate_trade_close_reward(
                    pnl, self.initial_equity, self.drawdown, margin_usage
                )

                self.position = None
                trade_closed = True

        # 4. Check Termination
        terminated = False
        if self.equity <= 0.02 * self.initial_equity:
            terminated = True
            reward += self.reward_engine.get_termination_penalty()

        if self.current_index >= len(self.df) - 50:
            terminated = True # End of data

        # 5. Seek Next Signal
        if not terminated:
            self._seek_next_signal()
            if self.current_index >= len(self.df) - 50:
                terminated = True

        return self._get_observation(), reward, terminated, False, {'equity': self.equity, 'drawdown': self.drawdown}

    def _denormalize(self, val: float, min_val: float, max_val: float) -> float:
        """[-1, 1] -> [min_val, max_val]"""
        return (val + 1) * (max_val - min_val) / 2 + min_val

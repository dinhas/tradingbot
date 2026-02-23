import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from .config import config
from .execution_engine import ExecutionEngine
from .reward_engine import RewardEngine
from .feature_engineering import FeatureEngineer
from .peak_labeling import StructuralLabeler
from .data_loader import DataLoader

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_dict: Dict[str, pd.DataFrame] = None):
        super(TradingEnv, self).__init__()

        if data_dict is None:
            loader = DataLoader()
            raw_data = loader.load_all_data()
            fe = FeatureEngineer()
            data_dict = fe.calculate_features(raw_data)
            labeler = StructuralLabeler()
            for asset in data_dict:
                data_dict[asset] = labeler.label_data(data_dict[asset])

        self.data_dict = data_dict
        self.assets = list(data_dict.keys())

        self.execution_engine = ExecutionEngine()
        self.reward_engine = RewardEngine()
        self.fe = FeatureEngineer()

        # Action Space: [SL_mult, RR_ratio, Risk_pct]
        # Normalized to [-1, 1] for SAC
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation Space: 30 alpha + ATR + VolPct + Equity% + DD% + Margin + PosState = 36
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)

        self.reset_env()

    def reset_env(self):
        self.equity = config.INITIAL_EQUITY
        self.max_equity = self.equity
        self.balance = self.equity
        self.current_asset = np.random.choice(self.assets)
        self.df = self.data_dict[self.current_asset]

        # Start at random index, leaving room for warmup and future lookahead
        self.current_step = np.random.randint(config.WARMUP_PERIOD, len(self.df) - 500)
        self.done = False

        # Account state
        self.max_drawdown = 0.0
        self.margin_usage = 0.0
        self.in_position = 0.0 # 0 or 1

        self.trades = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_env()
        return self._get_obs(), {}

    def _get_obs(self) -> np.array:
        row = self.df.iloc[self.current_step]

        # 30 Alpha features
        alpha_features = row[self.fe.alpha_feature_cols].values.astype(np.float32)

        # ATR & Vol Percentile
        atr = row['atr']
        vol_pct = row['vol_percentile']

        # Account features
        equity_pct = self.equity / config.INITIAL_EQUITY
        dd_pct = (self.max_equity - self.equity) / self.max_equity
        margin = self.margin_usage
        pos_state = float(self.in_position)

        obs = np.concatenate([
            alpha_features,
            [atr, vol_pct, equity_pct, dd_pct, margin, pos_state]
        ])
        return obs.astype(np.float32)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, bool, dict]:
        # Rescale actions from [-1, 1] to actual ranges
        sl_mult = self._rescale(action[0], -1, 1, config.SL_MULT_MIN, config.SL_MULT_MAX)
        rr_ratio = self._rescale(action[1], -1, 1, config.RR_RATIO_MIN, config.RR_RATIO_MAX)
        risk_pct = self._rescale(action[2], -1, 1, config.RISK_PCT_MIN, config.RISK_PCT_MAX)

        reward = 0.0

        # 1. Advance to next trade signal
        while self.current_step < len(self.df) - 1:
            row = self.df.iloc[self.current_step]

            # Check gate
            if row['meta_score'] > config.META_THRESHOLD and row['quality_score'] > config.QUALITY_THRESHOLD:
                # Trade Signal! Use action to open trade.
                reward = self._open_trade(sl_mult, rr_ratio, risk_pct)
                break

            self.current_step += 1

        # Check termination
        if self.equity <= config.INITIAL_EQUITY * (1 - config.TERMINATION_DRAWDOWN):
            self.done = True
            reward += self.reward_engine.get_termination_penalty()

        if self.current_step >= len(self.df) - 100:
            # Reached end of data for this asset, pick another one or loop
            self.current_asset = np.random.choice(self.assets)
            self.df = self.data_dict[self.current_asset]
            self.current_step = config.WARMUP_PERIOD

        obs = self._get_obs()
        return obs, reward, self.done, False, {"equity": self.equity, "drawdown": self.max_drawdown}

    def _open_trade(self, sl_mult, rr_ratio, risk_pct) -> float:
        row = self.df.iloc[self.current_step]
        atr = row['atr']
        mid_price = row['close']

        # Side: for this exercise, let's assume direction comes from alpha model.
        # Since RL doesn't see meta_score, it also doesn't see direction?
        # Actually, let's assume meta_score > 0 implies Long.
        # (In a real system, the Alpha model provides direction).
        # Let's say: meta_score > 0.55 usually means High Conviction.
        # But we need a direction. Let's use ret_1 sign as a proxy for direction signal if not provided.
        # Or better: Assume the gate is for LONG trades only for simplicity,
        # OR use a hidden direction. Let's use a hidden direction based on a simple trend.
        side = 'long' if row['ema9_dist'] > 0 else 'short'

        spread = config.SPREAD_Pips * 0.0001 # Assume 4 digits
        if 'JPY' in self.current_asset: spread *= 100
        elif 'XAU' in self.current_asset: spread = 0.4 # Gold spread

        entry_price = self.execution_engine.get_entry_price(mid_price, spread, side)

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
        # Volume in units
        volume_units = risk_amount / (sl_dist + 1e-9)
        lots = volume_units / 100_000.0
        if 'XAU' in self.current_asset: lots = volume_units / 100.0

        # Structural Reward
        reward = self.reward_engine.calculate_structural_reward(
            row['peak_distance'], row['valley_distance'], sl_dist, tp_dist
        )

        # Simulate Execution
        future_data = self.df.iloc[self.current_step + 1 : self.current_step + 200][['open', 'high', 'low', 'close']].values
        if len(future_data) == 0:
            return reward

        exit_price, is_closed, bars_held = self.execution_engine.calculate_trade_outcome(
            side, entry_price, sl_price, tp_price, atr, spread, future_data
        )

        # Calculate Profit
        if side == 'long':
            pips_profit = exit_price - entry_price
        else:
            pips_profit = entry_price - exit_price

        gross_profit = pips_profit * volume_units
        costs = self.execution_engine.calculate_costs(lots, entry_price, exit_price, self.current_asset)
        net_profit = gross_profit - costs

        # Update Account
        prev_equity = self.equity
        self.equity += net_profit
        self.max_equity = max(self.max_equity, self.equity)

        current_dd = (self.max_equity - self.equity) / self.max_equity
        dd_increment = max(0, current_dd - self.max_drawdown)
        self.max_drawdown = max(self.max_drawdown, current_dd)

        # Trade Reward
        reward += self.reward_engine.calculate_trade_reward(net_profit, dd_increment, config.INITIAL_EQUITY)

        # Advance current step
        self.current_step += bars_held

        return reward

    def _rescale(self, val, old_min, old_max, new_min, new_max):
        return (val - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

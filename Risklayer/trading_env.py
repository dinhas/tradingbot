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

class FastTradingEnv(gym.Env):
    def __init__(self, price_data: Dict[str, Dict[str, np.ndarray]], signal_data: Dict[str, np.ndarray]):
        super().__init__()
        self.price_data = price_data
        self.signal_data = signal_data

        # State: 36 features
        self.observation_space = spaces.Box(low=-10, high=10, shape=(config.STATE_DIM,), dtype=np.float32)
        # Action: SL Multiplier, RR Ratio, Risk Percent
        self.action_space = spaces.Box(low=-1, high=1, shape=(config.ACTION_DIM,), dtype=np.float32)

        self.execution_engine = ExecutionEngine()
        self.reward_engine = RewardEngine()

        # Current Signal State
        self.equity = config.INITIAL_EQUITY
        self.initial_equity = config.INITIAL_EQUITY
        self.max_equity = self.equity
        self.drawdown = 0.0

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if self.equity <= 0.02 * self.initial_equity:
            self.equity = config.INITIAL_EQUITY
            self.max_equity = self.equity
            self.drawdown = 0.0

        # Pick random signal
        self.signal_idx = np.random.randint(0, len(self.signal_data['indices']))
        self.current_asset = self.signal_data['assets'][self.signal_idx]
        self.current_global_idx = self.signal_data['indices'][self.signal_idx]

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        # Static part: alpha features + ATR + Vol % (32 values)
        obs_static = self.signal_data['obs_static'][self.signal_idx]

        # Dynamic part: Equity%, DD%, Margin, PosState (4 values)
        equity_pct = self.equity / self.initial_equity
        drawdown_pct = self.drawdown
        margin_usage = 0.0 # Will be 0 at entry
        pos_state = 0.0 # Signal is always entry point

        obs = np.zeros(config.STATE_DIM, dtype=np.float32)
        obs[:32] = obs_static
        obs[32] = equity_pct
        obs[33] = drawdown_pct
        obs[34] = margin_usage
        obs[35] = pos_state
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # 1. Action Denormalization
        sl_mult = (action[0] + 1) * (config.SL_MULTIPLIER_MAX - config.SL_MULTIPLIER_MIN) / 2 + config.SL_MULTIPLIER_MIN
        rr_ratio = (action[1] + 1) * (config.RR_RATIO_MAX - config.RR_RATIO_MIN) / 2 + config.RR_RATIO_MIN
        risk_pct = (action[2] + 1) * (config.RISK_PERCENT_MAX - config.RISK_PERCENT_MIN) / 2 + config.RISK_PERCENT_MIN

        # 2. Entry
        asset = self.current_asset
        idx = self.current_global_idx
        p_data = self.price_data[asset]

        atr = p_data['atr'][idx]
        close = p_data['close'][idx]
        pred_dir = self.signal_data['dir'][self.signal_idx]
        side = 'long' if pred_dir > 0 else 'short'

        # Entry logic (Fast version)
        spread = config.SPREADS.get(asset, 0.0)
        entry_price = close + (spread / 2) if side == 'long' else close - (spread / 2)

        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr_ratio
        sl_price = entry_price - sl_dist if side == 'long' else entry_price + sl_dist
        tp_price = entry_price + tp_dist if side == 'long' else entry_price - tp_dist

        # 3. Fast Vectorized Simulation
        # Lookahead from p_data
        lookahead = 2000
        start = idx + 1
        end = min(start + lookahead, len(p_data['high']))

        f_high = p_data['high'][start:end]
        f_low = p_data['low'][start:end]
        f_close = p_data['close'][start:end]

        if side == 'long':
            sl_hits = np.where(f_low <= sl_price)[0]
            tp_hits = np.where(f_high >= tp_price)[0]
        else:
            sl_hits = np.where(f_high >= sl_price)[0]
            tp_hits = np.where(f_low <= tp_price)[0]

        first_sl = sl_hits[0] if len(sl_hits) > 0 else 999999
        first_tp = tp_hits[0] if len(tp_hits) > 0 else 999999

        if first_sl == 999999 and first_tp == 999999:
            exit_price = f_close[-1] if len(f_close) > 0 else entry_price
        elif first_sl <= first_tp:
            exit_price = sl_price
        else:
            exit_price = tp_price

        # 4. Reward Calculation
        contract_size = config.CONTRACT_SIZES.get(asset, 100000)
        volume = (self.equity * risk_pct) / (sl_dist * contract_size + 1e-8)

        pnl = (exit_price - entry_price) * volume * contract_size if side == 'long' else (entry_price - exit_price) * volume * contract_size

        self.equity += pnl
        self.max_equity = max(self.max_equity, self.equity)
        self.drawdown = (self.max_equity - self.equity) / (self.max_equity + 1e-8)

        # Combined Reward
        reward = self.reward_engine.calculate_structural_reward(p_data['peak'][idx], p_data['valley'][idx], tp_dist, sl_dist)
        reward += self.reward_engine.calculate_trade_close_reward(pnl, self.initial_equity, self.drawdown, 0.0)

        terminated = self.equity <= 0.02 * self.initial_equity
        if terminated: reward += self.reward_engine.get_termination_penalty()

        # Always move to next random signal
        next_obs, _ = self.reset()

        return next_obs, reward, terminated, False, {'equity': self.equity, 'drawdown': self.drawdown, 'pnl': pnl}

class TradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, asset: Optional[str] = 'EURUSD', signals_df: Optional[pd.DataFrame] = None):
        super().__init__()
        self.df = df
        self.asset = asset
        self.signals_df = signals_df
        self.execution_engine = ExecutionEngine()
        self.reward_engine = RewardEngine()
        self.feature_engine = FeatureEngine()
        self.observation_space = spaces.Box(low=-10, high=10, shape=(config.STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(config.ACTION_DIM,), dtype=np.float32)
        if self.asset:
            self.feature_cols = self.feature_engine.get_observation_cols(self.asset)
        self.equity = config.INITIAL_EQUITY
        self.initial_equity = config.INITIAL_EQUITY
        self.max_equity = self.equity
        self.drawdown = 0.0
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if self.equity <= 0.02 * self.initial_equity:
            self.equity = config.INITIAL_EQUITY
            self.max_equity = self.equity
            self.drawdown = 0.0
        self.position = None
        if self.signals_df is not None:
            signal_idx = np.random.randint(0, len(self.signals_df))
            signal_row = self.signals_df.iloc[signal_idx]
            timestamp = signal_row.name
            try:
                self.current_index = self.df.index.get_loc(timestamp)
                if isinstance(self.current_index, slice): self.current_index = self.current_index.start
            except: self.current_index = np.random.randint(0, len(self.df) - 500)
            self.current_asset = signal_row['asset_name']
            self.feature_cols = self.feature_engine.get_observation_cols(self.current_asset)
        else:
            self.current_index = np.random.randint(500, max(501, len(self.df) - 5000))
            self.current_asset = self.asset or 'EURUSD'
            if not hasattr(self, 'feature_cols'): self.feature_cols = self.feature_engine.get_observation_cols(self.current_asset)
        self._ensure_signals_exist()
        return self._get_observation(), {}

    def _ensure_signals_exist(self):
        row = self.df.iloc[self.current_index]
        if self.signals_df is not None:
            timestamp = self.df.index[self.current_index]
            mask = (self.signals_df.index == timestamp) & (self.signals_df['asset_name'] == self.current_asset)
            matches = self.signals_df[mask]
            if len(matches) > 0:
                self.current_meta = matches.iloc[0]['meta_score']
                self.current_qual = matches.iloc[0]['quality_score']
                self.current_pred_dir = matches.iloc[0]['pred_direction']
            else: self.current_meta, self.current_qual, self.current_pred_dir = 0.0, 0.0, 0
        elif 'meta_score' in self.df.columns:
            self.current_meta, self.current_qual, self.current_pred_dir = row['meta_score'], row['quality_score'], row.get('pred_direction', 1)
        else: self.current_meta, self.current_qual, self.current_pred_dir = 0.5, 0.3, 1

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[self.current_index]
        alpha_features = row[self.feature_cols].values.astype(np.float32)
        atr_col = f"{self.current_asset}_atr_14" if f"{self.current_asset}_atr_14" in row else f"{self.current_asset}_atr"
        atr = row[atr_col] if atr_col in row else 0.0001
        vol_pct = row.get(f"{self.current_asset}_vol_percentile", 0.5)
        equity_pct, drawdown_pct, margin_usage, pos_state = self.equity / self.initial_equity, self.drawdown, 0.0, 1.0 if self.position else 0.0
        obs = np.concatenate([alpha_features, [atr, vol_pct, equity_pct, drawdown_pct, margin_usage, pos_state]])
        return np.pad(obs, (0, max(0, config.STATE_DIM - len(obs))))[:config.STATE_DIM]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        sl_mult = (action[0] + 1) * (config.SL_MULTIPLIER_MAX - config.SL_MULTIPLIER_MIN) / 2 + config.SL_MULTIPLIER_MIN
        rr_ratio = (action[1] + 1) * (config.RR_RATIO_MAX - config.RR_RATIO_MIN) / 2 + config.RR_RATIO_MIN
        risk_pct = (action[2] + 1) * (config.RISK_PERCENT_MAX - config.RISK_PERCENT_MIN) / 2 + config.RISK_PERCENT_MIN
        row = self.df.iloc[self.current_index]
        atr_col = f"{self.current_asset}_atr_14" if f"{self.current_asset}_atr_14" in row else f"{self.current_asset}_atr"
        atr = row[atr_col] if atr_col in row else 0.0001
        self._ensure_signals_exist()
        side = 'long' if self.current_pred_dir > 0 else 'short'
        entry_price = self.execution_engine.get_entry_price(self.current_asset, row[f"{self.current_asset}_close"], side, atr)
        sl_dist, tp_dist = atr * sl_mult, atr * sl_mult * rr_ratio
        sl_price = entry_price - sl_dist if side == 'long' else entry_price + sl_dist
        tp_price = entry_price + tp_dist if side == 'long' else entry_price - tp_dist
        volume = (self.equity * risk_pct) / (sl_dist * config.CONTRACT_SIZES.get(self.current_asset, 100000) + 1e-8)
        reward = 0
        if f"{self.current_asset}_peak_dist" in row: reward = self.reward_engine.calculate_structural_reward(row[f"{self.current_asset}_peak_dist"], row[f"{self.current_asset}_valley_dist"], tp_dist, sl_dist)
        start, end = self.current_index + 1, min(self.current_index + 2001, len(self.df))
        future_df = self.df.iloc[start:end]
        if side == 'long':
            sl_hits, tp_hits = np.where(future_df[f"{self.current_asset}_low"].values <= sl_price)[0], np.where(future_df[f"{self.current_asset}_high"].values >= tp_price)[0]
        else:
            sl_hits, tp_hits = np.where(future_df[f"{self.current_asset}_high"].values >= sl_price)[0], np.where(future_df[f"{self.current_asset}_low"].values <= tp_price)[0]
        first_sl, first_tp = sl_hits[0] if len(sl_hits) > 0 else 999999, tp_hits[0] if len(tp_hits) > 0 else 999999
        if first_sl == 999999 and first_tp == 999999: exit_price = future_df[f"{self.current_asset}_close"].values[-1] if not future_df.empty else row[f"{self.current_asset}_close"]
        elif first_sl <= first_tp: exit_price = sl_price
        else: exit_price = tp_price
        pnl = self.execution_engine.calculate_pnl(self.current_asset, side, entry_price, exit_price, volume)
        self.equity += pnl
        self.max_equity = max(self.max_equity, self.equity)
        self.drawdown = (self.max_equity - self.equity) / (self.max_equity + 1e-8)
        reward += self.reward_engine.calculate_trade_close_reward(pnl, self.initial_equity, self.drawdown, 0.0)
        terminated = self.equity <= 0.02 * self.initial_equity or self.current_index >= len(self.df) - 50
        if terminated and self.equity <= 0.02 * self.initial_equity: reward += self.reward_engine.get_termination_penalty()
        if not terminated: self.reset()
        return self._get_observation(), reward, terminated, False, {'equity': self.equity, 'drawdown': self.drawdown, 'pnl': pnl}

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional
import logging

from .config import RiskConfig
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .peak_labeling import PeakLabeler
from .execution_engine import ExecutionEngine, TradeResult
from .reward_engine import RewardEngine

class RiskTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config: RiskConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Components
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.peak_labeler = PeakLabeler(config)
        self.execution_engine = ExecutionEngine(config)
        self.reward_engine = RewardEngine(config)

        # Load and process data
        self.data_dict = self.data_loader.load_all_data()
        self.processed_data = {}
        self.peak_labels = {}

        for asset, df in self.data_dict.items():
            df_feat = self.feature_engineer.calculate_features(df)
            p_dist, v_dist = self.peak_labeler.label_peaks_valleys(df_feat)
            df_feat['peak_dist'] = p_dist
            df_feat['valley_dist'] = v_dist

            # Simulate entry signals (for filtering only, not in observation)
            np.random.seed(self.config.SEED)
            df_feat['meta_score'] = np.random.uniform(0.3, 0.8, len(df_feat))
            df_feat['quality_score'] = np.random.uniform(0.1, 0.6, len(df_feat))

            self.processed_data[asset] = df_feat

        # Select primary asset for simplicity in this env (could be multi-asset)
        self.current_asset = self.config.ASSETS[0]
        self.df = self.processed_data[self.current_asset]

        # Action Space: [SL_multiplier, RR_ratio, Risk_percent]
        # Normalized to [-1, 1] for SAC
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation Space:
        # 30 alpha features + ATR + Vol Percentile + Equity% + DD% + Margin Usage + PositionState
        # Total = 30 + 1 + 1 + 1 + 1 + 1 + 1 = 36
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)

        # State variables
        self.equity = self.config.INITIAL_EQUITY
        self.max_equity = self.config.INITIAL_EQUITY
        self.current_step = self.config.WARMUP_PERIOD
        self.position_active = False
        self.last_trade_result: Optional[TradeResult] = None

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if self.config.RANDOM_START:
            self.current_step = np.random.randint(self.config.WARMUP_PERIOD, len(self.df) - 1000)
        else:
            self.current_step = self.config.WARMUP_PERIOD

        self.equity = self.config.INITIAL_EQUITY
        self.max_equity = self.config.INITIAL_EQUITY
        self.position_active = False

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]

        # 30 Alpha Features
        alpha_features = row[self.feature_engineer.alpha_feature_names].values.astype(np.float32)

        # ATR and Vol Percentile
        atr = row['atr']
        vol_pct = row['vol_percentile']

        # Portfolio Metrics
        equity_pct = self.equity / self.config.INITIAL_EQUITY
        drawdown = (self.max_equity - self.equity) / self.max_equity
        margin_usage = 0.0 # Simplified for this environment
        pos_state = 1.0 if self.position_active else 0.0

        obs = np.concatenate([
            alpha_features,
            [atr, vol_pct, equity_pct, drawdown, margin_usage, pos_state]
        ]).astype(np.float32)

        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 1. Decode Action
        # SAC outputs [-1, 1], scale to config bounds
        sl_mult = self._rescale_action(action[0], self.config.SL_MULT_MIN, self.config.SL_MULT_MAX)
        rr_ratio = self._rescale_action(action[1], self.config.RR_RATIO_MIN, self.config.RR_RATIO_MAX)
        risk_pct = self._rescale_action(action[2], self.config.RISK_PCT_MIN, self.config.RISK_PCT_MAX)

        reward = 0.0
        done = False
        truncated = False
        self.last_trade_result = None # Clear previous result

        # 2. Entry Filter (RL doesn't see these, but they gate execution)
        row = self.df.iloc[self.current_step]
        meta_score = row['meta_score']
        quality_score = row['quality_score']

        can_trade = meta_score > self.config.META_THRESHOLD and quality_score > self.config.QUALITY_THRESHOLD

        if can_trade and not self.position_active:
            # Open Trade
            self.position_active = True

            # Execution Logic
            atr = row['atr']
            entry_price = row['close']

            sl_dist = atr * sl_mult
            tp_dist = sl_dist * rr_ratio

            # Simulate direction for variety (usually from Alpha model)
            side = "long" if row['ret_1'] > 0 else "short"

            if side == "long":
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist

            # Position Sizing
            risk_amount = self.equity * risk_pct
            lots = risk_amount / (sl_dist * 100000.0) if sl_dist > 0 else 0

            # Structural Reward
            reward += self.reward_engine.calculate_structural_reward(
                row['peak_dist'], row['valley_dist'], tp_dist, sl_dist
            )

            # Resolve Trade (Look ahead)
            df_subset = self.df.iloc[self.current_step + 1 : self.current_step + 500]
            result = self.execution_engine.resolve_trade(entry_price, side, sl_price, tp_price, df_subset)

            # Costs
            costs = self.execution_engine.calculate_order_costs(lots, atr)
            result.commission = costs['commission']
            result.slippage_cost = costs['slippage']

            # Net PnL (pnl already adjusted for side in execution_engine)
            result.net_pnl = (result.pnl * lots * 100000.0) - result.commission - result.slippage_cost - (result.spread_cost * lots * 100000.0)

            # Update Equity
            prev_equity = self.equity
            self.equity += result.net_pnl
            self.max_equity = max(self.max_equity, self.equity)

            dd_increment = max(0, (self.max_equity - self.equity) / self.max_equity - (self.max_equity - prev_equity) / self.max_equity)

            reward += self.reward_engine.calculate_trade_reward(result, dd_increment, 0.0)

            self.position_active = False
            self.last_trade_result = result

        # 3. Step Forward
        self.current_step += 1
        if self.current_step >= len(self.df) - 501:
            truncated = True

        # 4. Check Termination (98% Drawdown)
        if self.equity <= self.config.INITIAL_EQUITY * self.config.TERMINATION_THRESHOLD:
            done = True
            reward += self.reward_engine.get_termination_penalty()

        return self._get_obs(), reward, done, truncated, {"net_pnl": self.last_trade_result.net_pnl if self.last_trade_result else 0}

    def _rescale_action(self, action_val: float, min_val: float, max_val: float) -> float:
        """Rescales [-1, 1] action to [min_val, max_val]."""
        return min_val + (action_val + 1.0) * 0.5 * (max_val - min_val)

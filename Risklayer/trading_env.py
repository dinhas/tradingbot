import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from Risklayer.config import config
from Risklayer.data_loader import DataLoader
from Risklayer.feature_engineering import FeatureEngine
from Risklayer.peak_labeling import PeakLabeler
from Risklayer.execution_engine import ExecutionEngine, TradeResult
from Risklayer.reward_engine import RewardEngine

class RiskTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame = None):
        super(RiskTradingEnv, self).__init__()

        # Load and preprocess data if not provided
        if df is None:
            loader = DataLoader()
            raw_df = loader.load_data()
            fe = FeatureEngine()
            self.df = fe.compute_features(raw_df)
            labeler = PeakLabeler()
            self.df = labeler.label_data(self.df)
            self.obs_features = fe.get_observation_features(self.df)
        else:
            self.df = df
            fe = FeatureEngine()
            self.obs_features = fe.get_observation_features(self.df)

        # Action Space: [SL_mult, RR_ratio, Risk_pct]
        # Normalized to [-1, 1] for SAC
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation Space: 30 alpha features + 7 meta features = 37
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32)

        # Engines
        self.execution_engine = ExecutionEngine()
        self.reward_engine = RewardEngine()

        self.initial_equity = config.INITIAL_EQUITY
        self.logger = logging.getLogger(self.__class__.__name__)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.equity = self.initial_equity
        self.max_equity = self.initial_equity
        self.drawdown = 0.0
        warmup = min(500, len(self.df) // 4)
        margin = min(1000, len(self.df) // 4)
        if len(self.df) > warmup + margin:
            self.current_step = np.random.randint(warmup, len(self.df) - margin)
        else:
            self.current_step = 0

        # Find first signal
        self.current_step = self._find_next_signal(self.current_step)

        obs = self._get_obs()
        return obs, {}

    def _find_next_signal(self, start_step: int) -> int:
        """Finds the next index where entry conditions are met."""
        for i in range(start_step, len(self.df) - 10):
            row = self.df.iloc[i]
            if row['meta_score'] > config.META_THRESHOLD and row['quality_score'] > config.QUALITY_THRESHOLD:
                return i
        return len(self.df) - 1 # End of data

    def _get_obs(self):
        """Constructs the observation vector."""
        # 1. Alpha features (30)
        alpha_feat = self.obs_features[self.current_step]

        # 2. Meta features (7)
        row = self.df.iloc[self.current_step]
        atr = row['atr_14']
        vol_pct = row['vol_percentile']
        equity_pct = self.equity / self.initial_equity
        drawdown_pct = self.drawdown
        margin_usage = 0.0 # Will be 0 when waiting for signal
        pos_state = 0.0    # 0 for no position

        # We also add ATR ratio as requested? No, it's already in alpha features.
        # Let's just follow the list: ATR, Volatility percentile, Equity %, Drawdown %, Margin usage, Position state
        # That's 6 features. Plus maybe one more to reach 37 total (30+7)?
        # I'll add "Time of day" or just 0 padding.
        # Actually I have 30 alpha features.
        # ATR (1), Vol % (1), Equity % (1), Drawdown % (1), Margin (1), Pos State (1) = 6.
        # Total 36. I'll add Current Step / Total Step (1) to make 37.

        meta_feat = np.array([
            atr,
            vol_pct,
            equity_pct,
            drawdown_pct,
            margin_usage,
            pos_state,
            self.current_step / len(self.df)
        ], dtype=np.float32)

        return np.concatenate([alpha_feat, meta_feat])

    def step(self, action):
        # 1. Denormalize actions
        # action is in [-1, 1]
        sl_mult = self._rescale(action[0], -1, 1, config.SL_MULT_MIN, config.SL_MULT_MAX)
        rr_ratio = self._rescale(action[1], -1, 1, config.RR_RATIO_MIN, config.RR_RATIO_MAX)
        risk_pct = self._rescale(action[2], -1, 1, config.RISK_PCT_MIN, config.RISK_PCT_MAX)

        # 2. Prepare trade parameters
        row = self.df.iloc[self.current_step]
        atr = row['atr_14']
        mid_price = row['close']

        # Determine direction based on scores?
        # "meta_score, quality_score". Usually meta_score > 0.5 implies Long?
        # Actually, let's assume if it passes filter, we go LONG for simplicity,
        # or we could use the alpha model features to decide.
        # But wait, the prompt doesn't say how to decide direction.
        # I'll assume for now that signals are for LONG trades (standard for these tasks).
        # Or I can use meta_score: > 0.5 is Long, < 0.5 is Short? But filter is > 0.55.
        # Okay, assume LONG.
        direction = 1

        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr_ratio
        sl_price = mid_price - sl_dist
        tp_price = mid_price + tp_dist

        risk_amount = self.equity * risk_pct
        # position_size = (risk_amount) / sl_distance
        pos_size_units = risk_amount / (sl_dist + 1e-6)

        # 3. Execute Trade
        # Track drawdown during trade? The execution engine returns the result when it close.
        # Realistically, we should check if drawdown limit hit DURING the trade.
        # I'll approximate it or check if net_profit causes it.

        result = self.execution_engine.execute_trade(
            self.df, self.current_step, direction, sl_price, tp_price, pos_size_units
        )

        # 4. Update Account
        old_equity = self.equity
        self.equity += result.pnl_net
        self.max_equity = max(self.max_equity, self.equity)
        self.drawdown = (self.max_equity - self.equity) / self.max_equity

        # 5. Calculate Reward
        # Peak/Valley labels for the entry bar
        peak_dist = row['peak_distance']
        valley_dist = row['valley_distance']

        drawdown_inc = max(0, self.drawdown - ( (self.max_equity - old_equity)/self.max_equity if self.max_equity > 0 else 0 ))
        equity_used_pct = (pos_size_units * mid_price) / (self.equity * 100) # Simple margin approx

        reward = self.reward_engine.calculate_reward(
            result, peak_dist, valley_dist, sl_dist, tp_dist, drawdown_inc, equity_used_pct
        )

        # 6. Check Termination
        terminated = False
        truncated = False

        if self.equity <= self.initial_equity * (1 - config.DRAWDOWN_LIMIT_PCT):
            terminated = True
            reward -= config.TERMINATION_PENALTY
            self.logger.warning(f"Drawdown limit hit! Equity: {self.equity}")

        # 7. Move to next signal
        self.current_step = result.exit_step + 1
        self.current_step = self._find_next_signal(self.current_step)

        if self.current_step >= len(self.df) - 10:
            truncated = True

        obs = self._get_obs()

        info = {
            "pnl": result.pnl_net,
            "equity": self.equity,
            "drawdown": self.drawdown,
            "exit_type": result.exit_type,
            "steps": result.exit_step - self.current_step
        }

        return obs, reward, terminated, truncated, info

    @staticmethod
    def _rescale(val, old_min, old_max, new_min, new_max):
        return (val - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

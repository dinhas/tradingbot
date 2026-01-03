import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RunningMeanStd:
    """Tracks running mean and std for normalization."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0] if hasattr(x, 'shape') and len(x.shape) > 0 else 1
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var + 1e-8)


class TradeGuardEnv(gym.Env):
    """
    RL Environment for TradeGuard.
    Iterates through a pre-generated dataset of Alpha signals and their outcomes.
    
    Features proper normalization to ensure stable PPO training:
    - Observation normalization (z-score on features)
    - Reward normalization (scaled to [-clip, clip] range)
    - PnL normalization using dataset statistics
    """
    def __init__(self, config):
        super(TradeGuardEnv, self).__init__()
        
        self.config = config
        self.dataset_path = config['env']['dataset_path']
        self.reward_scaling = config['env'].get('reward_scaling', 1.0)
        self.reward_clip = config['env'].get('reward_clip', 2.0)
        self.penalty_factors = config['env'].get('penalty_factors', {'missed_win': 0.5, 'loss_avoided': 0.1})
        
        # Load Dataset
        self.df = pd.read_parquet(self.dataset_path)
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Define Spaces - normalized observations
        self.observation_space = spaces.Box(low=-10, high=10, shape=(25,), dtype=np.float32)
        
        # Actions: 0 = Block, 1 = Allow
        self.action_space = spaces.Discrete(2)
        
        # Internal State
        self.current_step = 0
        self.total_steps = len(self.df)
        
        # ========== FEATURE NORMALIZATION ==========
        raw_features = self.df[[f'f_{i}' for i in range(25)]].values.astype(np.float32)
        
        # Compute statistics per feature
        self.feature_mean = np.nanmean(raw_features, axis=0)
        self.feature_std = np.nanstd(raw_features, axis=0)
        # Prevent division by zero
        self.feature_std = np.where(self.feature_std < 1e-6, 1.0, self.feature_std)
        
        # Z-score normalize features
        self.features = (raw_features - self.feature_mean) / self.feature_std
        # Clip extreme values
        self.features = np.clip(self.features, -10, 10)
        # Handle NaN/Inf
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        
        logger.info(f"Features normalized. Mean range: [{self.feature_mean.min():.4f}, {self.feature_mean.max():.4f}], "
                    f"Std range: [{self.feature_std.min():.4f}, {self.feature_std.max():.4f}]")
        
        # ========== PNL/REWARD NORMALIZATION ==========
        self.pnl_array = self.df['pnl'].values.astype(np.float32)
        self.target_array = self.df['target'].values.astype(np.float32)
        
        # Compute PnL statistics for normalization
        self.pnl_mean = np.nanmean(self.pnl_array)
        self.pnl_std = np.nanstd(self.pnl_array)
        if self.pnl_std < 1e-6:
            self.pnl_std = 1.0
        
        # Normalize PnL to roughly unit variance BUT DO NOT CENTER
        # We want to preserve the sign of PnL (Profit vs Loss).
        # Centering would make an average winning trade look like 0 return.
        self.pnl_normalized = self.pnl_array / self.pnl_std
        
        logger.info(f"PnL normalized (Scaled Only). Mean: {self.pnl_mean:.4f}, Std: {self.pnl_std:.4f}")
        logger.info(f"TradeGuardEnv initialized with {self.total_steps} samples (Normalized, 25 Features).")

        # Track episode statistics
        self.episode_rewards = []
        self.episode_length = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # Shuffle dataset by default for better training stability
        should_shuffle = options.get('shuffle', True) if options else True
        if should_shuffle:
            indices = np.random.permutation(self.total_steps)
            self.features = self.features[indices]
            self.pnl_normalized = self.pnl_normalized[indices]
            self.target_array = self.target_array[indices]
            
        self.current_step = 0
        self.episode_rewards = []
        self.episode_length = 0
        
        obs = self.features[self.current_step]
        info = {}
        
        return obs, info

    def step(self, action):
        pnl = self.pnl_normalized[self.current_step]
        target = self.target_array[self.current_step]
        
        # ========== REWARD CALCULATION ==========
        if action == 1:  # Allow
            # Reward is the normalized PnL
            reward = pnl
        else:  # Block
            if target == 1:
                # Missed a winning trade - penalty proportional to potential gain
                reward = -abs(pnl) * self.penalty_factors.get('missed_win', 0.5)
            else:
                # Correctly blocked a losing trade - smaller positive reward
                if pnl < 0:
                    reward = abs(pnl) * self.penalty_factors.get('loss_avoided', 0.1)
                else:
                    # Blocked a small winner (conservative) - no penalty or small negative
                    reward = -abs(pnl) * 0.1
        
        # Scale and clip reward
        reward = reward * self.reward_scaling
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        
        self.episode_rewards.append(reward)
        self.episode_length += 1
        self.current_step += 1
        
        terminated = self.current_step >= self.total_steps
        truncated = False
        
        if not terminated:
            obs = self.features[self.current_step]
        else:
            obs = np.zeros(25, dtype=np.float32)
            
        info = {
            'step': self.current_step,
            'raw_pnl': self.pnl_array[self.current_step - 1] if self.current_step > 0 else 0,
            'action': action,
            'target': target
        }
        
        if terminated:
            info['episode'] = {
                'r': np.sum(self.episode_rewards),
                'l': self.episode_length
            }
        
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def get_normalization_stats(self):
        """Return normalization statistics for inference."""
        return {
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'pnl_mean': self.pnl_mean,
            'pnl_std': self.pnl_std
        }

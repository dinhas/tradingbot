import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TradeGuardEnv(gym.Env):
    """
    RL Environment for TradeGuard.
    Iterates through a pre-generated dataset of Alpha signals and their outcomes.
    """
    def __init__(self, config):
        super(TradeGuardEnv, self).__init__()
        
        self.config = config
        self.dataset_path = config['env']['dataset_path']
        self.reward_scaling = config['env'].get('reward_scaling', 1.0)
        self.penalty_factors = config['env'].get('penalty_factors', {'missed_win': 0.5, 'loss_avoided': 0.1})
        
        # Load Dataset
        self.df = pd.read_parquet(self.dataset_path)
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Define Spaces
        # Observations: 105 features (f_0 to f_104)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(105,), dtype=np.float32)
        
        # Actions: 0 = Block, 1 = Allow
        self.action_space = spaces.Discrete(2)
        
        # Internal State
        self.current_step = 0
        self.total_steps = len(self.df)
        
        # Cache features for faster access
        self.features = self.df[[f'f_{i}' for i in range(105)]].values.astype(np.float32)
        
        # OPTIMIZATION: Cache PnL and Target columns as numpy arrays
        self.pnl_arrays = {}
        self.target_arrays = {}
        for a in self.assets:
            self.pnl_arrays[a] = self.df[f'pnl_{a}'].values.astype(np.float32)
            self.target_arrays[a] = self.df[f'target_{a}'].values.astype(np.float32)
        
        logger.info(f"TradeGuardEnv initialized with {self.total_steps} samples.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        obs = self.features[self.current_step]
        info = {}
        
        return obs, info

    def step(self, action):
        # OPTIMIZATION: Use numpy arrays instead of pandas iloc
        reward = 0.0
        
        if action == 1: # Allow
            # Reward is the sum of actual PnLs
            for a in self.assets:
                reward += self.pnl_arrays[a][self.current_step]
        else: # Block
            for a in self.assets:
                pnl = self.pnl_arrays[a][self.current_step]
                target = self.target_arrays[a][self.current_step]
                
                if target == 1:
                    # Missed Win Penalty
                    reward -= pnl * self.penalty_factors.get('missed_win', 0.5)
                else:
                    # Loss Avoided Incentive
                    if pnl < 0:
                        reward += abs(pnl) * self.penalty_factors.get('loss_avoided', 0.1)
        
        reward *= self.reward_scaling
        
        self.current_step += 1
        terminated = self.current_step >= self.total_steps
        truncated = False
        
        if not terminated:
            obs = self.features[self.current_step]
        else:
            obs = np.zeros(105, dtype=np.float32)
            
        info = {
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

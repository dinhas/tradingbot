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
        self.reward_mode = config['env'].get('reward_mode', 'pnl') # 'pnl' or 'binary'
        self.reward_scaling = config['env'].get('reward_scaling', 1.0)
        self.penalty_factors = config['env'].get('penalty_factors', {'missed_win': 0.5, 'loss_avoided': 0.1})
        
        # Load Dataset
        self.df = pd.read_parquet(self.dataset_path)
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Define Spaces
        # Observations: 25 features (f_0 to f_24)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32)
        
        # Actions: 0 = Block, 1 = Allow
        self.action_space = spaces.Discrete(2)
        
        # Internal State
        self.current_step = 0
        self.total_steps = len(self.df)
        
        # Cache features for faster access (f_0 to f_24)
        self.features = self.df[[f'f_{i}' for i in range(25)]].values.astype(np.float32)
        
        # OPTIMIZATION: Cache PnL and Target columns as numpy arrays
        # Now these are simple columns since each row is a single trade signal
        self.pnl_array = self.df['pnl'].values.astype(np.float32)
        self.target_array = self.df['target'].values.astype(np.float32)
        
        logger.info(f"TradeGuardEnv initialized with {self.total_steps} samples over {len(self.df)/len(self.assets):.1f} timeframes (approx).")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        # Randomize start point during training to avoid correlated parallel workers
        if self.total_steps > 1000:
             self.current_step = np.random.randint(0, self.total_steps - 512)
        else:
             self.current_step = 0
             
        obs = self.features[self.current_step]
        info = {}
        
        return obs, info

    def step(self, action):
        reward = 0.0
        
        pnl = self.pnl_array[self.current_step]
        target = self.target_array[self.current_step] # 1 = Win, 0 = Loss
        
        if self.reward_mode == 'binary':
            # Binary / Classification Reward (Clean signal)
            # Goal: Maximize Accuracy (Allow Wins, Block Losses)
            
            if action == 1: # Allow
                if target == 1: # True Positive (Allowed a Win)
                    reward = 1.0 
                else: # False Positive (Allowed a Loss)
                    reward = -1.0 
            else: # Block
                if target == 1: # False Negative (Blocked a Win)
                     reward = -0.5 # Smaller penalty for missing out? Or equal -1.0?
                else: # True Negative (Blocked a Loss)
                     reward = 0.5 # Reward for saving capital
                     
        else:
            # Original PnL-based Reward (Noisy)
            if action == 1: # Allow
                reward = pnl
            else: # Block
                if target == 1:
                    # Missed Win Penalty (using 1.0 as a base or config value)
                    reward = -pnl * self.penalty_factors.get('missed_win', 0.5)
                else:
                    # Loss Avoided Incentive
                    if pnl < 0:
                        reward = abs(pnl) * self.penalty_factors.get('loss_avoided', 0.1)
        
        reward *= self.reward_scaling
        
        self.current_step += 1
        terminated = self.current_step >= self.total_steps
        truncated = False
        
        if not terminated:
            obs = self.features[self.current_step]
        else:
            obs = np.zeros(25, dtype=np.float32)
            
        info = {
            'step': self.current_step
        }
        
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

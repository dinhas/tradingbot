
import os
import sys
import logging
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Setup logging to file
logging.basicConfig(filename='debug_risk_load.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

def debug_load():
    logger.info("Starting Debug Load...")
    
    risk_model_path = "models/checkpoints/risk/model10M.zip"
    risk_norm_path = "models/checkpoints/risk/model10M.pkl"
    
    # Check files
    if not os.path.exists(risk_model_path):
        logger.error(f"Risk Model file missing: {risk_model_path}")
    else:
        logger.info(f"Risk Model file exists: {risk_model_path}")
        
    if not os.path.exists(risk_norm_path):
        logger.error(f"Risk Norm file missing: {risk_norm_path}")
    else:
        logger.info(f"Risk Norm file exists: {risk_norm_path}")

    # 1. Load Risk Model
    try:
        logger.info("Attempting to load Risk Model (PPO)...")
        risk_model = PPO.load(risk_model_path, device='cpu')
        logger.info(f"✅ Risk Model Loaded. Obs Space: {risk_model.observation_space}")
    except Exception as e:
        logger.error(f"❌ Failed to load Risk Model: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # 2. Load Risk Normalizer
    try:
        logger.info("Attempting to load Risk Normalizer...")
        class SimpleRiskEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(165,))
                self.action_space = spaces.Box(low=-1, high=1, shape=(3,))
            def reset(self, seed=None): return np.zeros(165), {}
            def step(self, action): return np.zeros(165), 0, False, False, {}

        env = DummyVecEnv([lambda: SimpleRiskEnv()])
        norm_env = VecNormalize.load(risk_norm_path, env)
        logger.info(f"✅ Risk Normalizer Loaded. Obs Shape: {norm_env.observation_space.shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load Risk Normalizer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
    # 3. Load Alpha
    alpha_path = "models/checkpoints/alpha/ppo_final_model.zip"
    try:
        logger.info("Attempting to load Alpha Model...")
        alpha = PPO.load(alpha_path, device='cpu')
        logger.info(f"✅ Alpha Model Loaded. Obs Space: {alpha.observation_space}")
    except Exception as e:
        logger.error(f"❌ Failed to load Alpha Model: {e}")

if __name__ == "__main__":
    debug_load()

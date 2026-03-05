import os
import sys
import logging
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from risk_ppo_env import RiskPPOEnv
from risk_model_ppo import RiskActorCriticPolicy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PPO_DATASET_PATH = os.environ.get("PPO_DATASET_PATH")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hyperparameters
TOTAL_TIMESTEPS = 1_000_000
BATCH_SIZE = 2048
N_STEPS = 4096
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    logger.info(f"Starting PPO Risk Model Training on {DEVICE}")
    if PPO_DATASET_PATH:
        logger.info(f"Using pre-filtered dataset from {PPO_DATASET_PATH}")
    
    # 1. Create Environment
    def make_env():
        return RiskPPOEnv(data_dir=DATA_DIR, dataset_path=PPO_DATASET_PATH)
    
    env = DummyVecEnv([make_env])
    
    # Normalize observations and rewards for better RL convergence
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. Initialize Model
    # net_arch is ignored because our custom policy uses ResidualBlocks internally
    model = PPO(
        RiskActorCriticPolicy,
        env,
        verbose=1,
        device=DEVICE,
        tensorboard_log=LOG_DIR,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=MODELS_DIR,
        name_prefix="ppo_risk_model"
    )
    
    # 4. Train
    logger.info("Entering training loop...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name="PPO_Risk_Optimization"
    )
    
    # 5. Save Final Model
    model_path = os.path.join(MODELS_DIR, "ppo_risk_model_final.zip")
    stats_path = os.path.join(MODELS_DIR, "ppo_risk_vecnormalize.pkl")
    
    model.save(model_path)
    env.save(stats_path)
    
    logger.info(f"Training Complete. Model saved to {model_path}")
    logger.info(f"Normalization stats saved to {stats_path}")

if __name__ == "__main__":
    train()

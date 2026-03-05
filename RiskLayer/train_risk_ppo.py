import os
import sys
import logging
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

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
MARKET_DATA_DIR = os.environ.get("MARKET_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
PPO_DATASET_PATH = os.environ.get("PPO_DATASET_PATH")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# CONFIG B — Main Training (Recommended)
# ══════════════════════════════════════════════
CONFIG_B = dict(
    learning_rate       = 1e-4,       
    n_steps             = 4096,       
    batch_size          = 512,
    n_epochs            = 10,
    gamma               = 0.995,      
    gae_lambda          = 0.95,
    clip_range          = 0.15,
    ent_coef            = 0.005,      
    vf_coef             = 0.6,        
    max_grad_norm       = 0.5,
    
    # policy_kwargs managed in RiskActorCriticPolicy or passed here
    policy_kwargs = dict(
        activation_fn = torch.nn.Tanh,
    ),

    total_timesteps     = 3_000_000,
)

class TradingMetricsCallback(BaseCallback):
    """
    Callback for logging custom trading metrics from the environment to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(TradingMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Check if there's info from the environment
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0] # Take metrics from the first env
            # Only log if these keys exist (they are added in RiskPPOEnv.step)
            if "trade/win_rate" in info:
                for key, value in info.items():
                    if isinstance(value, (int, float, np.float32, np.float64)):
                        self.logger.record(f"env/{key}", value)
        return True

def train():
    logger.info(f"Starting PPO Risk Model Training (CONFIG B) on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 1. Create Parallel Environments
    env_kwargs = {'data_dir': MARKET_DATA_DIR, 'dataset_path': PPO_DATASET_PATH}
    env = make_vec_env(RiskPPOEnv, n_envs=4, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)
    
    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. Initialize Model
    model = PPO(
        RiskActorCriticPolicy,
        env,
        verbose=1,
        device="auto",
        tensorboard_log=LOG_DIR,
        **{k: v for k, v in CONFIG_B.items() if k != 'total_timesteps'}
    )
    
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=MODELS_DIR,
        name_prefix="ppo_risk_model_v2"
    )
    metrics_callback = TradingMetricsCallback()
    
    # 4. Train
    logger.info("Entering training loop...")
    model.learn(
        total_timesteps=CONFIG_B['total_timesteps'],
        callback=[checkpoint_callback, metrics_callback],
        tb_log_name="PPO_Risk_V2"
    )
    
    # 5. Save Final Model
    model_path = os.path.join(MODELS_DIR, "ppo_risk_model_final_v2.zip")
    stats_path = os.path.join(MODELS_DIR, "ppo_risk_vecnormalize_v2.pkl")
    
    model.save(model_path)
    env.save(stats_path)
    
    logger.info(f"Training Complete. Model saved to {model_path}")

if __name__ == "__main__":
    train()

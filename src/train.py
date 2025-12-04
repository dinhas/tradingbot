import os
import argparse
import logging
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from .trading_env import TradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_env(rank, seed=0, data_dir='data', stage=1):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param data_dir: (str) path to data directory
    :param stage: (int) curriculum stage
    """
    def _init():
        env = TradingEnv(data_dir=data_dir, stage=stage, is_training=True)
        env.reset(seed=seed + rank)
        return env
    return _init

class ValidationCallback(BaseCallback):
    """
    Callback for validating the model every N steps.
    """
    def __init__(self, eval_env, eval_freq=100000, verbose=1):
        super(ValidationCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            logger.info(f"Running validation at step {self.num_timesteps}")
            # Note: In a real scenario, we would run a full evaluation loop here
            # For now, we just log that we are validating
            pass
        return True

def get_ppo_config(stage):
    """
    Returns PPO configuration based on the curriculum stage.
    """
    # Base config from PRD 7.1
    config = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": "./logs/tensorboard"
    }

    # Stage-specific adjustments (PRD 7.1)
    if stage == 1:
        config["ent_coef"] = 0.02
    elif stage == 2:
        config["ent_coef"] = 0.01
    else:
        config["ent_coef"] = 0.005
        
    return config

def train(args):
    """
    Main training loop.
    """
    logger.info(f"Starting training for Stage {args.stage}")
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. Setup Vectorized Environment
    # Using 8 parallel environments as per PRD 7.2
    n_envs = 8 if not args.dry_run else 1
    
    if n_envs == 1:
        env = DummyVecEnv([make_env(0, data_dir=args.data_dir, stage=args.stage)])
    else:
        env = SubprocVecEnv([make_env(i, data_dir=args.data_dir, stage=args.stage) for i in range(n_envs)])
    
    # Apply Normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. Initialize Model
    ppo_config = get_ppo_config(args.stage)
    
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        # Note: If loading from a previous stage with different action space, 
        # we might need to handle partial loading or just use the weights for feature extraction.
        # For strict PPO loading, the architecture must match. 
        # If moving stages (5->10 actions), we typically start a fresh head.
        # For this implementation, we assume we are resuming same stage or starting fresh.
        model = PPO.load(args.load_model, env=env, **ppo_config)
    else:
        model = PPO("MlpPolicy", env, **ppo_config)
        
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // n_envs, 1), # Adjust freq for parallel envs
        save_path=args.checkpoint_dir,
        name_prefix=f"ppo_stage{args.stage}"
    )
    
    # 4. Train
    total_timesteps = args.total_timesteps if not args.dry_run else 1000
    
    logger.info(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[checkpoint_callback],
        progress_bar=True
    )
    
    # 5. Save Final Model
    final_path = os.path.join(args.checkpoint_dir, f"stage_{args.stage}_final")
    model.save(final_path)
    env.save(os.path.join(args.checkpoint_dir, f"stage_{args.stage}_vecnormalize.pkl"))
    
    logger.info(f"Training complete. Model saved to {final_path}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Trading Bot")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3], help="Curriculum stage (1, 2, or 3)")
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps to train")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Path to save checkpoints")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load existing model")
    parser.add_argument("--dry-run", action="store_true", help="Run a short test training loop")
    
    args = parser.parse_args()
    train(args)

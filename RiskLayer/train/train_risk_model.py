import yaml
import argparse
import os
import sys
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging

# Ensure project root is in path
sys.path.append(os.getcwd())

from RiskLayer.env.risk_env import RiskTradingEnv


class SaveNormalizerCallback(BaseCallback):
    """Callback to save VecNormalize stats periodically."""
    def __init__(self, save_path, save_freq=50000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            norm_path = os.path.join(self.save_path, f"vec_normalize_{self.n_calls}.pkl")
            self.training_env.save(norm_path)
            if self.verbose > 0:
                logging.info(f"Saved normalizer to {norm_path}")
        return True


def load_config(config_path="RiskLayer/config/ppo_config.yaml"):
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Map activation function string to torch class
    if 'policy_kwargs' in config and 'activation_fn' in config['policy_kwargs']:
        act_fn_str = config['policy_kwargs']['activation_fn']
        if act_fn_str == "Tanh":
            config['policy_kwargs']['activation_fn'] = nn.Tanh
        elif act_fn_str == "ReLU":
            config['policy_kwargs']['activation_fn'] = nn.ReLU
        elif act_fn_str == "LeakyReLU":
            config['policy_kwargs']['activation_fn'] = nn.LeakyReLU
            
    return config


def train(dry_run=False, steps_override=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = load_config()
    
    # Overrides
    total_timesteps = steps_override if steps_override else config.get('total_timesteps', 10_000_000)
    
    models_dir = "RiskLayer/models/checkpoints"
    log_dir = "RiskLayer/models/logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logging.info(f"Starting Training Session. Dry Run: {dry_run}")
    logging.info(f"Total Timesteps: {total_timesteps:,}")
    
    # 1. Initialize Environment
    # Use 5 parallel envs (one per asset for coverage)
    n_envs = 1 if dry_run else 5
    
    def make_env():
        return RiskTradingEnv(is_training=True)

    if dry_run:
        env = DummyVecEnv([make_env])
    else:
        env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Wrap with VecNormalize for observation normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    logging.info("Environment wrapped with VecNormalize")

    # 2. Initialize Model with config hyperparameters
    policy_kwargs = config.get('policy_kwargs', None)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=config.get('learning_rate', 0.0001),
        n_steps=config.get('n_steps', 4096),
        batch_size=config.get('batch_size', 256),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.995),
        gae_lambda=config.get('gae_lambda', 0.98),
        clip_range=config.get('clip_range', 0.15),
        ent_coef=config.get('ent_coef', 0.005),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs
    )
    
    logging.info(f"Model initialized with policy: {model.policy}")
    
    if dry_run:
        logging.info("Dry Run: Executing a few steps to verify loop...")
        try:
            model.learn(total_timesteps=2000)
            logging.info("Dry Run Complete. System appears functional.")
        except Exception as e:
            logging.error(f"Dry Run Failed: {e}")
            raise e
        finally:
            env.close()
        return

    # 3. Training Loop
    logging.info(f"Starting Full Training for {total_timesteps:,} steps...")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save model every 100k steps
        save_path=models_dir,
        name_prefix="risk_model_ppo"
    )
    
    normalizer_callback = SaveNormalizerCallback(
        save_path=models_dir,
        save_freq=100000,  # Save normalizer every 100k steps
        verbose=1
    )
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=[checkpoint_callback, normalizer_callback],
            progress_bar=True
        )
        
        # Save final model and normalizer
        model.save(f"{models_dir}/risk_model_final")
        env.save(f"{models_dir}/vec_normalize.pkl")
        logging.info(f"Training Complete. Model saved to {models_dir}/risk_model_final.zip")
        logging.info(f"Normalizer saved to {models_dir}/vec_normalize.pkl")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving current state...")
        model.save(f"{models_dir}/risk_model_interrupted")
        env.save(f"{models_dir}/vec_normalize_interrupted.pkl")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Risk Model (TradeGuard)")
    parser.add_argument("--dry-run", action="store_true", help="Run a short verification test instead of full training")
    parser.add_argument("--steps", type=int, default=0, help="Total training timesteps (overrides config)")
    
    args = parser.parse_args()
    
    steps = args.steps if args.steps > 0 else None
    train(dry_run=args.dry_run, steps_override=steps)

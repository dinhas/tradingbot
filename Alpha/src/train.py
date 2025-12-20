import os
from pathlib import Path
import argparse
import logging
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from .trading_env import TradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_env(rank, seed=0, data_dir='data'):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param data_dir: (str) path to data directory
    """
    def _init():
        # Environment is now fixed to 'Stage 3' logic internally
        env = TradingEnv(data_dir=data_dir, is_training=True)
        env.reset(seed=seed + rank)
        return env
    return _init

class MetricsCallback(BaseCallback):
    """
    Enhanced callback for logging detailed training metrics and creating visualizations.
    """
    def __init__(self, eval_freq=1000, plot_freq=10000, verbose=1):
        super(MetricsCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.plot_freq = plot_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps = []
        
    def _on_step(self) -> bool:
        # Log every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            # Get episode statistics from the environment
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                
                # Log to TensorBoard
                self.logger.record("train/episode_reward", ep_info.get("r", 0))
                self.logger.record("train/episode_length", ep_info.get("l", 0))
                
                # Store for plotting
                self.episode_rewards.append(ep_info.get("r", 0))
                self.episode_lengths.append(ep_info.get("l", 0))
                self.timesteps.append(self.num_timesteps)
                
                if self.verbose > 0:
                    logger.info(f"Step {self.num_timesteps}: Reward={ep_info.get('r', 0):.2f}, Length={ep_info.get('l', 0)}")
        
        # Create plots periodically
        if self.n_calls % self.plot_freq == 0 and len(self.episode_rewards) > 0:
            self._create_plots()
        
        return True
    
    def _create_plots(self):
        """Create and save training progress plots."""
        try:
            import matplotlib.pyplot as plt
            
            project_root = Path(__file__).resolve().parent.parent.parent
            plot_dir = project_root / "logs" / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Episode Rewards
            ax1.plot(self.timesteps, self.episode_rewards, alpha=0.6, label='Episode Reward')
            if len(self.episode_rewards) > 10:
                # Add moving average
                window = min(50, len(self.episode_rewards) // 10)
                import numpy as np
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                ax1.plot(self.timesteps[window-1:], moving_avg, 'r-', linewidth=2, label=f'{window}-Episode MA')
            ax1.set_xlabel('Timesteps')
            ax1.set_ylabel('Episode Reward')
            ax1.set_title('Training Progress: Episode Rewards')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Episode Lengths
            ax2.plot(self.timesteps, self.episode_lengths, alpha=0.6, color='green', label='Episode Length')
            ax2.set_xlabel('Timesteps')
            ax2.set_ylabel('Episode Length (steps)')
            ax2.set_title('Episode Lengths Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = plot_dir / f"training_progress_{self.num_timesteps}.png"
            plt.savefig(plot_path, dpi=100)
            plt.close()
            
            logger.info(f"Saved training plot to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")


def load_ppo_config(config_path):
    """
    Loads PPO configuration from a YAML file.
    Always uses the 'stage 3' overrides if they exist as the new default.
    """
    import yaml
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
        
    config = full_config['default'].copy()
    
    # Apply Stage 3 overrides as the final production state
    if 'stages' in full_config and 3 in full_config['stages']:
        config.update(full_config['stages'][3])
        
    # Handle policy_kwargs
    if 'policy_kwargs' in config:
        # Map activation function string to class
        activation_map = {
            "ReLU": nn.ReLU,
            "Tanh": nn.Tanh,
            "LeakyReLU": nn.LeakyReLU
        }
        
        act_fn_name = config['policy_kwargs'].get('activation_fn')
        if act_fn_name in activation_map:
            config['policy_kwargs']['activation_fn'] = activation_map[act_fn_name]
            
    return config

def train(args):
    """
    Main training loop.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir_path = project_root / args.log_dir
    checkpoint_dir_path = project_root / args.checkpoint_dir
    data_dir_path = project_root / args.data_dir
    config_path = project_root / args.config

    logger.info("Starting training (Final Curriculum State)")
    
    # Create directories
    log_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    
    import multiprocessing

    # 1. Setup Vectorized Environment
    # MAX SPEED: Use all available CPU cores with SubprocVecEnv
    n_cpu = multiprocessing.cpu_count()
    n_envs = n_cpu if not args.dry_run else 1
    
    logger.info(f"Creating {n_envs} environment(s) using SubprocVecEnv (maximizing CPU usage)...")
    
    # Use SubprocVecEnv for true parallelism
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, data_dir=data_dir_path) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, data_dir=data_dir_path)])
    
    # Apply Normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. Initialize Model
    ppo_config = load_ppo_config(config_path)
    
    # Extract total_timesteps from config before removing it
    config_timesteps = ppo_config.pop('total_timesteps', 1500000)
    
    if args.total_timesteps is None:
        args.total_timesteps = config_timesteps
        logger.info(f"Using total_timesteps from config: {args.total_timesteps}")
    
    if args.load_model:
        load_model_path = project_root / args.load_model
        logger.info(f"Loading model from {load_model_path}")
        try:
            model = PPO.load(load_model_path, env=env, **ppo_config)
            logger.info("✅ Model loaded successfully - continuing training")
        except Exception as e:
            logger.warning(f"⚠️ Could not load model: {e}")
            logger.info("Starting fresh model instead")
            model = PPO("MlpPolicy", env, **ppo_config)
    else:
        logger.info("Creating new model from scratch")
        model = PPO("MlpPolicy", env, **ppo_config)
        
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // n_envs, 1),
        save_path=str(checkpoint_dir_path),
        name_prefix="ppo_final"
    )
    
    metrics_callback = MetricsCallback(
        eval_freq=10000,
        plot_freq=10000,
        verbose=1
    )
    
    # 4. Train
    total_timesteps = args.total_timesteps if not args.dry_run else 1000
    
    # Configure file logging
    from datetime import datetime
    log_file = log_dir_path / f"train_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging to {log_file}")
    logger.info(f"Training for {total_timesteps} timesteps...")
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[checkpoint_callback, metrics_callback],
        progress_bar=False
    )
    
    # 5. Save Final Model
    final_path = checkpoint_dir_path / "ppo_final_model.zip"
    model.save(final_path)
    env.save(checkpoint_dir_path / "ppo_final_vecnormalize.pkl")
    
    logger.info(f"Training complete. Model saved to {final_path}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Trading Bot")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total timesteps to train")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory relative to project root")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to log directory relative to project root")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Path to save checkpoints relative to project root")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load existing model")
    parser.add_argument("--config", type=str, default="Alpha/config/ppo_config.yaml", help="Path to PPO configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Run a short test training loop")
    
    args = parser.parse_args()
    train(args)
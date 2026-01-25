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
from stable_baselines3.common.monitor import Monitor
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from .trading_env import TradingEnv
except (ImportError, ValueError):
    from trading_env import TradingEnv

try:
    from .curriculum_callback import CurriculumCallback
except (ImportError, ValueError):
    from curriculum_callback import CurriculumCallback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_env(rank, seed=0, data_dir='data'):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = TradingEnv(data_dir=data_dir, is_training=True)
        env = Monitor(env)
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
                
                # Log to SB3 Logger
                self.logger.record("train/episode_reward", ep_info.get("r", 0))
                self.logger.record("train/episode_length", ep_info.get("l", 0))
                self.logger.record("train/win_rate", ep_info.get("win_rate", 0))
                self.logger.record("train/fee_pct", ep_info.get("fee_pct", 0))
                self.logger.record("train/total_trades", ep_info.get("total_trades", 0))
                
                # Store for plotting
                self.episode_rewards.append(ep_info.get("r", 0))
                self.episode_lengths.append(ep_info.get("l", 0))
                self.timesteps.append(self.num_timesteps)
                
                if self.verbose > 0:
                    logger.info(f"Step {self.num_timesteps}: Reward={ep_info.get('r', 0):.2f}, WinRate={ep_info.get('win_rate', 0):.1%}, Fee={ep_info.get('fee_pct', 0):.3f}%")
            else:
                if self.verbose > 0:
                    logger.info(f"Step {self.num_timesteps}: No completed episodes yet...")
        
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
            # Save to a fixed filename (no timestamp/steps) as requested
            plot_path = plot_dir / "training_progress.png"
            plt.savefig(plot_path, dpi=100)
            plt.close()
            
            logger.info(f"Saved training plot to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")


def load_ppo_config(config_path):
    """
    Loads PPO configuration from a YAML file.
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
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

def get_lr_schedule(base_lr, total_timesteps):
    """
    Returns a learning rate schedule function.
    Reduces LR by 3x between 0.8M and 1.2M steps to handle the first curriculum transition.
    """
    def lr_schedule(progress_remaining):
        # progress_remaining goes from 1.0 to 0.0
        current_step = total_timesteps * (1.0 - progress_remaining)
        
        # Dip LR around 1M mark where spreads are first introduced
        if 800_000 <= current_step <= 1_200_000:
             return base_lr / 3.0
        
        return base_lr
    return lr_schedule

def train(args):
    # 1. Paths
    data_dir_path = Path(args.data_dir).resolve()
    log_dir_path = Path(args.log_dir).resolve()
    checkpoint_dir_path = Path(args.checkpoint_dir).resolve()
    
    # Ensure directories exist
    log_dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Environment
    def create_env():
        env = TradingEnv(data_dir=str(data_dir_path), is_training=True)
        env = Monitor(env)
        return env

    # Using DummyVecEnv for stability and simpler curriculum management
    env = DummyVecEnv([create_env])
    
    # Wrap in VecNormalize
    vec_norm_path = checkpoint_dir_path / "ppo_final_vecnormalize.pkl"
    if args.load_model and vec_norm_path.exists():
        logger.info(f"Loading VecNormalize from {vec_norm_path}")
        env = VecNormalize.load(str(vec_norm_path), env)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Load Config and Model
    config = load_ppo_config(args.config)
    
    # Extract training-specific parameters
    base_lr = config.pop('learning_rate', 0.0003)
    configured_timesteps = config.pop('total_timesteps', 5000000)
    
    # Determine total timesteps for this run
    if args.dry_run:
        total_timesteps = 1000
    else:
        total_timesteps = args.total_timesteps or configured_timesteps

    # Set up learning rate schedule
    config['learning_rate'] = get_lr_schedule(base_lr, total_timesteps)

    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        model = PPO.load(args.load_model, env=env, **config)
    else:
        logger.info("Creating new PPO model")
        model = PPO("MlpPolicy", env, verbose=1, **config)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, total_timesteps // 10), 
        save_path=str(checkpoint_dir_path),
        name_prefix="alpha_model"
    )
    
    metrics_callback = MetricsCallback(eval_freq=1000, plot_freq=10000)
    curriculum_callback = CurriculumCallback(verbose=1)

    # 4. Train
    # Configure file logging
    from datetime import datetime
    # Log to a fixed filename (no timestamp) as requested
    log_file = log_dir_path / "train_alpha.log"
    
    file_handler = logging.FileHandler(log_file, mode='w') # mode='w' to overwrite
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Configure SB3 logger
    new_logger = configure(str(log_dir_path), ["stdout", "csv"])
    model.set_logger(new_logger)

    logger.info(f"Logging to {log_file}")
    logger.info(f"Training for {total_timesteps} timesteps with Curriculum Learning...")
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[checkpoint_callback, metrics_callback, curriculum_callback],
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
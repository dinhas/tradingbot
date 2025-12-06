import os
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
            import os
            
            os.makedirs("logs/plots", exist_ok=True)
            
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
            plt.savefig(f'logs/plots/training_progress_{self.num_timesteps}.png', dpi=100)
            plt.close()
            
            logger.info(f"Saved training plot to logs/plots/training_progress_{self.num_timesteps}.png")
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    Decays from initial_value to a minimum of 1e-5.
    
    :param initial_value: Initial learning rate
    :return: Schedule function for stable-baselines3
    """
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, 1e-5)
    return func


def get_ppo_config(stage):
    """
    Returns PPO configuration based on the curriculum stage.
    
    UPDATED for better Stage 1 learning:
    - Lower entropy coefficient (0.003) to reduce random exploration
    - Separate policy/value networks for better value function learning
    - Learning rate schedule for stable convergence
    - Larger batches and conservative updates
    """
    # Neural Network Architecture
    # UPDATED: Separate networks for policy and value function
    # - Policy network: [256, 256, 128] - learns trading decisions
    # - Value network: [512, 256, 128] - larger capacity for return prediction
    policy_kwargs = {
        "net_arch": dict(
            pi=[256, 256, 128],   # Policy network (actor)
            vf=[512, 256, 128]    # Value network (critic) - larger for better value estimation
        ),
        "activation_fn": nn.ReLU
    }
    
    # UPDATED config based on TensorBoard analysis:
    # - Lower learning rate with decay for stability
    # - Larger batches for better gradient estimates
    # - More conservative updates (lower clip_range)
    # - Stronger value function learning (higher vf_coef)
    config = {
        "learning_rate": linear_schedule(1e-4),  # CHANGED: Slower with decay (was 3e-4 fixed)
        "n_steps": 4096,           # CHANGED: More experience before update (was 2048)
        "batch_size": 256,         # CHANGED: Larger batches (was 64)
        "n_epochs": 5,             # CHANGED: Fewer epochs to prevent overfitting (was 10)
        "gamma": 0.995,            # CHANGED: Value future rewards more (was 0.99)
        "gae_lambda": 0.95,
        "clip_range": 0.1,         # CHANGED: More conservative updates (was 0.2)
        "vf_coef": 1.0,            # CHANGED: Stronger value learning (was 0.5)
        "max_grad_norm": 0.3,      # CHANGED: Reduce gradient spikes (was 0.5)
        "verbose": 1,
        "tensorboard_log": "./logs/tensorboard",
        "policy_kwargs": policy_kwargs
    }

    # Stage-specific adjustments
    # UPDATED: Much lower entropy to reduce random exploration
    # Previous values caused too much randomness (40% win rate ≈ random)
    if stage == 1:
        config["ent_coef"] = 0.003   # CHANGED: Was 0.02 (6.7x lower)
    elif stage == 2:
        config["ent_coef"] = 0.002   # CHANGED: Was 0.01
    else:
        config["ent_coef"] = 0.001   # CHANGED: Was 0.005
        
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
    # Note: Using DummyVecEnv for cloud compatibility (SubprocVecEnv crashes during heavy preprocessing)
    # For local training with faster CPUs, you can switch back to SubprocVecEnv for parallel speedup
    n_envs = 4 if not args.dry_run else 1  # Reduced from 8 to 4 for stability
    
    # Always use DummyVecEnv on cloud platforms (Colab/Kaggle)
    logger.info(f"Creating {n_envs} environment(s) using DummyVecEnv...")
    env = DummyVecEnv([make_env(i, data_dir=args.data_dir, stage=args.stage) for i in range(n_envs)])
    
    # Apply Normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. Initialize Model
    ppo_config = get_ppo_config(args.stage)
    
    if args.load_model:
        logger.info(f"Loading model from {args.load_model}")
        # For Stage 1: Safe to load since action space is constant (5 outputs)
        # For Stage 2/3: Only load if resuming same stage, otherwise start fresh
        try:
            model = PPO.load(args.load_model, env=env, **ppo_config)
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
        save_freq=max(100000 // n_envs, 1), # Adjust freq for parallel envs
        save_path=args.checkpoint_dir,
        name_prefix=f"ppo_stage{args.stage}"
    )
    
    metrics_callback = MetricsCallback(
        eval_freq=1000,      # Log metrics every 1000 steps
        plot_freq=10000,     # Create plots every 10k steps
        verbose=1
    )
    
    # 4. Train
    total_timesteps = args.total_timesteps if not args.dry_run else 1000
    
    logger.info(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps, 
        callback=[checkpoint_callback, metrics_callback],
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
    parser.add_argument("--total_timesteps", type=int, default=1500000, help="Total timesteps to train (default: 1.5M for Stage 1)")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Path to log directory")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Path to save checkpoints")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load existing model (for continuing training)")
    parser.add_argument("--dry-run", action="store_true", help="Run a short test training loop")
    
    args = parser.parse_args()
    train(args)


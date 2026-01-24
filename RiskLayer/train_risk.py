import os
import sys
import torch
import torch.nn as nn
import numpy as np
import multiprocessing
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.sac.policies import SACPolicy, Actor, ContinuousCritic
from torch.nn import LayerNorm, ReLU

# Add src to path so we can import the environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from risk_env import RiskManagementEnv

class Tee:
    """Redirect stdout/stderr to both console and file."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = TeeStderr(self.file, self.stderr)
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

class TeeStderr:
    """Handle stderr separately."""
    def __init__(self, file, stderr):
        self.file = file
        self.stderr = stderr
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stderr.write(text)
        self.stderr.flush()
    
    def flush(self):
        self.file.flush()
        self.stderr.flush()

# --- Configuration for MAX SPEED ---
N_CPU = multiprocessing.cpu_count()
N_ENVS = min(N_CPU, 16) # Cap at 16 to prevent excessive overhead

print(f"Detected {N_CPU} CPUs. Using {N_ENVS} parallel environments for MAX SPEED.")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# --- SAC Hyperparameters (User Specified) ---
TOTAL_TIMESTEPS = 5_000_000
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005 # Soft update coefficient
BUFFER_SIZE = 1_000_000
TARGET_ENTROPY = -3.0 # Negative of action dimension (3)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
DATASET_PATH = os.path.join(PROJECT_ROOT, "risk_dataset.parquet")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Custom Network Implementation with LayerNorm ---

class LayerNormMLP(nn.Module):
    """
    Standard MLP with LayerNorm and ReLU.
    Reduced Architecture: [128, 128] (0.5x size to reduce overfitting)
    """
    def __init__(self, input_dim, net_arch=[128, 128]):
        super().__init__()
        layers = []
        last_dim = input_dim
        for latent_dim in net_arch:
            layers.append(nn.Linear(last_dim, latent_dim))
            layers.append(nn.LayerNorm(latent_dim))
            layers.append(nn.ReLU())
            last_dim = latent_dim
        self.latent_dim = last_dim
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class CustomSACPolicy(SACPolicy):
    """
    Custom SAC Policy to inject LayerNorm into Actor and Critic networks.
    """
    def make_actor(self, features_extractor=None):
        # Initial actor creation using standard SB3 logic
        actor = super().make_actor(features_extractor)
        
        # Safely get features dimension from the created actor
        if features_extractor is not None:
            input_dim = features_extractor.features_dim
        elif hasattr(actor, "features_extractor") and actor.features_extractor is not None:
            input_dim = actor.features_extractor.features_dim
        else:
            # Fallback to observation space shape for flat vectors
            input_dim = int(np.prod(self.observation_space.shape))
        
        # Inject LayerNorm into the latent pi network
        actor.latent_pi = LayerNormMLP(input_dim, net_arch=[128, 128])
        return actor

    def make_critic(self, features_extractor=None):
        # Initial critic creation using standard SB3 logic
        critic = super().make_critic(features_extractor)
        
        # Safely get features dimension from the created critic
        if features_extractor is not None:
            f_dim = features_extractor.features_dim
        elif hasattr(critic, "features_extractor") and critic.features_extractor is not None:
            f_dim = critic.features_extractor.features_dim
        else:
            # Fallback to observation space shape for flat vectors
            f_dim = int(np.prod(self.observation_space.shape))
            
        # SAC Critic input is state + action
        input_dim = f_dim + self.action_space.shape[0]
        
        # Inject LayerNorm into EACH twin Q-network by creating a new ModuleList
        new_q_networks = []
        for _ in range(len(critic.q_networks)):
            new_q_networks.append(
                nn.Sequential(
                    LayerNormMLP(input_dim, net_arch=[128, 128]),
                    nn.Linear(128, 1)
                )
            )
        critic.q_networks = nn.ModuleList(new_q_networks)
        return critic

class ConsoleLoggingCallback(BaseCallback):
    """
    Custom callback to mimic PPO-style logging for SAC (Off-Policy).
    Calculates pseudo-explained variance and tracks reward trends.
    """
    def __init__(self, check_freq=1000, verbose=1):
        super(ConsoleLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.value_losses = []
        
    def _on_step(self) -> bool:
        # Track rewards from infos if available
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])

        if self.n_calls % self.check_freq == 0:
            # Retrieve training logs
            if hasattr(self.model, "logger"):
                logs = self.model.logger.name_to_value
                actor_loss = logs.get("train/actor_loss", 0.0)
                critic_loss = logs.get("train/critic_loss", 0.0)
                ent_coef = logs.get("train/ent_coef", 0.0)
            
            # Calculate stats
            mean_rew = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            
            # Print PPO-like summary
            print(f"\n--- Step {self.num_timesteps} ---")
            print(f"  Mean Reward (100 ep): {mean_rew:.2f}")
            print(f"  Actor Loss:           {actor_loss:.4f}")
            print(f"  Critic Loss (Val):    {critic_loss:.4f} (Proxy for Value Loss)")
            print(f"  Entropy Coef:         {ent_coef:.4f}")
            print(f"---------------------------")
            
        return True

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                info = self.locals['infos'][0] 
                if 'equity' in info: self.logger.record('custom/equity', info['equity'])
                if 'pnl' in info: self.logger.record('custom/pnl', info['pnl'])
                if 'lots' in info: self.logger.record('custom/lots', info['lots'])
        return True

def make_env(rank, seed=0):
    def _init():
        env = RiskManagementEnv(dataset_path=DATASET_PATH, initial_equity=10000.0, is_training=True)
        env.reset(seed=seed + rank)
        return env
    return _init

def train():
    print(f"Starting SAC Training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    # 1. Create Vectorized Environment
    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    vec_env = VecMonitor(vec_env, LOG_DIR)

    # 2. Define Model (SAC)
    # UPDATED: Added weight_decay to fight overfitting
    # UPDATED: Reduced net_arch to [128, 128] via CustomSACPolicy
    # UPDATED: Batched updates (train_freq=64, gradient_steps=64) for SPEED
    
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], qf=[128, 128]),
        activation_fn=ReLU,
        optimizer_kwargs=dict(weight_decay=1e-5) # L2 Regularization to reduce overfitting
    )
    
    model = SAC(
        CustomSACPolicy, # Use custom policy with LayerNorm and reduced size
        vec_env,
        learning_rate=LR_ACTOR,
        buffer_size=BUFFER_SIZE,
        learning_starts=5000,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=64,      # OPTIMIZATION: Collect 64 steps
        gradient_steps=64,  # OPTIMIZATION: Then do 64 updates
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device,
        ent_coef='auto',
        target_entropy=TARGET_ENTROPY
    )

    # 2b. Explicitly ensure everything is on the correct device
    model.policy.to(device)
    if hasattr(model, 'critic_target') and model.critic_target is not None:
        model.critic_target.to(device)

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // N_ENVS,
        save_path=CHECKPOINT_DIR,
        name_prefix="risk_model_sac"
    )
    
    tb_callback = TensorboardCallback()
    console_callback = ConsoleLoggingCallback(check_freq=7500)

    # 4. Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, tb_callback, console_callback],
            progress_bar=True
        )
        
        # 5. Save Final Model and Stats
        final_path = os.path.join(MODELS_DIR, "risk_model_sac_final")
        model.save(final_path)
        vec_env.save(os.path.join(MODELS_DIR, "vec_normalize_sac.pkl"))
        print(f"Training Complete. Model saved to {final_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        model.save(os.path.join(MODELS_DIR, "risk_model_sac_interrupted"))
        vec_env.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_risk_sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_file)
    
    try:
        train()
    finally:
        tee.close()

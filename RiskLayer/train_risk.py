import os
import sys
import torch
import torch.nn as nn
import numpy as np
import multiprocessing
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
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

# --- Configuration ---
N_CPU = multiprocessing.cpu_count()
N_ENVS = 8 

print(f"Detected {N_CPU} CPUs. Using {N_ENVS} parallel environments.")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from typing import Callable
def linear_schedule(initial_value: float, final_value: float = 3e-5) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining
    return func

# --- PPO Hyperparameters ---
TOTAL_TIMESTEPS = 6_000_000
LEARNING_RATE = 3e-4
N_STEPS = 4096 
BATCH_SIZE = 1024
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.90
CLIP_RANGE = 0.2
ENT_COEF = 0.02
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Consistent storage paths
RISK_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "risk")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints", "risk")
DATASET_PATH = os.path.join(PROJECT_ROOT, "risk_dataset.parquet")

os.makedirs(RISK_MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Custom Network Implementation with LayerNorm ---

class LayerNormMLP(nn.Module):
    """
    Standard MLP with LayerNorm and ReLU.
    Architecture: [256, 256, 128]
    """
    def __init__(self, input_dim, net_arch=[256, 256, 128]):
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

class CustomMlpExtractor(nn.Module):
    """
    Custom MLP Extractor that returns latent policy and value features.
    """
    def __init__(self, feature_dim: int, net_arch=[256, 256, 128]):
        super().__init__()
        self.policy_net = LayerNormMLP(feature_dim, net_arch=net_arch)
        self.value_net = LayerNormMLP(feature_dim, net_arch=net_arch)
        self.latent_dim_pi = net_arch[-1]
        self.latent_dim_vf = net_arch[-1]

    def forward(self, features: torch.Tensor):
        return self.policy_net(features), self.value_net(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

class CustomPPOPolicy(ActorCriticPolicy):
    """
    Custom PPO Policy to inject LayerNorm into Actor and Critic (Value) networks.
    """
    def __init__(self, *args, **kwargs):
        super(CustomPPOPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Build the network layers.
        We override the default MLP extractor to use our CustomMlpExtractor.
        """
        self.mlp_extractor = CustomMlpExtractor(self.features_dim, net_arch=[256, 256, 128])

class ConsoleLoggingCallback(BaseCallback):
    """
    Custom callback to log PPO training progress.
    """
    def __init__(self, check_freq=1000, verbose=1):
        super(ConsoleLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        
    def _on_step(self) -> bool:
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])

        if self.n_calls % self.check_freq == 0:
            if hasattr(self.model, "logger"):
                logs = self.model.logger.name_to_value
                pg_loss = logs.get("train/policy_gradient_loss", 0.0)
                v_loss = logs.get("train/value_loss", 0.0)
                entropy = logs.get("train/entropy_loss", 0.0)
                explained_var = logs.get("train/explained_variance", 0.0)
            
            mean_rew = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            
            print(f"\n--- Step {self.num_timesteps} ---")
            print(f"  Mean Reward (100 ep): {mean_rew:.2f}")
            print(f"  Policy Loss:          {pg_loss:.4f}")
            print(f"  Value Loss:           {v_loss:.4f}")
            print(f"  Explained Var:        {explained_var:.4f}")
            print(f"  Entropy:              {entropy:.4f}")
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
    print(f"Starting PPO Risk Model Training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    # 1. Create Vectorized Environment
    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    vec_env = VecMonitor(vec_env, LOG_DIR)

    # 2. Define Model (PPO)
    policy_kwargs = dict(
        activation_fn=ReLU,
        optimizer_kwargs=dict(weight_decay=1e-5) 
    )
    
    model = PPO(
        CustomPPOPolicy, 
        vec_env,
        learning_rate=linear_schedule(LEARNING_RATE),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device
    )

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000 // N_ENVS,
        save_path=CHECKPOINT_DIR,
        name_prefix="risk_model_ppo"
    )
    
    tb_callback = TensorboardCallback()
    console_callback = ConsoleLoggingCallback(check_freq=5000)

    # 4. Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, tb_callback, console_callback],
            progress_bar=True
        )
        
        # 5. Save Final Model and Stats
        final_path = os.path.join(RISK_MODELS_DIR, "risk_model_final")
        model.save(final_path)
        vec_env.save(os.path.join(RISK_MODELS_DIR, "vec_normalize.pkl"))
        print(f"Training Complete. Model saved to {final_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        model.save(os.path.join(RISK_MODELS_DIR, "risk_model_interrupted")),
        vec_env.close()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_risk_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_file)
    
    try:
        train()
    finally:
        tee.close()

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
N_ENVS = N_CPU

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
    Standard MLP with LayerNorm and ReLU as requested.
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
    
    def forward(self, x):
        return self.mlp(x)

class CustomSACPolicy(SACPolicy):
    """
    Custom SAC Policy to inject LayerNorm into Actor and Critic networks.
    """
    def make_actor(self, features_extractor=None):
        # Use the provided features_extractor or the default one
        if features_extractor is None:
            features_extractor = self.features_extractor
        
        # Initial actor creation using standard SB3 logic
        actor = super().make_actor(features_extractor)
        
        # Inject LayerNorm into the latent pi network
        # The heads (mu/log_std) already expect 128 because of policy_kwargs['net_arch']
        input_dim = features_extractor.features_dim
        actor.latent_pi = LayerNormMLP(input_dim, net_arch=[256, 256, 128]).to(self.device)
        return actor

    def make_critic(self, features_extractor=None):
        # Use the provided features_extractor or the default one
        if features_extractor is None:
            features_extractor = self.features_extractor

        # Initial critic creation using standard SB3 logic
        critic = super().make_critic(features_extractor)
        
        # SAC Critic input is state + action
        input_dim = features_extractor.features_dim + self.action_space.shape[0]
        
        # Inject LayerNorm into each twin Q-network
        for q_net in critic.q_networks:
            # In SB3, q_net is a Sequential where the first element is the MLP
            q_net[0] = LayerNormMLP(input_dim, net_arch=[256, 256, 128]).to(self.device)
        return critic

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

def evaluate_policy(model, env, n_eval_episodes=50):
    print("\n--- Starting Evaluation ---")
    episode_returns = []
    episode_win_rates = []
    
    for i in range(n_eval_episodes):
        obs = env.reset()
        done = [False]
        pnls = []
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            if 'pnl' in info[0]: pnls.append(info[0]['pnl'])
            
        episode_returns.append(np.sum(pnls))
        wins = np.sum(np.array(pnls) > 0)
        episode_win_rates.append(wins / len(pnls) if pnls else 0)

    print(f"Eval Results: Avg Return: ${np.mean(episode_returns):.2f}, Avg Win Rate: {np.mean(episode_win_rates)*100:.2f}%")

def make_env(rank, seed=0):
    def _init():
        env = RiskManagementEnv(dataset_path=DATASET_PATH, initial_equity=10.0, is_training=True)
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
    # Key SAC Components:
    # - Soft Actor-Critic: Off-policy algorithm that maximizes expected return and entropy.
    # - Entropy Tuning: Automatically adjusts alpha (temperature) to maintain exploration.
    # - Twin Critics: Uses two Q-networks to mitigate overestimation bias.
    # - Target Networks: Uses moving average (tau=0.005) for stable Q-value targets.
    
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], qf=[256, 256, 128]),
        activation_fn=ReLU,
    )
    
    # We use SB3's standard MLP Policy but provide the user-requested architecture.
    # Note: To strictly enforce LayerNorm in ALL hidden layers as requested, 
    # we rely on SB3's internal create_mlp or custom policy override if needed.
    # For simplicity and robustness with SB3 parallel envs, we'll use the dict-based net_arch
    # and if LayerNorm is strictly required, we can inject it via the features extractor 
    # or a custom layer class.
    
    # REVISION: To ensure LayerNorm IS actually used between layers:
    # We'll use a custom policy-kwargs setup.
    
    model = SAC(
        CustomSACPolicy, # Use custom policy with LayerNorm
        vec_env,
        learning_rate=LR_ACTOR,
        buffer_size=BUFFER_SIZE,
        learning_starts=5000,
        batch_size=BATCH_SIZE,
        tau=TAU,
        gamma=GAMMA,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device,
        ent_coef='auto',
        target_entropy=TARGET_ENTROPY
    )

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // N_ENVS,
        save_path=CHECKPOINT_DIR,
        name_prefix="risk_model_sac"
    )
    
    tb_callback = TensorboardCallback()

    # 4. Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, tb_callback],
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

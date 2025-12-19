import os
import sys
import torch
import numpy as np
import multiprocessing
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from torch.nn import LeakyReLU

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
# CPU Utilization
N_CPU = multiprocessing.cpu_count()
# User requested MAXIMUM SPEED. Using ALL available cores.
# Warning: System might become sluggish during training.
N_ENVS = N_CPU

print(f"Detected {N_CPU} CPUs. Using {N_ENVS} parallel environments for MAX SPEED.")

# Optimize for fixed input size
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# PPO Hyperparameters (Expert Tuned for Financial Data)
# HYPERPARAMETERS - "Anti-Collapse" Edition 🛡️

TOTAL_TIMESTEPS = 10_000_000 
LEARNING_RATE = 5e-5      # 📉 HALVED. Slower learning = more stable updates.
N_STEPS = 4096            # ✂️ REDUCED. Update more frequently so policy doesn't get stale.
BATCH_SIZE = 512          # 📉 SMALLER. More updates per epoch, better gradient estimation.
GAMMA = 0.99              # ⬆️ INCREASED. Care a bit more about future rewards (trends).
GAE_LAMBDA = 0.95         # ⬆️ STANDARD. Smoothes out variance better.
ENT_COEF = 0.05           # 🚀 DOUBLED+. Force it to explore. "Don't get cocky."
VF_COEF = 0.5
MAX_GRAD_NORM = 0.3       # 🔒 TIGHTER. Cap those violent gradients.
CLIP_RANGE = 0.2          # 🔓 LOOSER. Gives the new LR room to work without clipping instantly.
N_EPOCHS = 3              # 📉 REDUCED. Stop memorizing the batch! 3 passes is enough.

# Why this works:
# 1. Lower LR + Fewer Epochs = The model stops overfitting on a single batch of data.
# 2. Higher Ent_Coef = It stops spamming the same "buy" button and actually thinks.
# 3. Higher Gamma = It looks for the bigger bag 💰, not just the next tick.

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Fallback logic for finding dataset
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, 'risk_dataset.parquet'),
    os.path.join(os.getcwd(), 'risk_dataset.parquet'),
    os.path.join(os.getcwd(), 'RiskLayer', 'risk_dataset.parquet'),
    'risk_dataset.parquet'
]

DATASET_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        DATASET_PATH = p
        break

if DATASET_PATH is None:
    # Default to standard path but warn
    DATASET_PATH = os.path.join(BASE_DIR, 'risk_dataset.parquet')
    print(f"WARNING: risk_dataset.parquet not found in common locations. Defaulting to {DATASET_PATH}")
else:
    print(f"Found Dataset at: {DATASET_PATH}")

MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
CHECKPOINT_DIR = os.path.join(MODELS_DIR, 'checkpoints')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class EntropyDecayCallback(BaseCallback):
    """
    Decays entropy coefficient linearly over time to encourage 
    exploration early (learning to block) and exploitation later.
    """
    def __init__(self, initial_ent=0.05, final_ent=0.005, decay_steps=3_000_000):
        super().__init__()
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.decay_steps = decay_steps
    
    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        new_ent = self.initial_ent - progress * (self.initial_ent - self.final_ent)
        self.model.ent_coef = new_ent
        return True

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional values in tensorboard.
    Logged less frequently to save I/O.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Only log every 100 steps to reduce overhead
        if self.n_calls % 100 == 0:
            if 'info' in self.locals and len(self.locals['info']) > 0:
                info = self.locals['info'][0] 
                if 'equity' in info:
                    self.logger.record('custom/equity', info['equity'])
                if 'pnl' in info:
                    self.logger.record('custom/pnl', info['pnl'])
                if 'lots' in info:
                     self.logger.record('custom/lots', info['lots'])
        return True

def make_env(rank, seed=0):
    """Utility function for multiprocessed env."""
    def _init():
        env = RiskManagementEnv(dataset_path=DATASET_PATH, initial_equity=10.0, is_training=True)
        env.reset(seed=seed + rank)
        return env
    return _init

def train():
    print(f"Starting High-Performance Training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    # 0. Pre-generate Cache (Sequentially) to avoid race conditions
    print("Pre-validating dataset cache...")
    temp_env = RiskManagementEnv(dataset_path=DATASET_PATH, initial_equity=10.0, is_training=True)
    del temp_env
    print("Cache validated. Starting parallel environments...")

    # 1. Create Vectorized Environment (Parallel)
    # SubprocVecEnv runs each env in a separate process
    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    
    # CRITICAL: Normalize observations. 
    # Financial features are on vastly different scales (Prices vs Slopes vs PnL)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    vec_env = VecMonitor(vec_env, LOG_DIR) # Monitor wrapper for logging

    # Network Architecture (Expert Recommended: Larger for 165 features)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 256, 128],  # Policy network (actor)
            vf=[512, 256, 128]   # Value network (critic)
        ),
        activation_fn=LeakyReLU,  # Better for financial data
        log_std_init=-1.0,  # Start with lower action variance
    )

    # 2. Define Model (PPO)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=LEARNING_RATE,
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
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // N_ENVS, # Roughly 500k steps (adjusted for parallel envs)
        save_path=CHECKPOINT_DIR,
        name_prefix="risk_model_ppo"
    )
    
    entropy_callback = EntropyDecayCallback(initial_ent=0.05, final_ent=0.005)
    tb_callback = TensorboardCallback()

    # 4. Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, entropy_callback, tb_callback],
            progress_bar=True
        )
        
        # 5. Save Final Model and Normalization Stats
        final_path = os.path.join(MODELS_DIR, "risk_model_final")
        model.save(final_path)
        # Important: Save the environment normalization statistics
        vec_env.save(os.path.join(MODELS_DIR, "vec_normalize.pkl"))
        print(f"Training Complete. Model saved to {final_path}")
        print(f"Normalization stats saved to {os.path.join(MODELS_DIR, 'vec_normalize.pkl')}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted manually.")
        model.save(os.path.join(MODELS_DIR, "risk_model_interrupted"))
        vec_env.close()
        print("Saved interrupted model.")
    except Exception as e:
        print(f"Error: {e}")
        vec_env.close()

if __name__ == "__main__":
    # Windows requires this
    multiprocessing.freeze_support()
    
    # Setup logging to file - capture all terminal output
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_file)
    
    try:
        print(f"All terminal output will be saved to: {log_file}")
        train()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
    finally:
        tee.close()
        print(f"Log saved to: {log_file}")

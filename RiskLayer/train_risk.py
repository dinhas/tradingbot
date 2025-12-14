import os
import sys
import torch
import numpy as np
import multiprocessing
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

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
# Leave 2 cores for system/backtesting if user is doing other things, or use all if dedicated.
# Using N_CPU - 2 is safer for responsiveness, but for "MAX SPEED" we'll use max(1, N - 1)
N_ENVS = max(1, N_CPU - 1)

print(f"Detected {N_CPU} CPUs. Using {N_ENVS} parallel environments.")

# PPO Hyperparameters (Optimized for Throughput)
TOTAL_TIMESTEPS = 5_000_000 # Increased since we train faster
LEARNING_RATE = 3e-4
N_STEPS = 2048 # Steps per env per update. Total batch = N_STEPS * N_ENVS
BATCH_SIZE = 512 # Larger batch size for GPU efficiency
GAMMA = 0.99
GAE_LAMBDA = 0.95
ENT_COEF = 0.01 
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

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

    # 1. Create Vectorized Environment (Parallel)
    # SubprocVecEnv runs each env in a separate process
    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env, LOG_DIR) # Monitor wrapper for logging

    # 2. Define Model (PPO)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=10,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=0.2,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        verbose=1,
        tensorboard_log=LOG_DIR,
        device=device
    )

    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // (N_ENVS), # Save roughly every 50k total steps
        save_path=CHECKPOINT_DIR,
        name_prefix="risk_model_ppo"
    )
    
    tb_callback = TensorboardCallback()

    # 4. Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, tb_callback],
            progress_bar=True
        )
        
        # 5. Save Final Model
        final_path = os.path.join(MODELS_DIR, "risk_model_final")
        model.save(final_path)
        print(f"Training Complete. Model saved to {final_path}")
        
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

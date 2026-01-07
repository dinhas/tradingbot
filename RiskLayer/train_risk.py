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
import logging

# Configure logging to output to stdout (which Tee will capture)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Add src to path so we can import the environment
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# from risk_env import RiskManagementEnv (Moved inside make_env)

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
# HYPERPARAMETERS - "Max Efficiency Edition" 🚀

TOTAL_TIMESTEPS = 15_000_000 
LEARNING_RATE = 1.5e-4    # 📈 Increased for maximum learning in 5M steps.
N_STEPS = 16384           # ⬆️ INCREASED. Large window for stable gradient updates.
BATCH_SIZE = 2048         # ⬆️ INCREASED. Stability with higher Learning Rate.
GAMMA = 0.98              # 📉 Slightly lower. Focus on high-quality trades, not infinite future.
GAE_LAMBDA = 0.95         
ENT_COEF = 0.05           # ⬆️ Increased starting entropy for aggressive exploration.
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5       
CLIP_RANGE = 0.2          
N_EPOCHS = 8             # ⬆️ INCREASED. Squeeze more learning from each batch.

# Why this works:
# 1. 1.5e-4 LR + 5M Steps = Aggressive but stable convergence.
# 2. 16384 Steps = ~160 episodes per env per update (High sample efficiency).
# 3. 5M Timesteps = Focused learning on core reward dynamics.

# Paths
# Using absolute paths based on script location for robustness
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")
DATASET_PATH = os.path.join(PROJECT_ROOT, "risk_dataset.parquet")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class StepBasedEntropyCallback(BaseCallback):
    """
    6) ENTROPY SCHEDULE
    - 0 to 5M steps: ent_coef = 0.06
    - 5M to 15M steps: ent_coef = 0.03
    - 15M+ steps: ent_coef = 0.015
    """
    def __init__(self):
        super().__init__()

    def _on_step(self):
        steps = self.num_timesteps
        if steps < 5_000_000:
            self.model.ent_coef = 0.06
        elif steps < 10_000_000:
            self.model.ent_coef = 0.03
        else:
            self.model.ent_coef = 0.015
        return True

class GlobalStepCallback(BaseCallback):
    """Updates the shared global step counter for environments."""
    def __init__(self, shared_step, update_freq=1024):
        super().__init__()
        self.shared_step = shared_step
        self.update_freq = update_freq
    
    def _on_step(self):
        # PERFORMANCE FIX: Only update shared memory every N steps to avoid lock contention
        if self.n_calls % self.update_freq == 0:
            self.shared_step.value = self.num_timesteps
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

def make_env(rank, seed=0, shared_step=None):
    """Utility function for multiprocessed env."""
    def _init():
        # Lazy import to ensure clean subprocess initialization
        from risk_env import RiskManagementEnv
        env = RiskManagementEnv(dataset_path=DATASET_PATH, initial_equity=10.0, is_training=True, shared_step_counter=shared_step)
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
    # Shared Step Counter for Staged Rewards
    # Using Manager to avoid mmap pickling issues with spawn/cloudpickle on Linux/Kaggle
    # value is a proxy, which is picklable
    manager = multiprocessing.Manager()
    shared_step = manager.Value('l', 0)
    
    # SubprocVecEnv runs each env in a separate process
    env_fns = [make_env(i, shared_step=shared_step) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    
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

    # Network Architecture (Increased by 50%)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[768, 384, 192],  # Policy network (actor)
            vf=[768, 384, 192]   # Value network (critic)
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
    checkpoint_callback = CheckpointCallback(
        save_freq=500_000 // N_ENVS, # Roughly 500k steps (adjusted for parallel envs)
        save_path=CHECKPOINT_DIR,
        name_prefix="risk_model_ppo"
    )
    
    entropy_callback = StepBasedEntropyCallback()
    global_step_callback = GlobalStepCallback(shared_step)
    tb_callback = TensorboardCallback()

    # 4. Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, entropy_callback, tb_callback, global_step_callback],
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

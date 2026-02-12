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

# PPO Hyperparameters — V3 "Intelligent Risk Allocation"
# Tuned for 60-dim obs, 2-dim action (SL, TP multipliers)

TOTAL_TIMESTEPS = 5_000_000
LEARNING_RATE = 5e-5          # More stable for 2-action space
N_STEPS = 2048                # Good balance for stability
BATCH_SIZE = 256              # Standard batch size
GAMMA = 0.99                  # standard discount for clean SL/TP rewards
GAE_LAMBDA = 0.95             # Standard
ENT_COEF = 0.03               # Lower entropy needed for 2 continuous actions
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5           # Standard clipping
CLIP_RANGE = 0.2              # Standard clipping
N_EPOCHS = 5                  # Slightly more epochs to extract more from rollouts

# V3 Rationale:
# 1. 60-dim obs (was 70) => leaner network
# 2. 2 actions (was 4) => faster convergence, lower entropy needed (0.03)
# 3. 5M steps => ensures deep refinement of SL/TP strategy

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Fallback logic for finding dataset
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, 'risk_dataset.parquet'),
    os.path.join(os.path.dirname(BASE_DIR), 'data', 'risk_dataset.parquet'),
    os.path.join(os.getcwd(), 'data', 'risk_dataset.parquet'),
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
    Decays entropy coefficient linearly over time.
    V3: Higher initial entropy (0.08) for exploration,
    decays to 0.01 over first 2M steps.
    """
    def __init__(self, initial_ent=0.03, final_ent=0.005, decay_steps=2_000_000):
        super().__init__()
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.decay_steps = decay_steps
    
    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        new_ent = self.initial_ent - progress * (self.initial_ent - self.final_ent)
        self.model.ent_coef = new_ent
        return True

class RollingMetricsCallback(BaseCallback):
    """
    V3: Tracks rolling Sharpe ratio and Profit Factor across episodes.
    Logs equity, PnL, and rolling performance metrics to TensorBoard.
    """
    def __init__(self, window=50, verbose=0):
        super(RollingMetricsCallback, self).__init__(verbose)
        self.window = window
        self.pnl_history = []

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            if 'infos' in self.locals and len(self.locals['infos']) > 0:
                for info in self.locals['infos']:
                    if 'equity' in info:
                        self.logger.record('custom/equity', info['equity'])
                    if 'pnl' in info:
                        pnl = info['pnl']
                        self.logger.record('custom/pnl', pnl)
                        if pnl != 0:
                            self.pnl_history.append(pnl)
                    if 'lots' in info:
                        self.logger.record('custom/lots', info['lots'])
        
        # Log rolling metrics every 500 steps
        if self.n_calls % 500 == 0 and len(self.pnl_history) >= 10:
            recent = self.pnl_history[-self.window:]
            recent_np = np.array(recent)
            
            # Rolling Sharpe
            mean_r = np.mean(recent_np)
            std_r = np.std(recent_np)
            sharpe = mean_r / max(std_r, 1e-8)
            self.logger.record('rolling/sharpe', sharpe)
            
            # Profit Factor
            wins = recent_np[recent_np > 0]
            losses = recent_np[recent_np < 0]
            gross_profit = np.sum(wins) if len(wins) > 0 else 0
            gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-8
            pf = gross_profit / max(gross_loss, 1e-8)
            self.logger.record('rolling/profit_factor', pf)
            
            # Win rate
            win_rate = len(wins) / max(len(recent), 1)
            self.logger.record('rolling/win_rate', win_rate)
            
            # Avg PnL
            self.logger.record('rolling/avg_pnl', mean_r)
        
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

    # Network Architecture
    # 60 features -> 256 -> 128 -> 64 -> 2 actions
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128, 64],   # Policy network (actor)
            vf=[256, 128, 64]    # Value network (critic)
        ),
        activation_fn=LeakyReLU,
        log_std_init=-1.5,
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
    
    entropy_callback = EntropyDecayCallback(initial_ent=0.03, final_ent=0.005, decay_steps=3_000_000)
    metrics_callback = RollingMetricsCallback(window=50)

    # 4. Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, entropy_callback, metrics_callback],
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

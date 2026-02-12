import os
import multiprocessing
from torch.nn import LeakyReLU

# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATASET_PATH = os.path.join(PROJECT_ROOT, "risk_dataset.parquet")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")

# --- Training Parameters ---
TOTAL_TIMESTEPS = 5_000_000
N_CPU = multiprocessing.cpu_count()
N_ENVS = N_CPU # Use all cores for MAX SPEED

# PPO Hyperparameters
LEARNING_RATE = 1e-4
N_STEPS = 8192
BATCH_SIZE = 4096
N_EPOCHS = 4
GAMMA = 0.98
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.05
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Entropy Decay
INITIAL_ENT = 0.05
FINAL_ENT = 0.005
ENT_DECAY_STEPS = 4_000_000 # Decay over 80% of training

# Network Architecture
POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 128], # Reduced to prevent overfitting to noise
        vf=[256, 128]
    ),
    activation_fn=LeakyReLU,
    log_std_init=-1.0,
)

# --- Environment Settings ---
INITIAL_EQUITY = 10.0
DRAWDOWN_TERMINATION_THRESHOLD = 0.95
MAX_LEVERAGE = 100.0
BASE_SPREAD_PIPS = 1.2
SLIPPAGE_MAX_PIPS = 0.5

# Observation/Action Space
OBS_DIM = 45 # 40 Market + 5 Account
ACTION_DIM = 2 # SL_Mult, TP_Mult

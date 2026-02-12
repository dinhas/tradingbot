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

# Add current dir and src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
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
N_CPU = config.N_CPU
N_ENVS = config.N_ENVS

print(f"Detected {N_CPU} CPUs. Using {N_ENVS} parallel environments for MAX SPEED.")

# Optimize for fixed input size
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# PPO Hyperparameters (Loaded from config)
TOTAL_TIMESTEPS = config.TOTAL_TIMESTEPS
LEARNING_RATE = config.LEARNING_RATE
N_STEPS = config.N_STEPS
BATCH_SIZE = config.BATCH_SIZE
GAMMA = config.GAMMA
GAE_LAMBDA = config.GAE_LAMBDA
ENT_COEF = config.ENT_COEF
VF_COEF = config.VF_COEF
MAX_GRAD_NORM = config.MAX_GRAD_NORM
CLIP_RANGE = config.CLIP_RANGE
N_EPOCHS = config.N_EPOCHS

# Paths
MODELS_DIR = config.MODELS_DIR
LOG_DIR = config.LOG_DIR
CHECKPOINT_DIR = config.CHECKPOINT_DIR
DATASET_PATH = config.DATASET_PATH

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class EntropyDecayCallback(BaseCallback):
    """
    Decays entropy coefficient linearly over time to encourage 
    exploration early (learning to block) and exploitation later.
    """
    def __init__(self, initial_ent=config.INITIAL_ENT, final_ent=config.FINAL_ENT, decay_steps=config.ENT_DECAY_STEPS):
        super().__init__()
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.decay_steps = decay_steps
    
    def _on_step(self):
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        new_ent = self.initial_ent - progress * (self.initial_ent - self.final_ent)
        self.model.ent_coef = new_ent
        return True

class CustomStatsCallback(BaseCallback):
    """
    Forces episode statistics into the SB3 console table and 
    provides immediate console feedback when an episode finishes.
    """
    def __init__(self, verbose=0):
        super(CustomStatsCallback, self).__init__(verbose)
        from collections import deque
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.last_step_count = 0

    def _on_step(self) -> bool:
        # Heartbeat every 10000 steps to confirm callback is alive
        # if self.num_timesteps - self.last_step_count >= 10000:
        #     print(f"--- [Callback Heartbeat] Steps: {self.num_timesteps} | Finished Eps: {len(self.episode_rewards)} | Total Env Steps: {self.num_timesteps} ---", flush=True)
        #     self.last_step_count = self.num_timesteps

        if 'infos' in self.locals:
            for info in self.locals['infos']:
                # Individual Monitor wrapper uses 'episode'
                # VecMonitor might use something else, but 'episode' is standard
                if 'episode' in info:
                    ep_rew = info['episode']['r']
                    ep_len = info['episode']['l']
                    self.episode_rewards.append(ep_rew)
                    self.episode_lengths.append(ep_len)
                    # print(f" >>> [Episode End] Reward: {ep_rew:.2f} | Length: {ep_len} | Total Avg: {np.mean(self.episode_rewards):.2f}", flush=True)
        
        # Periodically record to ensure the rollout group appears in SB3 table
        if len(self.episode_rewards) > 0:
            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards))
            self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths))
        elif self.num_timesteps > 50000:
            # Force record 0s if nothing is happening so we see the rows in the table
            self.logger.record("rollout/ep_rew_mean", 0.0)
            self.logger.record("rollout/ep_len_mean", 0.0)
            
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

from stable_baselines3.common.monitor import Monitor

def make_env(rank, seed=0):
    """Utility function for multiprocessed env."""
    def _init():
        env = RiskManagementEnv(dataset_path=DATASET_PATH, initial_equity=config.INITIAL_EQUITY, is_training=True)
        # Wrap each individual env in a Monitor for reliable logging
        env = Monitor(env)
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
    temp_env = RiskManagementEnv(dataset_path=DATASET_PATH, initial_equity=config.INITIAL_EQUITY, is_training=True)
    del temp_env
    print("Cache validated. Starting parallel environments...")

    # 1. Create Vectorized Environment (Parallel)
    # SubprocVecEnv runs each env in a separate process
    env_fns = [make_env(i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    
    # We already wrap individual envs with Monitor in make_env,
    # so we don't strictly need VecMonitor here, which avoids the UserWarning.
    # vec_env = VecMonitor(vec_env, LOG_DIR) 

    # CRITICAL: Normalize observations. 
    # Financial features are on vastly different scales (Prices vs Slopes vs PnL)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    
    # Network Architecture (Expert Recommended: Larger for 45 features)
    policy_kwargs = config.POLICY_KWARGS

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
    
    entropy_callback = EntropyDecayCallback()
    tb_callback = TensorboardCallback()
    stats_callback = CustomStatsCallback()

    # 4. Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=[checkpoint_callback, entropy_callback, tb_callback, stats_callback],
            log_interval=1,
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

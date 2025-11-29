import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO
from trading_env import TradingEnv
import os
import torch

# Create directories for logs and models
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    """
    def _init():
        env = TradingEnv()
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    # --- Configuration matches PRD Section 7.3 & 7.4 ---
    import multiprocessing
    NUM_CPU = multiprocessing.cpu_count()
    TOTAL_TIMESTEPS = 850000 # Phase 3: 600k steps
    
    print(f"Initializing {NUM_CPU} parallel environments (using all available cores)...")
    # SubprocVecEnv runs environments in separate processes for true parallelism
    env = SubprocVecEnv([make_env(i) for i in range(NUM_CPU)])
    
    # CRITICAL FIX: Wrap with VecNormalize for reward stabilization
    print("Wrapping with VecNormalize (reward normalization)...")
    env = VecNormalize(
        env,
        norm_obs=False,      # We already normalize observations in TradingEnv
        norm_reward=True,    # CRITICAL: Normalize rewards to prevent value explosion
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )

    print("Setting up RecurrentPPO model with LSTM Policy...")
    
    # Hyperparameters from PRD Section 7.3
model = RecurrentPPO(
    "MlpLstmPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/",

    # --- Core Training Stability ---
    learning_rate=1.5e-4,      # Lower LR = less volatility
    n_steps=1536,              # Slightly shorter rollouts reduce pattern memorization
    batch_size=128,            # Bigger batch = smoother gradients
    n_epochs=8,                # Lower replay cycles = less overfitting

    # --- Temporal + VA tradeoffs ---
    gamma=0.99,                # Keep
    gae_lambda=0.92,           # Slightly lower = less variance, more stability

    # --- Regularization (very important) ---
    clip_range=0.15,           # Tighter clip = safer learning
    ent_coef=0.005,            # Reduced entropy = no over-exploration
    vf_coef=0.4,               # Slightly lower = avoids value overfitting
    max_grad_norm=0.5,         # Keep

    # --- LSTM settings ---
    policy_kwargs=dict(
        lstm_hidden_size=96,      # Lower capacity = less memorizing noise
        n_lstm_layers=1,          # 2 layers = high overfit risk, so reduce
        enable_critic_lstm=True
    ),

    device="cuda" if torch.cuda.is_available() else "cpu"
)

    # Checkpoint Callback (PRD Section 7.4)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000 // NUM_CPU,  # Adjust for parallel envs
        save_path='./models/',
        name_prefix='recurrent_ppo_trading',
        save_vecnormalize=True  # Save normalization stats
    )

    print(f"Starting training for {TOTAL_TIMESTEPS} steps on {model.device}...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            progress_bar=True
        )
        print("Training complete.")
        
        # Save Final Model
        model.save("models/recurrent_ppo_final")
        env.save("models/recurrent_ppo_final_vecnormalize.pkl")  # Save normalization stats
        print("Final model saved to models/recurrent_ppo_final.zip")
        print("VecNormalize stats saved to models/recurrent_ppo_final_vecnormalize.pkl")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user! Saving emergency checkpoint...")
        model.save("models/recurrent_ppo_interrupted")
        env.save("models/recurrent_ppo_interrupted_vecnormalize.pkl")
        print("✅ Interrupted model saved to models/recurrent_ppo_interrupted.zip")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

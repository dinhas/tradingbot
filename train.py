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
    TOTAL_TIMESTEPS = 1_000_000  # Increased to 1M for comprehensive learning
    
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

    # --- Core Training ---
    learning_rate=8e-5,        # Slightly lower but still active
    n_steps=2048,              # Longer rollouts = more context
    batch_size=64,             # Smaller batch = more gradient signal
    n_epochs=12,               # More learning per rollout = real improvement

    # --- Exploration + Learning ---
    ent_coef=0.02,             # Boost exploration (THIS = huge difference)
    clip_range=0.25,           # Loosen it up so PPO can actually update

    # --- Temporal weighting ---
    gamma=0.995,               # Long-term reward matters more
    gae_lambda=0.95,           # More consistent GAE estimates

    # --- Value Loss ---
    vf_coef=0.5,               # Standard stable baseline
    max_grad_norm=0.7,         # Allow slightly bigger updates

    # --- LSTM settings ---
    policy_kwargs=dict(
        lstm_hidden_size=128,     # More memory = more edge
        n_lstm_layers=2,          # Deeper temporal learning
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

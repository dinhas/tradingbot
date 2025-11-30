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
    TOTAL_TIMESTEPS = 2_500_000 # Increased to 1M for comprehensive learning

    
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

        # --- Core Training Stability & Speed ---
        # Increased to speed up learning with the larger dataset.
        learning_rate=3e-4, 
        # Increased rollout length to capture longer-term dependencies in market data.
        n_steps=2048, 
        # Increased batch size for smoother, more representative gradient updates.
        batch_size=256, 
        # Slightly increased replay cycles to better utilize the collected data.
        n_epochs=10, 
        
        # --- Temporal & VA Tradeoffs (Kept Stable) ---
        gamma=0.99, 
        gae_lambda=0.92, 

        # --- Regularization ---
        # Increased slightly to allow for larger, more impactful updates.
        clip_range=0.2, 
        # Significantly reduced to focus the agent on exploitation (refining the best policy).
        ent_coef=0.001, 
        # Kept stable to prevent value function overfitting.
        vf_coef=0.4, 
        max_grad_norm=0.5, 

        # --- LSTM settings (Increased Capacity) ---
        policy_kwargs=dict(
            # Increased capacity to better memorize and model complex market regimes over 7 years.
            lstm_hidden_size=128, 
            # Kept at 1 layer to minimize risk of high overfitting.
            n_lstm_layers=1, 
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

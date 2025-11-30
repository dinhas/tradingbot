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
    TOTAL_TIMESTEPS = 1_500_000 # Increased to 1M for comprehensive learning

    
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
    
    # Conservative Hyperparameters - Anti-Overfitting Focus
    # Balanced for learning entry quality signals on 7 years of data
    # without memorizing specific patterns
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",

        # --- Core Training Stability & Speed ---
        # CONSERVATIVE: Keep learning rate stable to prevent unstable updates
        learning_rate=3e-4,  # Moderate LR prevents overfitting to recent data
        
        # Large rollout = better temporal understanding, no overfitting risk
        n_steps=2048, 
        
        # LARGER batch = MORE stable gradients = LESS overfitting
        batch_size=512,  # Was 256, larger batch reduces variance
        
        # CONSERVATIVE: Don't over-train on same batch (overfitting risk)
        n_epochs=10,  # Keep at 10 - more epochs = memorization risk
        
        # --- Temporal & Credit Assignment ---
        gamma=0.99,  # Standard discount factor
        
        # Moderate GAE lambda for good credit assignment without bias
        gae_lambda=0.94,  # Was 0.92, slight increase for entry quality signals
        
        # --- Regularization (CRITICAL for Anti-Overfitting) ---
        # Conservative clip range prevents large, unstable updates
        clip_range=0.2,  # Keep stable - prevents overfitting to outliers
        
        # HIGH entropy = MORE exploration = LESS overfitting
        # This is the MOST IMPORTANT anti-overfitting parameter
        ent_coef=0.015,  # 15x original - strong exploration penalty
        
        # Moderate value coefficient
        vf_coef=0.5,  # Prevent value function from dominating policy
        
        # Gradient clipping prevents exploding gradients
        max_grad_norm=0.5,  # Keep stable

        # --- LSTM settings (Conservative Size) ---
        policy_kwargs=dict(
            # CONSERVATIVE: Keep LSTM small to prevent memorization
            # Smaller network = harder to memorize 7 years of specific patterns
            lstm_hidden_size=128,  # Keep original - prevents overfitting
            
            # Single layer = less capacity = less overfitting
            n_lstm_layers=1, 
            
            # Enable critic LSTM for value estimation
            enable_critic_lstm=True,
            
            # ADD DROPOUT for regularization (CRITICAL anti-overfitting)
            lstm_dropout=0.1,  # 10% dropout prevents co-adaptation
            
            # Network architecture regularization
            net_arch=dict(
                pi=[128],  # Policy network: single 128 layer (was default 64)
                vf=[128]   # Value network: single 128 layer
            ),
            
            # Orthogonal initialization helps generalization
            ortho_init=True
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

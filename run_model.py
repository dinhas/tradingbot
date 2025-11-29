import gymnasium as gym
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from trading_env import TradingEnv
import os

def run_inference(model_path="final_model", stats_path="final_model_vecnormalize.pkl"):
    """
    Runs the trained model on the TradingEnv.
    """
    print(f"Loading model from {model_path}...")
    
    # 1. Load the Environment
    # We must wrap it exactly as we did during training (DummyVecEnv + VecNormalize)
    # However, for inference, we set training=False and norm_reward=False
    env = DummyVecEnv([lambda: TradingEnv()])
    
    # Load normalization statistics
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = False      # Do not update stats during inference
        env.norm_reward = False   # We want to see real rewards/PnL
    else:
        print("⚠️ Warning: Normalization stats not found! Model performance may be degraded.")

    # 2. Load the Model
    model = RecurrentPPO.load(model_path, env=env)
    
    # 3. Run Inference Loop
    obs = env.reset()
    
    # LSTM States: (hidden_state, cell_state)
    # Initial state is None or zeros
    lstm_states = None
    
    # Episode start mask
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    print("\nStarting Inference Loop...")
    print("-" * 50)
    
    total_reward = 0
    steps = 0
    
    try:
        while True:
            # Predict Action
            # deterministic=True means we take the most likely action (no exploration)
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            
            # Step Environment
            obs, rewards, dones, infos = env.step(action)
            
            # Update tracking
            episode_starts = dones
            total_reward += rewards[0]
            steps += 1
            
            # Print status every 100 steps
            if steps % 100 == 0:
                info = infos[0]
                print(f"Step {steps}: Portfolio=${info['portfolio_value']:.2f} | Return={info['return']:.4f}")
            
            if dones[0]:
                print("-" * 50)
                print(f"Episode Finished!")
                print(f"Total Steps: {steps}")
                print(f"Final Portfolio Value: ${infos[0]['portfolio_value']:.2f}")
                break
                
    except KeyboardInterrupt:
        print("\nInference stopped by user.")
        
    return infos[0]

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists("final_model.zip"):
        print("❌ Model file not found. Please train the model first.")
    else:
        run_inference()

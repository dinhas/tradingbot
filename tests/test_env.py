import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_env import TradingEnv

def test_env():
    print("Initializing TradingEnv...")
    try:
        env = TradingEnv(stage=3, is_training=True)
        print("✅ Environment initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize environment: {e}")
        return

    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    # Test Reset
    print("\nTesting Reset...")
    obs, info = env.reset()
    print(f"✅ Reset successful. Observation shape: {obs.shape}")
    
    if obs.shape != (140,):
        print(f"❌ Observation shape mismatch! Expected (140,), got {obs.shape}")
    
    # Test Step
    print("\nTesting Step...")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"✅ Step successful.")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Observation shape: {obs.shape}")

    # Test Episode Loop
    print("\nRunning short episode (10 steps)...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break
            
    print("✅ Episode loop finished.")

if __name__ == "__main__":
    test_env()

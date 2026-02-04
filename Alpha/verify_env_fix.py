
import sys
import os
import logging
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'Alpha', 'src'))

try:
    from trading_env import TradingEnv
except ImportError:
    print("Could not import TradingEnv. Check paths.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)

def test_environment():
    print("Initializing TradingEnv...")
    env = TradingEnv(
        data_dir="e:/tradingbot/Alpha/data", 
        is_training=True
    )
    
    print("Resetting environment...")
    obs, info = env.reset()
    
    print("Running 10 steps...")
    total_reward = 0
    for i in range(10):
        # Action: Random direction
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: Reward={reward:.4f}, Done={done}")
        
        if done:
            break
            
    print("Test Complete. Environment is functional.")

if __name__ == "__main__":
    test_environment()

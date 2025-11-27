from trading_env import TradingEnv
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def test_environment():
    print("Initializing Environment...")
    try:
        env = TradingEnv()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    print("Resetting Environment...")
    obs, info = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    print(f"Initial Info: {info}")
    
    # Verify Observation Shape
    if obs.shape != (97,):
        print(f"❌ Observation shape mismatch! Expected (97,), got {obs.shape}")
    else:
        print("✅ Observation shape correct.")

    print("\nRunning Simulation Loop...")
    for i in range(10):
        # Random Action (Softmax will be applied inside)
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio Value: ${info['portfolio_value']:.2f}")
        print(f"  Fees Paid: ${info['fees']:.4f}")
        print(f"  Terminated: {terminated}")
        
        if terminated or truncated:
            print("Episode Ended.")
            break
            
    print("\n✅ Test Complete.")

if __name__ == "__main__":
    test_environment()

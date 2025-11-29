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
    print(f"Action Space Shape: {env.action_space.shape}")
    
    # Verify Action Space
    if env.action_space.shape != (9,):
        print(f"❌ Action space mismatch! Expected (9,), got {env.action_space.shape}")
        return
    else:
        print("✅ Action space correct (9 dims).")

    print("\nRunning Simulation Loop...")
    for i in range(10):
        # Random Action (9 dims)
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}:")
        print(f"  SL Mult: {env.sl_multiplier:.2f}x ATR | TP Mult: {env.tp_multiplier:.2f}x ATR")
        print(f"  Reward: {reward:.4f}")
        print(f"  Portfolio Value: ${info['portfolio_value']:.2f}")
        print(f"  Fees Paid: ${info['fees']:.4f}")
        
        if terminated or truncated:
            print("Episode Ended.")
            break
            
    print("\n✅ Test Complete.")

if __name__ == "__main__":
    test_environment()

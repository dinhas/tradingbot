import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_env import TradingEnv

def test_env_compliance():
    print("Initializing TradingEnv for Compliance Check...")
    try:
        env = TradingEnv(stage=3, is_training=True)
        print("✅ Environment initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize environment: {e}")
        return

    # Test Reset
    obs, info = env.reset()
    
    # 1. Test Risk Management (Global Exposure)
    print("\nTesting Risk Management (Global Exposure)...")
    # Try to open max positions on all assets
    # Max Pos Size = 50% of equity
    # Max Total Exposure = 60% of equity
    # If we open 2 positions at max size, we hit 100% -> Should be rejected
    
    # Force actions
    # Action: [Dir, Size, SL, TP]
    # Size = 1.0 (Max)
    
    # Asset 0: Buy, Max Size
    action_0 = np.zeros(20)
    action_0[0] = 1.0 # Buy
    action_0[1] = 1.0 # Max Size
    
    obs, reward, term, trunc, info = env.step(action_0)
    pos_0 = env.positions[env.assets[0]]
    
    if pos_0:
        print(f"✅ Position 0 opened. Size: {pos_0['size']:.2f}")
    else:
        print("❌ Position 0 failed to open!")
        
    # Asset 1: Buy, Max Size (Should be rejected or capped?)
    # PRD says: "Validate total exposure... If fails: Reject trade"
    # Current Exposure = 50%. New Exposure = 50%. Total = 100% > 60%.
    # Should be rejected.
    
    action_1 = np.zeros(20)
    action_1[0] = 1.0 # Maintain Buy on Asset 0
    action_1[1] = 1.0 # Maintain Max Size on Asset 0
    action_1[4] = 1.0 # Buy Asset 1
    action_1[5] = 1.0 # Max Size
    
    obs, reward, term, trunc, info = env.step(action_1)
    pos_1 = env.positions[env.assets[1]]
    
    if pos_1 is None:
        print("✅ Position 1 rejected due to Global Exposure limit (Expected).")
    else:
        print(f"❌ Position 1 opened! Risk Management Failed. Size: {pos_1['size']:.2f}")

    # 2. Test Reward Function Components
    print("\nTesting Reward Function...")
    # We have one position open.
    # Step to generate some PnL and holding time
    
    # Force a holding period
    for _ in range(5):
        env.step(np.zeros(20)) # Hold
        
    # Check if reward is calculated (non-zero)
    # It might be small due to normalization
    print(f"Last Reward: {reward}")
    
    # Check if transaction costs were tracked
    print(f"Transaction Costs (Step): {env.transaction_costs_step}")
    
    # Check Holding Penalty
    # We held for 5 steps.
    # Penalty = -0.01 * num_pos * (avg_age / 100)
    # num_pos = 1. avg_age ~ 5.
    # Penalty ~ -0.01 * 1 * 0.05 = -0.0005
    # This is part of the total reward.
    
    print("✅ Reward function executed without error.")

if __name__ == "__main__":
    test_env_compliance()


import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from risk_env import RiskManagementEnv

def test_env():
    dataset_path = os.path.join(os.path.dirname(__file__), 'risk_dataset.parquet')
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        # Create a dummy one for testing if needed, though real one is preferred
        return

    print("Initializing Environment...")
    env = RiskManagementEnv(dataset_path=dataset_path, initial_equity=100.0, is_training=False)
    
    print(f"Min Lots: {env.MIN_LOTS}")
    assert env.MIN_LOTS == 0.001, f"Expected MIN_LOTS 0.001, got {env.MIN_LOTS}"
    
    obs,info = env.reset()
    print("Environment Reset. Observation shape:", obs.shape)
    
    # Test Step
    # Action: [SL_Mult, TP_Mult, Risk_Factor]
    # let's try a small risk
    action = np.array([-0.5, -0.5, -0.9], dtype=np.float32) # Low risk
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("\n--- Step 1 Info ---")
    print(f"Reward: {reward}")
    print(f"PnL: {info['pnl']}")
    print(f"Equity: {info['equity']}")
    print(f"Lots: {info['lots']}")
    print(f"Exit Reason: {info['exit']}")
    
    # Check for NaN in obs
    if np.any(np.isnan(obs)):
        print("CRITICAL: NaNs found in observation!")
    else:
        print("Observation NaN check: PASS")
        
    # Test Risk Violation Penalty logic (indirectly)
    # Try max risk
    action_high = np.array([1.0, 1.0, 1.0], dtype=np.float32) # Max risk
    obs, reward, terminated, truncated, info = env.step(action_high)
    
    print("\n--- Step 2 (High Risk) Info ---")
    print(f"Reward: {reward}")
    print(f"Lots: {info['lots']}")
    
    print("\nVerification Complete.")

if __name__ == "__main__":
    test_env()

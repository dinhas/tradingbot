"""
Risk Model V3 - Reward Verification
Tests that the environment loads new columns and calculates rewards correctly.
"""
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from risk_env import RiskManagementEnv

def test_rewards():
    print("Initializing Environment...")
    # Use the small test dataset we generated
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'test_v3_final.parquet')
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Run generate_risk_dataset.py first.")
        return

    env = RiskManagementEnv(dataset_path=dataset_path, initial_equity=100.0, is_training=True)
    obs, _ = env.reset()
    print(f"Observation Shape: {obs.shape}")
    assert obs.shape == (70,), f"Expected (70,), got {obs.shape}"
    
    # Test 1: Block Action
    print("\n--- Test 1: Block Action ---")
    # Action: SL=1.0, TP=1.0, Risk=0.0 (Block), Size=0.5
    action_block = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32) 
    obs, reward, term, trunc, info = env.step(action_block)
    print(f"Block Reward: {reward:.4f}")
    assert info['is_blocked'], "Action should be blocked"
    assert -3.0 <= reward <= 3.0, f"Block reward {reward} out of range [-3, 3]"

    # Test 2: Trade Action (Winner)
    print("\n--- Test 2: Trade Action ---")
    # Action: SL=1.0, TP=1.0, Risk=0.5 (Trade), Size=1.0 (Max)
    action_trade = np.array([-1.0, -1.0, 0.0, 1.0], dtype=np.float32)
    
    # Force a favorable setup if possible, or just observe
    obs, reward, term, trunc, info = env.step(action_trade)
    print(f"Trade Reward: {reward:.4f}")
    print(f"PnL: {info['pnl']:.4f}")
    print(f"Position Size Factor: {info['size_f']:.4f}")
    print(f"Sizing Reward: {info['siz_rew']:.4f}")
    
    assert -10.0 <= reward <= 10.0, f"Trade reward {reward} out of range [-10, 10]"
    
    # Validation: Sizing reward should have same sign as PnL
    if info['pnl'] != 0:
        assert np.sign(info['siz_rew']) == np.sign(info['pnl']), "Sizing reward sign mismatch"

    print("\n✅ Reward Verification PASSED")

if __name__ == "__main__":
    test_rewards()

"""
Risk Model V3 - Smoke Test
Tests that the environment can be instantiated with the new 70-dim observation space
and 4-dim action space without errors.

NOTE: This requires a dataset with 40-dim features. Will create a minimal dummy dataset
for testing purposes.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import tempfile

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import using absolute path within project structure
import os
os.chdir(str(project_root))
from src.risk_env import RiskManagementEnv

def create_dummy_dataset(n_rows=150):
    """Create a minimal dummy dataset with 40-dim features per row"""
    print(f"Creating dummy dataset with {n_rows} rows...")
    
    data = {
        'direction': np.random.choice([1.0, -1.0], n_rows),
        'entry_price': np.random.uniform(1.0, 2.0, n_rows),
        'atr': np.random.uniform(0.0001, 0.01, n_rows),
        'max_profit_pct': np.random.uniform(0.001, 0.03, n_rows),
        'max_loss_pct': np.random.uniform(-0.03, -0.001, n_rows),
        'close_1000_price': np.random.uniform(1.0, 2.0, n_rows),
        'pair': np.random.choice(['EURUSD', 'GBPUSD', 'XAUUSD'], n_rows),
        # V3: 40-dim features (25 asset + 15 global)
        'features': [np.random.randn(40).astype(np.float32) for _ in range(n_rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Save to temp parquet
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet')
    df.to_parquet(tmpfile.name)
    print(f"Dataset saved to: {tmpfile.name}")
    return tmpfile.name

def test_environment_init():
    """Test 1: Can we create the environment?"""
    print("\n=== Test 1: Environment Initialization ===")
    dataset_path = create_dummy_dataset(150)
    
    try:
        env = RiskManagementEnv(
            dataset_path=dataset_path,
            initial_equity=10.0,
            is_training=True
        )
        print(f"[OK] Environment created successfully")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Expected: obs=(70,), action=(4,)")
        
        # Validate dimensions
        assert env.observation_space.shape == (70,), f"Expected obs shape (70,), got {env.observation_space.shape}"
        assert env.action_space.shape == (4,), f"Expected action shape (4,), got {env.action_space.shape}"
        print("[OK] Dimensions are correct!")
        
        return env, dataset_path
    except Exception as e:
        print(f"[FAIL] Environment initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, dataset_path

def test_reset(env):
    """Test 2: Can we reset the environment?"""
    print("\n=== Test 2: Environment Reset ===")
    try:
        obs, info = env.reset()
        print(f"[OK] Reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation min: {obs.min():.4f}, max: {obs.max():.4f}")
        print(f"   Any NaN: {np.isnan(obs).any()}, Any Inf: {np.isinf(obs).any()}")
        
        assert obs.shape == (70,), f"Expected obs shape (70,), got {obs.shape}"
        assert not np.isnan(obs).any(), "Observation contains NaN"
        assert not np.isinf(obs).any(), "Observation contains Inf"
        print("[OK] Observation is valid!")
        
        return True
    except Exception as e:
        print(f"[FAIL] Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step(env):
    """Test 3: Can we take steps?"""
    print("\n=== Test 3: Environment Step ===")
    try:
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {i+1}:")
            print(f"  Action: {action}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Obs shape: {obs.shape}")
            print(f"  Terminated: {terminated}, Truncated: {truncated}")
            print(f"  Info keys: {list(info.keys())}")
            
            assert obs.shape == (70,), f"Expected obs shape (70,), got {obs.shape}"
            assert not np.isnan(obs).any(), "Observation contains NaN"
            assert not np.isinf(obs).any(), "Observation contains Inf"
            
            if terminated or truncated:
                print("  Episode ended, resetting...")
                env.reset()
        
        print("[OK] All steps completed successfully!")
        return True
    except Exception as e:
        print(f"[FAIL] Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("RISK MODEL V3 - SMOKE TEST")
    print("="*60)
    
    # Test 1: Init
    env, dataset_path = test_environment_init()
    if env is None:
        print("\n[FAIL] SMOKE TEST FAILED - Environment initialization failed")
        return False
    
    # Test 2: Reset
    if not test_reset(env):
        print("\n[FAIL] SMOKE TEST FAILED - Reset failed")
        return False
    
    # Test 3: Step
    if not test_step(env):
        print("\n[FAIL] SMOKE TEST FAILED - Step failed")
        return False
    
    print("\n" + "="*60)
    print("[OK] ALL SMOKE TESTS PASSED!")
    print("="*60)
    print("\nRisk Model V3 environment is working correctly with:")
    print("  - 70-dim observation space (40 market + 5 account + 5 PnL + 20 action history)")
    print("  - 4-dim action space (SL, TP, Risk, Position_Size)")
    print("\nReady for dataset generation and training!")
    
    # Cleanup
    import os
    try:
        os.unlink(dataset_path)
        # Also try to remove cache
        cache_dir = dataset_path.replace('.parquet', '') + '_cache'
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
    except:
        pass
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


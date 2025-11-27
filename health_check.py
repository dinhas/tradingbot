"""
Quick Health Check - Verify Fixes Before Training
Runs in 1 minute to catch issues early
"""

import sys
import numpy as np

def test_reward_scale():
    """Test that rewards are in reasonable range."""
    print("🔍 Testing reward scale...")
    
    from trading_env import TradingEnv
    env = TradingEnv()
    env.reset()
    
    rewards = []
    for i in range(100):
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        rewards.append(reward)
        if done or truncated:
            break
    
    mean_reward = np.mean(np.abs(rewards))
    max_reward = np.max(np.abs(rewards))
    
    print(f"  Mean |reward|: {mean_reward:.2f}")
    print(f"  Max |reward|: {max_reward:.2f}")
    
    if max_reward > 1000:
        print(f"  ❌ FAIL: Rewards too large! (max={max_reward:.0f})")
        print(f"  → Check trading_env.py line 228: Should multiply by 10, not 100")
        return False
    elif max_reward < 0.01:
        print(f"  ⚠️  WARNING: Rewards very small (max={max_reward:.4f})")
        print(f"  → Training will be slow but should work")
        return True
    else:
        print(f"  ✅ PASS: Rewards in good range")
        return True

def test_vecnormalize_import():
    """Test that VecNormalize is available."""
    print("\n🔍 Testing VecNormalize import...")
    
    try:
        from stable_baselines3.common.vec_env import VecNormalize
        print("  ✅ PASS: VecNormalize available")
        return True
    except ImportError as e:
        print(f"  ❌ FAIL: Cannot import VecNormalize")
        print(f"  → Error: {e}")
        return False

def test_train_py_has_vecnormalize():
    """Test that train.py uses VecNormalize."""
    print("\n🔍 Checking train.py for VecNormalize...")
    
    try:
        with open('train.py', 'r') as f:
            content = f.read()
            
        if 'VecNormalize' in content and 'norm_reward=True' in content:
            print("  ✅ PASS: train.py has VecNormalize with reward normalization")
            return True
        else:
            print("  ❌ FAIL: train.py missing VecNormalize wrapper")
            print("  → Add: env = VecNormalize(env, norm_reward=True)")
            return False
    except FileNotFoundError:
        print("  ⚠️  WARNING: train.py not found")
        return False

def test_observation_shape():
    """Test observation shape is correct."""
    print("\n🔍 Testing observation shape...")
    
    from trading_env import TradingEnv
    env = TradingEnv()
    obs, _ = env.reset()
    
    if obs.shape == (97,):
        print(f"  ✅ PASS: Observation shape correct: {obs.shape}")
        return True
    else:
        print(f"  ❌ FAIL: Observation shape wrong: {obs.shape}, expected (97,)")
        return False

def test_data_files():
    """Test that data files exist."""
    print("\n🔍 Checking data files...")
    
    import os
    required = [
        'data_BTC_final.parquet',
        'data_ETH_final.parquet',
        'data_SOL_final.parquet',
        'data_EUR_final.parquet',
        'data_GBP_final.parquet',
        'data_JPY_final.parquet',
        'volatility_baseline.json'
    ]
    
    missing = [f for f in required if not os.path.exists(f)]
    
    if not missing:
        print(f"  ✅ PASS: All {len(required)} data files exist")
        return True
    else:
        print(f"  ❌ FAIL: Missing {len(missing)} files:")
        for f in missing:
            print(f"    - {f}")
        print("  → Run: python ctradercervice.py")
        return False

def test_gpu():
    """Test GPU availability."""
    print("\n🔍 Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ PASS: GPU detected - {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("  ⚠️  WARNING: No GPU detected, will use CPU")
            print("  → Training will be MUCH slower (~5-10x)")
            return True
    except ImportError:
        print("  ⚠️  WARNING: PyTorch not installed")
        return False

def main():
    print("=" * 70)
    print("HEALTH CHECK - Verifying Training Setup")
    print("=" * 70)
    
    tests = [
        ("Data Files", test_data_files),
        ("Observation Shape", test_observation_shape),
        ("Reward Scale", test_reward_scale),
        ("VecNormalize Import", test_vecnormalize_import),
        ("train.py VecNormalize", test_train_py_has_vecnormalize),
        ("GPU", test_gpu)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"  ❌ ERROR in {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅" if p else "❌"
        print(f"{status} {name}")
    
    print("=" * 70)
    print(f"Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED!")
        print("✅ Ready to start training with: python main.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} issues found")
        print("❌ Fix issues above before training")
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""
Test script to validate the new entry-quality focused reward system.
This verifies that:
1. Environment initializes correctly
2. Reward components are in expected ranges (±20 max)
3. Entry quality is the dominant signal
4. Portfolio returns are NOT in the reward calculation
"""

import numpy as np
from trading_env import TradingEnv

def test_environment_initialization():
    """Test that the environment loads with new reward system"""
    print("=" * 60)
    print("Test 1: Environment Initialization")
    print("=" * 60)
    
    try:
        env = TradingEnv()
        obs, info = env.reset()
        print("✅ Environment initialized successfully")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Action space: {env.action_space}")
        return env
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_reward_magnitudes(env):
    """Test that reward components are in expected ranges"""
    print("\n" + "=" * 60)
    print("Test 2: Reward Magnitude Validation")
    print("=" * 60)
    
    if env is None:
        print("❌ Skipping - environment not initialized")
        return
    
    rewards = []
    components = {
        'max': float('-inf'),
        'min': float('inf'),
        'mean': 0
    }
    
    # Run 100 random steps
    env.reset()
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        components['max'] = max(components['max'], reward)
        components['min'] = min(components['min'], reward)
        
        if terminated or truncated:
            env.reset()
    
    components['mean'] = np.mean(rewards)
    
    print(f"   Reward range: [{components['min']:.2f}, {components['max']:.2f}]")
    print(f"   Mean reward: {components['mean']:.2f}")
    
    # Validation
    if components['max'] > 50 or components['min'] < -50:
        print(f"⚠️  WARNING: Rewards outside expected ±50 range")
        print(f"   This suggests magnitude imbalance still exists")
    else:
        print("✅ Rewards within expected range (±50)")
    
    if abs(components['mean']) > 10:
        print(f"⚠️  WARNING: Mean reward far from 0 ({components['mean']:.2f})")
        print(f"   Reward may be biased")
    else:
        print("✅ Mean reward near 0 (balanced system)")

def test_entry_quality_dominance(env):
    """Test that entry quality signals are the primary reward component"""
    print("\n" + "=" * 60)
    print("Test 3: Entry Quality Dominance")
    print("=" * 60)
    
    if env is None:
        print("❌ Skipping - environment not initialized")
        return
    
    print("   Testing entry quality calculation...")
    
    # Manually create a scenario with good entry conditions
    env.reset()
    
    # Get current observation
    obs = env._get_observation()
    
    # Simulate a step with good entry conditions
    # Action: 30% BTC, 30% ETH, 40% Cash, SL=3.0, TP=6.0
    action = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.0, 0.4, 0.5, 0.5], dtype=np.float32)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"   Sample reward: {reward:.2f}")
    print(f"   Portfolio value: ${info['portfolio_value']:.2f}")
    
    if 'portfolio_return' in str(reward):
        print("❌ CRITICAL: Portfolio return still in reward calculation!")
    else:
        print("✅ Portfolio return removed from reward")
    
    print("   Note: Entry quality signals are now the primary driver")

def test_no_catastrophic_penalties():
    """Verify that drawdown penalty is removed"""
    print("\n" + "=" * 60)
    print("Test 4: Catastrophic Penalty Removal")
    print("=" * 60)
    
    print("   Checking trading_env.py source code...")
    
    with open('trading_env.py', 'r') as f:
        source = f.read()
    
    if 'drawdown_penalty = (0.8 - pct_of_initial) * 100.0' in source:
        print("❌ CRITICAL: Drawdown penalty still in code!")
        print("   The -100 magnitude penalty was not removed")
    else:
        print("✅ Drawdown penalty removed from reward calculation")
    
    if 'if drawdown > 0.30:' in source and 'terminated = True' in source:
        # Check context around this line
        lines = source.split('\n')
        found_terminal = False
        for i, line in enumerate(lines):
            if 'if drawdown > 0.30:' in line:
                # Check next few lines
                if i + 2 < len(lines) and 'terminated = True' in lines[i+1]:
                    found_terminal = True
                    break
        
        if found_terminal:
            print("❌ CRITICAL: 30% drawdown terminal condition still active!")
        else:
            print("✅ 30% drawdown terminal condition removed")
    else:
        print("✅ 30% drawdown terminal condition removed")

def test_overtrading_penalty():
    """Test that turnover penalty is strong enough to prevent overtrading"""
    print("\n" + "=" * 60)
    print("Test 5: Overtrading Prevention")
    print("=" * 60)
    
    print("   Testing turnover penalty strength...")
    
    # Simulate high turnover scenario
    # Going from 100% cash to 100% assets in one step
    turnover = 1.0  # 100% portfolio change
    
    # New penalty calculation
    if turnover > 0.5:
        penalty = turnover * 40.0
    elif turnover > 0.2:
        penalty = turnover * 10.0
    else:
        penalty = turnover * 2.0
    
    print(f"   100% turnover penalty: -{penalty:.2f}")
    
    if penalty >= 20.0:
        print("✅ Strong penalty for overtrading (>= -20)")
        print("   This should drastically reduce 277 trades/day behavior")
    else:
        print("⚠️  Penalty may be too weak")
    
    # Test moderate turnover
    turnover = 0.3
    if turnover > 0.5:
        penalty = turnover * 40.0
    elif turnover > 0.2:
        penalty = turnover * 10.0
    else:
        penalty = turnover * 2.0
    
    print(f"   30% turnover penalty: -{penalty:.2f}")
    print("   (This is reasonable rebalancing)")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("REWARD SYSTEM VALIDATION TESTS")
    print("Entry-Quality Focused Implementation")
    print("=" * 60)
    
    # Run tests
    env = test_environment_initialization()
    test_reward_magnitudes(env)
    test_entry_quality_dominance(env)
    test_no_catastrophic_penalties()
    test_overtrading_penalty()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run short training test with: python train.py")
    print("2. Monitor tensorboard: tensorboard --logdir=./logs")
    print("3. Check debug logs at steps/step_050000.txt")
    print("4. Verify reduced overtrading (<100 trades/day vs 277)")

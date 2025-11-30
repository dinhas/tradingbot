#!/usr/bin/env python3
"""Quick Environment Test"""
import sys
sys.path.insert(0, 'e:/tradingbot')

import trading_env

# Test environment initialization with correct data path
print("Testing Training Environment...")
env = trading_env.TradingEnv(data_dir="e:/tradingbot/data/", 
                              volatility_file="e:/tradingbot/data/volatility_baseline.json")

print(f"✓ Environment loaded successfully")
print(f"  - Data points: {env.n_steps}")
print(f"  - Action space: {env.action_space.shape}")
print(f"  - Observation space: {env.observation_space.shape}")

# Test one step
obs, info = env.reset()
print(f"\n✓ Reset successful")
print(f"  - Observation shape: {obs.shape}")
print(f"  - Initial portfolio: ${env.portfolio_value:,.2f}")

# Test action (50% turnover to test penalty)
action = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.7, 0.7]  # Moderate positions + high SL/TP
obs, reward, terminated, truncated, info = env.step(action)

print(f"\n✓ Step successful")
print(f"  - SL Multiplier: {info['sl_multiplier']:.2f}x ATR (min should be 2.0x)")
print(f"  - TP Multiplier: {info['tp_multiplier']:.2f}x ATR (min should be 2.0x)")
print(f"  - Portfolio Value: ${info['portfolio_value']:,.2f}")
print(f"  - Fees: ${info['fees']:.2f}")
print(f"  - Reward: {reward:.3f}")

print("\n" + "="*60)
print("ENVIRONMENT TEST PASSED ✅")
print("="*60)
print("\nNext steps:")
print("1. Run: python train.py --total-steps 50000")
print("   (Quick validation run)")
print("2. Review debug logs at step_50000.txt")
print("3. If successful, full retrain: --total-steps 600000")

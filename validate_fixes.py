import numpy as np
from src.trading_env import TradingEnv
import logging

# Configure logging to see the debug output from reward calculation
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')

# Test 1: Initialization
print("Test 1: Initialization")
env = TradingEnv(stage=1, is_training=True)
print(f"✓ Leverage initialized: {env.leverage}")

# Test 2: Transaction costs
print("\nTest 2: Transaction Cost Check")
obs = env.reset()
initial_equity = env.equity
action = np.array([0.8, 0, 0, 0, 0])  # Open EURUSD position
obs, reward, done, trunc, info = env.step(action)
cost_per_trade = initial_equity - env.equity
print(f"Position opened, cost: ${cost_per_trade:.2f}")
print(f"✓ Cost reasonable: {cost_per_trade < 10}")  # Should be ~$5, not $25

# Test 3: Churn detection
print("\nTest 3: Churn Detection")
action = np.array([0, 0, 0, 0, 0])  # Close
obs, reward1, _, _, _ = env.step(action)
action = np.array([0.8, 0, 0, 0, 0])  # Reopen immediately
obs, reward2, _, _, _ = env.step(action)
print(f"Reward without churn: {reward1:.6f}")
print(f"Reward with churn: {reward2:.6f}")
print(f"✓ Churn penalty applied: {reward2 < reward1}")

# Test 4: Reward distribution
print("\nTest 4: Reward Distribution (200 random steps)")
env.reset()
rewards = []
for i in range(200):
    action = env.action_space.sample()
    obs, reward, done, trunc, info = env.step(action)
    rewards.append(reward)
    if done or trunc:
        break

print(f"Mean:  {np.mean(rewards):+.6f}")
print(f"Std:   {np.std(rewards):.6f}")
print(f"Min:   {np.min(rewards):+.6f}")
print(f"Max:   {np.max(rewards):+.6f}")
print(f"Positive: {100*np.mean(np.array(rewards)>0):.1f}%")
print(f"✓ Mean near zero: {abs(np.mean(rewards)) < 0.005}")
# Relaxed check for positivity as random actions might lose money
print(f"✓ Reasonable positivity: {20 < 100*np.mean(np.array(rewards)>0) < 80}")

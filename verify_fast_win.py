import sys
from pathlib import Path
import numpy as np

# Add Alpha/src to path
sys.path.append(str(Path(__file__).parent / "Alpha" / "src"))

from trading_env import TradingEnv

def test_fast_win_reward():
    # Create a dummy environment
    env = TradingEnv(is_training=True)
    env.reset()
    
    # Manually set start_equity to simplify math
    env.start_equity = 10000.0
    env.equity = 10000.0
    
    # Mock a trade outcome that is a FAST TP
    # Suppose TP is hit in 5 bars with $100 profit
    fast_outcome = {
        'exit_reason': 'TP',
        'bars_held': 5,
        'pnl': 100.0,
        'closed': True
    }
    
    # We need to monkeypatch _simulate_trade_outcome_with_timing to return this
    original_simulate = env._simulate_trade_outcome_with_timing
    env._simulate_trade_outcome_with_timing = lambda asset: fast_outcome
    
    # Execute a "Buy" action (0.5 results in 1)
    # _parse_action([0.5]) returns {'direction': 1, ...}
    obs, reward, done, truncated, info = env.step(np.array([0.5]))
    
    print(f"--- Fast Win Test (5 bars, $100 profit) ---")
    print(f"Total Reward (Step 0): {reward:.4f}")
    
    # Mock a trade outcome that is a SLOW TP
    # Suppose TP is hit in 100 bars with $100 profit
    slow_outcome = {
        'exit_reason': 'TP',
        'bars_held': 100,
        'pnl': 100.0,
        'closed': True
    }
    
    env.reset()
    env.start_equity = 10000.0
    env.equity = 10000.0
    env._simulate_trade_outcome_with_timing = lambda asset: slow_outcome
    
    obs, reward, done, truncated, info = env.step(np.array([0.5]))
    print(f"\n--- Slow Win Test (100 bars, $100 profit) ---")
    print(f"Initial Reward (Step 0): {reward:.4f}")
    
    # Now simulate the rest of the 100 bars
    cumulative_reward = reward
    for i in range(99):
        # Action 0 (Flat) should HOLD if FORCE_HOLD is True
        obs, reward, done, truncated, info = env.step(np.array([0.0]))
        cumulative_reward += reward
        
    print(f"Cumulative Reward over 100 steps: {cumulative_reward:.4f}")

if __name__ == "__main__":
    try:
        test_fast_win_reward()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

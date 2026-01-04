import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from LiveExecution.src.features import FeatureManager
from Alpha.src.trading_env import TradingEnv

def compare_raw_features():
    print("=== Lightweight Feature Comparison: Training vs Live ===")
    
    # 1. Load Data and Align (Small window to save memory)
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    dfs = {}
    common_index = None
    
    print("Loading data tails...")
    for asset in assets:
        # Load last 1500 rows to ensure enough context for technical indicators and TradingEnv reset
        df = pd.read_parquet(project_root / "data" / f"{asset}_5m.parquet").tail(1500)
        dfs[asset] = df
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    # Ensure we have enough bars
    if len(common_index) < 1000:
        print(f"Error: Not enough common bars ({len(common_index)})")
        return

    # Use a fixed slice of 1000 bars
    test_size = min(len(common_index), 1000)
    common_index = common_index[-test_size:]
    data_dict = {asset: dfs[asset].loc[common_index] for asset in assets}
    print(f"Using {len(common_index)} common bars for comparison.")

    # 2. Portfolio State Mock
    # Matching the structure expected by FeatureEngine.get_observation
    portfolio_state = {
        'equity': 10000.0,
        'initial_equity': 10000.0,
        'peak_equity': 10000.0,
        'margin_usage_pct': 0.0,
        'drawdown': 0.0,
        'num_open_positions': 0,
        'EURUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0, 'entry_price': 0, 'current_sl': 0, 'current_tp': 0},
        'GBPUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0, 'entry_price': 0, 'current_sl': 0, 'current_tp': 0},
        'USDJPY': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0, 'entry_price': 0, 'current_sl': 0, 'current_tp': 0},
        'USDCHF': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0, 'entry_price': 0, 'current_sl': 0, 'current_tp': 0},
        'XAUUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0, 'entry_price': 0, 'current_sl': 0, 'current_tp': 0}
    }

    # 3. Live Execution Path
    print("\n--- Live Execution Path ---")
    fm = FeatureManager()
    for asset in assets:
        for ts, row in data_dict[asset].iterrows():
            candle = row.to_dict()
            candle['timestamp'] = ts
            fm.push_candle(asset, candle)
    
    # Get the raw observation for the last step
    live_obs = fm.get_alpha_observation('EURUSD', portfolio_state)
    
    # 4. Training Env Path
    print("\n--- Training Env Path ---")
    # Note: TradingEnv internal state is initialized to match our mock
    env = TradingEnv(data=data_dict, is_training=False)
    # TradingEnv.reset() sets equity to 10000.0 and peak_equity to 10000.0
    env.reset()
    # Align to same asset and last step
    env.set_asset('EURUSD')
    env.current_step = len(common_index) - 1
    
    train_obs = env._get_observation()

    # 5. Verification
    print("\n--- Verification Results ---")
    diff = np.abs(train_obs - live_obs)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    
    # Compare feature by feature for the 40 features
    if max_diff < 1e-6:
        print("✅ SUCCESS: Features match perfectly!")
    else:
        print("❌ FAILURE: Discrepancies found.")
        # Find which features differ
        mismatches = np.where(diff > 1e-6)[0]
        print(f"Number of mismatching features: {len(mismatches)} / 40")
        
        # Detail the first few mismatches
        print("\nMismatch Details (Indices [0-24] Asset, [25-39] Global):")
        for idx in mismatches[:10]:
            print(f"Feature {idx:2d}: Train={train_obs[idx]:10.6f}, Live={live_obs[idx]:10.6f}, Diff={diff[idx]:10.6f}")

if __name__ == "__main__":
    compare_raw_features()

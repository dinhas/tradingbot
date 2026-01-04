import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Import paths
import sys
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from LiveExecution.src.features import FeatureManager
from Alpha.src.trading_env import TradingEnv
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

def compare_logic():
    print("=== Feature Comparison: Training vs Live ===")
    
    # 1. Load Data and Align
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    dfs = {}
    common_index = None
    for asset in assets:
        try:
            # Only load the tail to save memory and time
            df = pd.read_parquet(project_root / "data" / f"{asset}_5m.parquet").tail(2000)
            dfs[asset] = df
            print(f"Loaded {asset}: {len(df)} rows")
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        except Exception as e:
            print(f"FAILED to load {asset}: {e}")
            raise e
    
    if common_index is None or len(common_index) == 0:
        print("ERROR: No common bars found in the tails of the files!")
        return
    
    print(f"Total common bars found: {len(common_index)}")
    if len(common_index) > 700:
        common_index = common_index[-700:]
    data_dict = {asset: dfs[asset].loc[common_index] for asset in assets}
    print(f"Using {len(common_index)} common bars for comparison.")
    
    # 2. Portfolio State Mock
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

    # 3. Live Execution Pipeline
    print("\n--- Live Execution Pipeline ---")
    fm = FeatureManager()
    # Populate history
    for asset in assets:
        # FeatureManager.update_data expect a Trendbar object, but we can bypass or use push_candle
        for ts, row in data_dict[asset].iterrows():
            candle = row.to_dict()
            candle['timestamp'] = ts
            fm.push_candle(asset, candle)
    
    live_obs = fm.get_alpha_observation('EURUSD', portfolio_state)
    print(f"Live Observation (raw from FM, 40 features):")
    print(live_obs)

    # 4. Training Environment Pipeline
    print("\n--- Training Environment Pipeline ---")
    def make_env():
        return TradingEnv(data=data_dict, is_training=False)
    
    env = DummyVecEnv([make_env])
    # Load the normalizer used in training
    norm_path = project_root / "models" / "checkpoints" / "alpha" / "ppo_final_vecnormalize.pkl"
    if norm_path.exists():
        env = VecNormalize.load(str(norm_path), env)
        env.training = False
        env.norm_reward = False
        print("Loaded VecNormalize from alpha checkpoint.")
    
    # Initialize the env properly
    env.reset()
    raw_env = env.envs[0].unwrapped
    raw_env.set_asset('EURUSD')
    raw_env.current_step = len(data_dict['EURUSD']) - 1 # Use the very last step
    
    # Get observation from TradingEnv
    train_obs_raw = raw_env._get_observation()
    print(f"Training Obs (raw from env, 40 features):")
    # Show first 5 and last 5 to avoid too much output
    print(f"[{train_obs_raw[0]:.4f}, {train_obs_raw[1]:.4f}, {train_obs_raw[2]:.4f}, ... {train_obs_raw[-2]:.4f}, {train_obs_raw[-1]:.4f}]")
    
    # Now apply VecNormalize to the train_obs
    # In SB3, env.step/reset returns normalized obs. 
    # Since we manually moved the step, we use env.normalize_obs
    train_obs_norm = env.normalize_obs(train_obs_raw.reshape(1, -1))[0]
    print(f"\nTraining Obs (after VecNormalize):")
    print(train_obs_norm)

    # 5. Full Live Pipeline (with VecNormalize)
    # This simulates what ModelLoader does
    live_obs_norm = env.normalize_obs(live_obs.reshape(1, -1))[0]
    print(f"\nLive Obs (after VecNormalize):")
    print(live_obs_norm)

    # 6. Comparison
    diff = np.abs(train_obs_raw - live_obs)
    max_diff = np.max(diff)
    print(f"\nMax difference between raw observations: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✅ RAW FEATURES MATCH!")
    else:
        print("❌ RAW FEATURES DISCREPANCY DETECTED!")
        # Find which features differ
        mismatches = np.where(diff > 1e-5)[0]
        print(f"First 10 mismatches at indices: {mismatches[:10]}")
        for idx in mismatches[:5]:
            print(f"Index {idx}: Train={train_obs_raw[idx]:.6f}, Live={live_obs[idx]:.6f}, Diff={diff[idx]:.6f}")

    diff_norm = np.abs(train_obs_norm - live_obs_norm)
    max_diff_norm = np.max(diff_norm)
    print(f"Max difference between normalized observations: {max_diff_norm:.8f}")
    if max_diff_norm < 1e-5:
        print("✅ NORMALIZED FEATURES MATCH!")
    else:
        print("❌ NORMALIZED FEATURES DISCREPANCY!")

if __name__ == "__main__":
    compare_logic()

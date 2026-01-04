import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from LiveExecution.src.features import FeatureManager
from RiskLayer.src.risk_env import RiskManagementEnv

def compare_risk_features():
    print("=== Risk Feature Comparison: Training vs Live ===")
    
    # 1. Setup paths and data
    risk_dataset_path = project_root / "RiskLayer" / "risk_dataset.parquet"
    if not risk_dataset_path.exists():
        print(f"Error: Risk dataset not found at {risk_dataset_path}")
        return

    # To compare the Alpha features (0-39) we need the raw OHLCV data
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    dfs = {}
    common_index = None
    
    print("Loading data tails for Alpha features...")
    for asset in assets:
        # Load last 1500 rows
        df_path = project_root / "data" / f"{asset}_5m.parquet"
        if not df_path.exists():
            print(f"Error: {df_path} not found")
            return
        # Get total row count first to avoid loading everything if possible
        table = pd.read_parquet(df_path)
        df = table.tail(1500)
        del table # Free memory
        dfs[asset] = df
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    # Align to common index
    if common_index is None or len(common_index) == 0:
        print("Error: No common bars found across assets.")
        return
        
    test_size = min(len(common_index), 500)
    common_index = common_index[-test_size:]
    data_dict = {asset: dfs[asset].loc[common_index] for asset in assets}
    print(f"Using {len(common_index)} bars for Alpha feature reconstruction.")

    # 2. Setup Mock Portfolio State
    # Note: RiskManagementEnv expects Account features (40-44) derived from equity/peak_equity
    portfolio_state = {
        'equity': 10.0,
        'initial_equity': 10.0,
        'peak_equity': 10.0,
        'balance': 10.0,
        'num_open_positions': 0,
        'EURUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0},
        'GBPUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0},
        'USDJPY': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0},
        'USDCHF': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0},
        'XAUUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0}
    }

    # 3. Live Execution Path
    print("\n--- Live Execution Path ---")
    fm = FeatureManager()
    for asset in assets:
        for ts, row in data_dict[asset].iterrows():
            candle = row.to_dict()
            candle['timestamp'] = ts
            fm.push_candle(asset, candle)
    
    # Get the 45-feature risk observation for the last step
    # We pass 'EURUSD' as the target asset (this affects the first 25 features of alpha_obs)
    alpha_obs = fm.get_alpha_observation('EURUSD', portfolio_state)
    live_risk_obs = fm.get_risk_observation('EURUSD', alpha_obs, portfolio_state)
    
    # 4. Training Env Path
    print("\n--- Training Env Path ---")
    # RiskManagementEnv uses a pre-generated Parquet file. 
    # To compare fairly, we need to find the row in the parquet that matches our 'latest_ts'
    # Use cached preprocessed dataframe if the latest timestamp matches
    non_empty_dfs = [df for df in fm.history.values() if not df.empty]
    if not non_empty_dfs:
        print("Error: All asset histories are empty.")
        return
        
    latest_ts = max(df.index.max() for df in non_empty_dfs)
    
    env = RiskManagementEnv(dataset_path=str(risk_dataset_path), initial_equity=10.0, is_training=False)
    # The dataset might not have entries for every timestamp if it was filtered by Alpha.
    # But we can check features_array directly or reset and Step until match.
    
    # Let's try to find a matching feature vector in the dataset if possible, 
    # but the most robust way is to check the Account State logic (Index 40-44)
    # AND the Alpha Feature logic (Index 0-39) separately.
    
    env.reset()
    # Mocking env state to match portfolio_state
    env.equity = 10.0
    env.peak_equity = 10.0
    
    # For index 0-39, we'll use the feature array from the env's current step
    train_risk_obs = env._get_observation()
    
    # 5. Verification
    print("\n--- Verification: Account Features (40-44) ---")
    # [40: equity_norm, 41: drawdown, 42: leverage_placeholder, 43: risk_cap_mult, 44: padding]
    live_account = live_risk_obs[40:]
    train_account = train_risk_obs[40:]
    
    account_diff = np.abs(live_account - train_account)
    print(f"Equity Norm Match:  Live={live_account[0]:.4f}, Train={train_account[0]:.4f}")
    print(f"Drawdown Match:     Live={live_account[1]:.4f}, Train={train_account[1]:.4f}")
    print(f"Risk Cap Match:     Live={live_account[3]:.4f}, Train={train_account[3]:.4f}")
    
    if np.max(account_diff) < 1e-6:
        print("✅ Account Features Match!")
    else:
        print("❌ Account Features Mismatch.")

    print("\n--- Verification: Market Features (0-39) ---")
    # Market features depend on the row in the parquet file.
    # This script verifies that the ASSEMBLY logic (concatenate alpha + account) is correct.
    # To truly verify market features, one would need to align the parquet row with the raw data.
    
    is_concatenated = np.allclose(live_risk_obs[:40], alpha_obs)
    if is_concatenated:
        print("✅ Assembly SUCCESS: Risk observation correctly incorporates Alpha features.")
    else:
        print("❌ Assembly FAILURE: Risk observation does NOT match input Alpha features.")

    print(f"\nLive Observation Size: {len(live_risk_obs)}")
    print(f"Train Observation Size: {len(train_risk_obs)}")
    
    if len(live_risk_obs) == len(train_risk_obs) == 45:
        print("✅ Observation dimensions are consistent (45).")
    else:
        print(f"❌ Dimension mismatch: Live={len(live_risk_obs)}, Train={len(train_risk_obs)}")

if __name__ == "__main__":
    compare_risk_features()

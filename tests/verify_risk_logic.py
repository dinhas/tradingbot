import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from LiveExecution.src.features import FeatureManager

def verify_risk_logic():
    print("=== Risk Logic Verification: Account Features (40-44) ===")
    
    # Setup Mock Portfolio State
    # Note: Live Risk observation [40-44] = [EquityNorm, Drawdown, Leverage, RiskCap, Padding]
    portfolio_state = {
        'equity': 8.0,
        'initial_equity': 10.0,
        'peak_equity': 12.0,
        'balance': 8.0,
        'num_open_positions': 0,
        'EURUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0},
        'GBPUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0},
        'USDJPY': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0},
        'USDCHF': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0},
        'XAUUSD': {'has_position': 0, 'position_size': 0, 'unrealized_pnl': 0, 'position_age': 0}
    }

    fm = FeatureManager()
    
    # 1. Expected results based on RiskManagementEnv logic
    # equity_norm = 8.0 / 10.0 = 0.8
    # drawdown = 1.0 - (8.0 / 12.0) = 0.3333
    # risk_cap_mult = max(0.2, 1.0 - (0.3333 * 2.0)) = 0.3333
    
    expected = [0.8, 0.333333, 0.0, 0.333333, 0.0]
    
    # 2. Live result
    alpha_obs_mock = np.zeros(40, dtype=np.float32)
    live_risk_obs = fm.get_risk_observation('EURUSD', alpha_obs_mock, portfolio_state)
    live_account = live_risk_obs[40:]
    
    print("\n--- Account Feature Match ---")
    headers = ["Feature", "Expected", "Live", "Diff"]
    print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10}")
    
    features = ["EquityNorm", "Drawdown", "Leverage", "RiskCap", "Padding"]
    for i, name in enumerate(features):
        diff = abs(expected[i] - live_account[i])
        print(f"{name:<15} {expected[i]:<10.4f} {live_account[i]:<10.4f} {diff:<10.6f}")

    if np.allclose(live_account, expected, atol=1e-5):
        print("\n✅ SUCCESS: Risk Account features match training logic!")
    else:
        print("\n❌ FAILURE: Mismatch in Risk Account features.")

    # 3. Verify concatenation
    if len(live_risk_obs) == 45 and np.allclose(live_risk_obs[:40], alpha_obs_mock):
        print("✅ SUCCESS: Observation is 45-dim and correctly concatenated.")
    else:
        print("❌ FAILURE: Concatenation or Dimension error.")

if __name__ == "__main__":
    verify_risk_logic()

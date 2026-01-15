
import numpy as np
import pickle
import os
import sys
import pandas as pd
from pathlib import Path

# Ensure project root is in path
sys.path.append(os.getcwd())

from RiskLayer.env.risk_env import RiskTradingEnv

def verify():
    print("=== RISK MODEL FIX VERIFICATION ===")
    
    # 1. Test Feature Order in Environment
    print("\n[1/3] Checking Feature Alignment...")
    try:
        env = RiskTradingEnv(is_training=False, max_rows=100)
        env.reset()
        
        # Get the first asset's columns
        asset = env.assets[0]
        indices = env.asset_col_indices[asset]
        asset_cols = [env.processed_data.columns[i] for i in indices]
        
        # Check first 5 (Should be OHLCV)
        expected_ohlcv = ['open', 'high', 'low', 'close', 'volume']
        found_ohlcv = [col.replace(f"{asset}_", "") for col in asset_cols[:5]]
        
        if found_ohlcv == expected_ohlcv:
            print(f"  SUCCESS: OHLCV is at the start {found_ohlcv}")
            mapping_pass = True
        else:
            print(f"  FAILURE: Expected {expected_ohlcv}, but found {found_ohlcv}")
            mapping_pass = False
            
    except Exception as e:
        print(f"  ERROR during alignment check: {e}")
        mapping_pass = False

    # 2. Test Normalizer Health (If it exists)
    print("\n[2/3] Checking Normalizer Statistics...")
    norm_path = Path("RiskLayer/models/checkpoints/vec_normalize.pkl")
    if not norm_path.exists():
        # Fallback to the one we were inspecting if the new one isn't there yet
        norm_path = Path("models/risk/risk_model_final_vecnormalize.pkl")
        print(f"  NOTE: New normalizer not found at {norm_path.parent}. Checking current model...")

    try:
        with open(norm_path, "rb") as f:
            norm_data = pickle.load(f)
        
        means = norm_data.obs_rms.mean
        
        # Check Index 0 (Price)
        # Should be > 1.0 for most pairs
        price_mean = means[0]
        
        # Check Index 85 (Equity Norm)
        # New code caps it at 5.0. Old code was 1756.
        equity_mean = means[85] if len(means) > 85 else 0
        
        print(f"  Price Mean (Idx 0): {price_mean:.4f}")
        print(f"  Equity Mean (Idx 85): {equity_mean:.4f}")
        
        if equity_mean > 100:
            print("  FAILURE: Equity Mean is still corrupted (> 100). Model needs retraining with fixed code.")
            stats_pass = False
        elif equity_mean == 0:
             print("  WARNING: Normalizer shape mismatch (older 45-dim model?)")
             stats_pass = False
        else:
            print("  SUCCESS: Equity Mean is healthy.")
            stats_pass = True
            
    except Exception as e:
        print(f"  ERROR during stats check: {e}")
        stats_pass = False

    # 3. Value Distribution Check
    print("\n[3/3] Checking Observation Consistency...")
    try:
        obs, _ = env.reset()
        # Verify no NaNs and reasonable ranges
        if np.isnan(obs).any():
            print("  FAILURE: NaNs found in observation!")
            obs_pass = False
        else:
            # Check if equity_norm (last features) is around 1.0
            equity_val = obs[83] # spread, id, equity_norm (idx 83 in 86-dim)
            if 0.5 <= obs[-3] <= 1.5: # equity_norm is 3rd from last
                print(f"  SUCCESS: Reset Observation Equity: {obs[-3]:.2f}")
                obs_pass = True
            else:
                print(f"  WARNING: Observation Equity {obs[-3]:.2f} is outside expected reset range (0.5-1.5)")
                obs_pass = True # Warning only as it depends on initial_equity
                
    except Exception as e:
        print(f"  ERROR during observation check: {e}")
        obs_pass = False

    # Final Verdict
    print("\n" + "="*30)
    if mapping_pass and stats_pass and obs_pass:
        print("FINAL STATUS: PASS")
        print("Your Risk Model is now correctly aligned and ready for training.")
    elif mapping_pass and not stats_pass:
        print("FINAL STATUS: CODE FIXED / STATS OLD")
        print("The code is fixed, but you are still using the OLD normalizer.")
        print("ACTION: Please run: python RiskLayer/train/train_risk_model.py --steps 100000")
    else:
        print("FINAL STATUS: FAIL")
        print("Please check the errors above.")
    print("="*30)

if __name__ == "__main__":
    verify()

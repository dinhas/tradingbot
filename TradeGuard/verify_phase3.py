import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from generate_dataset import FeatureEngine

def verify_phase3():
    print("Verifying Phase 3: Regime & Session Features")
    print("-" * 50)
    
    fe = FeatureEngine()
    
    # --- 1. Verify Market Regime (Features 21-30) ---
    print("\n[Market Regime Features (21-30)]")
    # Create trending data
    closes = np.linspace(1.1000, 1.1200, 200) # Strong uptrend
    df = pd.DataFrame({
        'open': closes - 0.0005,
        'high': closes + 0.0010,
        'low': closes - 0.0010,
        'close': closes,
        'volume': [1000] * 200
    })
    
    regime_feats = fe.calculate_market_regime(df)
    
    labels_regime = [
        "ADX (14)", "DI+", "DI-", "Aroon Up", "Aroon Down",
        "Hurst (100)", "Efficiency Ratio", "LinReg Slope",
        "Trend Bias (SMA200)", "BB Width"
    ]
    
    for label, val in zip(labels_regime, regime_feats):
        print(f"{label:20}: {val:.4f}")
        
    # --- 2. Verify Session Edge (Features 31-40) ---
    print("\n[Session Edge Features (31-40)]")
    # Mock Time: London Open (08:00 UTC)
    ts = pd.Timestamp("2024-06-19 08:30:00")
    print(f"Timestamp: {ts}")
    
    session_feats = fe.calculate_session_edge(ts)
    
    labels_session = [
        "Hour Sin", "Hour Cos", "DoW Sin", "DoW Cos",
        "Is London", "Is NY", "Is Asian", "Minute Prog",
        "Market Overlap", "Day Progress"
    ]
    
    for label, val in zip(labels_session, session_feats):
        print(f"{label:20}: {val:.4f}")
        
    print("-" * 50)
    print("Verification Complete.")

if __name__ == "__main__":
    verify_phase3()

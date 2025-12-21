
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.tradeguard_feature_builder import TradeGuardFeatureBuilder

def verify():
    print("Verifying Phase 2: Feature Engineering & Parity...")
    
    # 1. Setup dummy data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="5min")
    df_dict = {
        'EURUSD': pd.DataFrame({
            'open': np.random.rand(100), 'high': np.random.rand(100),
            'low': np.random.rand(100), 'close': np.random.rand(100),
            'volume': np.random.rand(100)
        }, index=dates)
    }
    
    # 2. Instantiate builder
    builder = TradeGuardFeatureBuilder(df_dict)
    print(f"Precomputed features for {len(df_dict)} assets.")
    
    # 3. Build features for a step
    portfolio_state = {
        'equity': 10000.0, 'peak_equity': 10000.0, 'total_exposure': 0,
        'open_positions_count': 0, 'recent_trades': [],
        'asset_action_raw': 0.5, 'asset_recent_actions': [0.5]*5,
        'asset_signal_persistence': 1.0, 'asset_signal_reversal': 0.0
    }
    trade_info = {'entry_price': 1.1, 'sl': 1.09, 'tp': 1.12}
    
    features = builder.build_features('EURUSD', 50, trade_info, portfolio_state)
    
    print(f"Feature vector length: {len(features)}")
    if len(features) == 60:
        print("Verified: Feature vector has correct length (60).")
        print("Phase 2 Verification SUCCESSFUL.")
        return True
    else:
        print(f"ERROR: Expected 60 features, got {len(features)}")
        return False

if __name__ == "__main__":
    if verify():
        sys.exit(0)
    else:
        sys.exit(1)

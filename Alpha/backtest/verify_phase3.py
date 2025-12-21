
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import unittest.mock
import tempfile
import shutil

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.backtest_full_system import FullSystemBacktest

def verify():
    print("Verifying Phase 3: Integrated Backtest Loop...")
    
    test_dir = tempfile.mkdtemp()
    try:
        data_dir = os.path.join(test_dir, "data")
        os.makedirs(data_dir)
        
        # 1. Setup dummy data
        assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="5min")
        for asset in assets:
            df = pd.DataFrame({
                'open': np.random.rand(1000) + 1.1,
                'high': np.random.rand(1000) + 1.2,
                'low': np.random.rand(1000) + 1.0,
                'close': np.random.rand(1000) + 1.1,
                'volume': np.random.rand(1000) * 1000
            }, index=dates)
            df.to_parquet(os.path.join(data_dir, f"{asset}_5m.parquet"))

        # 2. Mock models
        with (unittest.mock.patch('Alpha.backtest.backtest_full_system.PPO.load') as mock_ppo_load,
              unittest.mock.patch('Alpha.backtest.backtest_full_system.load_tradeguard_model') as mock_guard_load):
            
            mock_alpha = unittest.mock.Mock()
            mock_alpha.predict.return_value = (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), None)
            
            mock_risk = unittest.mock.Mock()
            mock_risk.predict.return_value = (np.array([1.0, 1.0, 1.0]), None)
            
            mock_ppo_load.side_effect = [mock_alpha, mock_risk]
            
            mock_guard_model = unittest.mock.Mock()
            # Set prob to 0.1, threshold to 0.5 -> Should block
            mock_guard_model.predict.return_value = np.array([0.1])
            mock_guard_load.return_value = (mock_guard_model, {'threshold': 0.5})
            
            print("Initializing FullSystemBacktest...")
            bt = FullSystemBacktest(
                "dummy_alpha", "dummy_risk", "dummy_guard", "dummy_meta",
                data_dir=data_dir
            )
            
            print("Running backtest for 5 steps...")
            bt.env.max_steps = 205
            bt.run_backtest(episodes=1)
            
            print(f"Blocked trades: {len(bt.blocked_trades)}")
            
            if len(bt.blocked_trades) > 0:
                print(f"Verified: TradeGuard blocked {len(bt.blocked_trades)} signals as expected.")
                print("Phase 3 Verification SUCCESSFUL.")
                return True
            else:
                print("ERROR: No signals were blocked despite low probability.")
                return False
                
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    if verify():
        sys.exit(0)
    else:
        sys.exit(1)


import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import unittest.mock
import tempfile
import shutil
from datetime import datetime, timedelta

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.backtest_full_system import FullSystemBacktest

def verify():
    print("Verifying Phase 5: Metrics & Comparative Analysis...")
    
    test_dir = tempfile.mkdtemp()
    try:
        data_dir = os.path.join(test_dir, "data")
        os.makedirs(data_dir)
        
        # 1. Setup dummy data
        assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="5min")
        for asset in assets:
            df = pd.DataFrame({
                'open': np.ones(1000) * 1.1,
                'high': np.ones(1000) * 1.101,
                'low': np.ones(1000) * 1.099,
                'close': np.ones(1000) * 1.1,
                'volume': np.ones(1000) * 1000
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
            # Force block
            mock_guard_model.predict.return_value = np.array([0.1])
            mock_guard_load.return_value = (mock_guard_model, {'threshold': 0.5})
            
            bt = FullSystemBacktest(
                "dummy_alpha", "dummy_risk", "dummy_guard", "dummy_meta",
                data_dir=data_dir
            )
            
            # Mock peek-ahead to return a loss (so it's a good block)
            bt.env._simulate_trade_outcome_with_timing = unittest.mock.Mock()
            bt.env._simulate_trade_outcome_with_timing.return_value = {
                'pnl': -0.01, 'reason': 'sl', 'exit_step': 210
            }
            
            print("Running backtest for metrics check...")
            bt.env.max_steps = 210
            metrics_tracker = bt.run_backtest(episodes=1)
            
            print("Calculating metrics...")
            results = metrics_tracker.calculate_metrics()
            
            # Check for new metric fields
            if 'tradeguard' in results and 'baseline_return' in results:
                print("SUCCESS: Found 'tradeguard' and 'baseline_return' in metrics.")
                print(f"Approval Rate: {results['tradeguard']['approval_rate']:.2%}")
                print(f"Baseline Return: {results['baseline_return']:.2%}")
                print(f"Full System Return: {results['total_return']:.2%}")
                print(f"Net Value-Add vs Baseline: {results['net_value_add_vs_baseline']:.2%}")
                
                if results['total_return'] > results['baseline_return']:
                    print("Verified: Full System outperformed Baseline (as expected for good blocks).")
                
                print("Phase 5 Verification SUCCESSFUL.")
                return True
            else:
                print("ERROR: Missing extended metrics in results.")
                print(f"Available keys: {list(results.keys())}")
                return False
                
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    if verify():
        sys.exit(0)
    else:
        sys.exit(1)

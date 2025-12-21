
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
from Alpha.backtest.backtest import generate_full_system_charts

def verify():
    print("Verifying Phase 6: Reporting & Visualization...")
    
    test_dir = tempfile.mkdtemp()
    try:
        data_dir = os.path.join(test_dir, "data")
        output_dir = Path(os.path.join(test_dir, "results"))
        os.makedirs(data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
            # Alternating approved/blocked
            mock_guard_model.predict.side_effect = [np.array([0.8]), np.array([0.1])] * 10
            mock_guard_load.return_value = (mock_guard_model, {'threshold': 0.5})
            
            bt = FullSystemBacktest(
                "dummy_alpha", "dummy_risk", "dummy_guard", "dummy_meta",
                data_dir=data_dir
            )
            
            # Mock peek-ahead
            bt.env._simulate_trade_outcome_with_timing = unittest.mock.Mock()
            bt.env._simulate_trade_outcome_with_timing.return_value = {
                'pnl': -0.01, 'reason': 'sl', 'exit_step': 210
            }
            
            print("Running backtest...")
            bt.env.max_steps = 220
            metrics_tracker = bt.run_backtest(episodes=1)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"Generating charts in {output_dir}...")
            generate_full_system_charts(metrics_tracker, {}, 3, output_dir, timestamp)
            
            # 3. Verify files exist
            expected_files = [
                f"full_system_analysis_stage3_{timestamp}.png",
                f"tradeguard_performance_stage3_{timestamp}.png"
            ]
            
            all_exist = True
            for f in expected_files:
                if not (output_dir / f).exists():
                    print(f"ERROR: Missing chart file {f}")
                    all_exist = False
                else:
                    print(f"Verified: {f} exists.")
            
            if all_exist:
                print("Phase 6 Verification SUCCESSFUL.")
                return True
            else:
                return False
                
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    if verify():
        sys.exit(0)
    else:
        sys.exit(1)

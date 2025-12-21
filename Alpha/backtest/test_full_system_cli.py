
import unittest
import os
import sys
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import unittest.mock

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.backtest_full_system import run_full_system_backtest

class TestFullSystemCLI(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_dir = self.test_dir / "data"
        self.output_dir = self.test_dir / "results"
        self.data_dir.mkdir()
        self.output_dir.mkdir()
        
        # Create dummy data
        assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="5min")
        for asset in assets:
            # Trend for EURUSD to trigger TP/SL
            trend = np.linspace(0, 0.1, 1000) if asset == 'EURUSD' else np.zeros(1000)
            base_price = 1.1 + trend
            df = pd.DataFrame({
                'open': base_price,
                'high': base_price + 0.001,
                'low': base_price - 0.001,
                'close': base_price,
                'volume': np.ones(1000) * 1000
            }, index=dates)
            df.to_parquet(self.data_dir / f"{asset}_5m.parquet")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_run_full_system_backtest_cli(self):
        """Test the end-to-end execution via run_full_system_backtest"""
        # Mock models and their loading
        with (unittest.mock.patch('Alpha.backtest.backtest_full_system.PPO.load') as mock_ppo_load,
              unittest.mock.patch('Alpha.backtest.backtest_full_system.load_tradeguard_model') as mock_guard_load,
              unittest.mock.patch('Alpha.backtest.backtest_full_system.generate_full_system_charts') as mock_charts):
            
            # Setup mock behavior
            mock_alpha = unittest.mock.Mock()
            mock_alpha.predict.return_value = (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), None)
            
            mock_risk = unittest.mock.Mock()
            mock_risk.predict.return_value = (np.array([1.0, 1.0, 1.0]), None)
            
            mock_ppo_load.side_effect = [mock_alpha, mock_risk]
            
            mock_guard_model = unittest.mock.Mock()
            mock_guard_model.predict.return_value = np.array([0.8])
            mock_guard_load.return_value = (mock_guard_model, {'threshold': 0.5})
            
            # Create arguments
            args = argparse.Namespace(
                alpha_model="dummy_alpha",
                risk_model="dummy_risk",
                guard_model="dummy_guard",
                guard_meta="dummy_meta",
                data_dir=str(self.data_dir),
                output_dir=self.output_dir,
                episodes=1,
                initial_equity=10.0
            )
            
            # Run
            run_full_system_backtest(args)
            
            # Verify output files
            metrics_files = list(self.output_dir.glob("metrics_full_system_*.json"))
            self.assertEqual(len(metrics_files), 1)
            
            trades_files = list(self.output_dir.glob("trades_full_system_*.csv"))
            # If no trades happened (all flat signals), this might be 0, but Alpha signals 1.0
            self.assertGreater(len(trades_files), 0)
            
            # Verify charts function was called
            mock_charts.assert_called()

if __name__ == "__main__":
    unittest.main()

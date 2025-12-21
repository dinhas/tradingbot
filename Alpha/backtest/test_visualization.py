
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import tempfile
import shutil
from datetime import datetime, timedelta

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.backtest import FullSystemMetrics, generate_full_system_charts

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_full_system_charts_generation(self):
        """Test that generate_full_system_charts creates the expected files"""
        metrics = FullSystemMetrics()
        t0 = datetime.now()
        
        # Add dummy equity data
        for i in range(100):
            t = t0 + timedelta(minutes=5*i)
            metrics.add_equity_point(t, 10.0 + i*0.1)
            metrics.add_shadow_equity_point(t, 10.0 + i*0.05)
            
        # Add dummy trade data
        metrics.add_trade({
            'timestamp': t0 + timedelta(minutes=30),
            'asset': 'EURUSD',
            'net_pnl': 0.5,
            'direction': 1,
            'prob': 0.8
        })
        
        # Add dummy blocked trades
        metrics.add_blocked_trade({
            'timestamp': t0 + timedelta(minutes=60),
            'asset': 'GBPUSD',
            'theoretical_pnl': -0.2,
            'prob': 0.3,
            'outcome': 'sl'
        })
        
        # Generate charts
        generate_full_system_charts(metrics, {}, 3, self.test_dir, self.timestamp)
        
        # Verify files exist
        expected_files = [
            f"full_system_analysis_stage3_{self.timestamp}.png",
            f"tradeguard_performance_stage3_{self.timestamp}.png"
        ]
        
        for f in expected_files:
            file_path = self.test_dir / f
            self.assertTrue(file_path.exists(), f"Expected chart file {f} not found.")

if __name__ == "__main__":
    unittest.main()

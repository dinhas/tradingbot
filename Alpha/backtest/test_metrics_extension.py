
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.backtest import BacktestMetrics, FullSystemMetrics

class TestMetricsExtension(unittest.TestCase):
    def test_tradeguard_metrics_calculation(self):
        metrics = FullSystemMetrics()
        
        # Add some successful trades
        metrics.add_trade({'net_pnl': 0.01, 'asset': 'EURUSD'})
        metrics.add_trade({'net_pnl': 0.02, 'asset': 'GBPUSD'})
        
        # Add some blocked trades
        # Good block: avoided a loss of 0.015
        metrics.add_blocked_trade({'theoretical_pnl': -0.015, 'asset': 'EURUSD'})
        # Bad block: missed a profit of 0.005
        metrics.add_blocked_trade({'theoretical_pnl': 0.005, 'asset': 'USDJPY'})
        
        tg_metrics = metrics.calculate_tradeguard_metrics()
        
        # Approval Rate: 2 approved / 4 total = 0.5
        self.assertEqual(tg_metrics['approval_rate'], 0.5)
        
        # Block Accuracy: 1 good block / 2 total blocked = 0.5
        self.assertEqual(tg_metrics['block_accuracy'], 0.5)
        
        # Net Value-Add: 0.015 (avoided) - 0.005 (missed) = 0.010
        self.assertAlmostEqual(tg_metrics['net_value_add_pct'], 0.010)
        self.assertEqual(tg_metrics['total_blocked'], 2)

    def test_calculate_metrics_integration(self):
        from datetime import datetime, timedelta
        metrics = FullSystemMetrics()
        t0 = datetime.now()
        t1 = t0 + timedelta(days=1)
        
        metrics.add_equity_point(t0, 10.0)
        metrics.add_shadow_equity_point(t0, 10.0)
        
        # Win one trade
        metrics.add_trade({'net_pnl': 0.1, 'asset': 'EURUSD', 'pnl': 0.1, 'fees': 0})
        metrics.add_equity_point(t1, 10.1)
        
        # Block one trade that would have been a loss of 0.2
        metrics.add_blocked_trade({'theoretical_pnl': -0.02, 'asset': 'GBPUSD'})
        # Shadow portfolio would have executed this loss
        metrics.add_shadow_equity_point(t1, 9.9) # 10.1 - 0.2 (approx)
        
        results = metrics.calculate_metrics()
        
        self.assertIn('tradeguard', results)
        self.assertIn('baseline_return', results)
        self.assertIn('net_value_add_vs_baseline', results)
        
        # Real return: (10.1 - 10) / 10 = 0.01
        # Shadow return: (9.9 - 10) / 10 = -0.01
        self.assertAlmostEqual(results['total_return'], 0.01)
        self.assertAlmostEqual(results['baseline_return'], -0.01)
        self.assertAlmostEqual(results['net_value_add_vs_baseline'], 0.02)

if __name__ == "__main__":
    unittest.main()

import unittest
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

class TestFeatureEngine(unittest.TestCase):
    def setUp(self):
        # We'll create the FeatureEngine class soon
        try:
            from generate_dataset import FeatureEngine
            self.FeatureEngine = FeatureEngine
        except ImportError:
            self.FeatureEngine = None

    def test_alpha_confidence_features(self):
        """Test calculation of Alpha Model Confidence features (1-10)."""
        if self.FeatureEngine is None:
            self.skipTest("FeatureEngine not implemented yet")
            
        fe = self.FeatureEngine()
        
        # Mock inputs
        market_row = pd.Series({
            'EURUSD_close': 1.1000,
            'GBPUSD_close': 1.3000
        })
        
        portfolio_state = {
            'equity': 10000,
            'peak_equity': 10500,
            'total_exposure': 4000,
            'open_positions_count': 2,
            'recent_trades': [
                {'pnl': 100}, {'pnl': -50}, {'pnl': 200}, {'pnl': 150}, {'pnl': -100},
                {'pnl': 50}, {'pnl': 80}, {'pnl': -20}, {'pnl': 120}, {'pnl': 300}
            ],
            'asset_action_raw': 0.75,
            'asset_recent_actions': [0.5, 0.6, 0.7, 0.8, 0.75],
            'asset_signal_persistence': 3,
            'asset_signal_reversal': 0
        }
        
        features = fe.calculate_alpha_confidence(market_row, portfolio_state)
        
        # We expect 10 features
        self.assertEqual(len(features), 10)
        
        # Feature 1: alpha_action_raw
        self.assertEqual(features[0], 0.75)
        # Feature 2: alpha_action_abs
        self.assertEqual(features[1], 0.75)
        # Feature 3: alpha_action_std
        self.assertAlmostEqual(features[2], np.std([0.5, 0.6, 0.7, 0.8, 0.75]), places=5)
        # Feature 4: alpha_signal_persistence
        self.assertEqual(features[3], 3)
        # Feature 5: alpha_signal_reversal
        self.assertEqual(features[4], 0)
        # Feature 6: alpha_portfolio_drawdown
        self.assertAlmostEqual(features[5], 1 - (10000 / 10500), places=5)
        # Feature 7: alpha_open_positions
        self.assertEqual(features[6], 2)
        # Feature 8: alpha_margin_usage
        self.assertEqual(features[7], 4000 / 10000)
        # Feature 9: alpha_recent_win_rate
        self.assertEqual(features[8], 0.7) # 7 wins out of 10
        # Feature 10: alpha_recent_pnl
        self.assertEqual(features[9], 830) # Sum of recent_trades pnl

if __name__ == '__main__':
    unittest.main()


import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.tradeguard_feature_builder import TradeGuardFeatureBuilder

class TestTradeGuardFeatureBuilder(unittest.TestCase):
    def setUp(self):
        # Create dummy data for testing
        dates = pd.date_range(start="2023-01-01", periods=200, freq="5min")
        self.assets = ['EURUSD', 'GBPUSD']
        self.df_dict = {}
        for asset in self.assets:
            df = pd.DataFrame({
                'open': np.random.rand(200) + 1.1,
                'high': np.random.rand(200) + 1.2,
                'low': np.random.rand(200) + 1.0,
                'close': np.random.rand(200) + 1.1,
                'volume': np.random.rand(200) * 1000
            }, index=dates)
            self.df_dict[asset] = df
            
        self.builder = TradeGuardFeatureBuilder(self.df_dict)
        
    def test_precompute_market_features(self):
        """Test that market features (B, C, D, F) are precomputed correctly"""
        # 40 features total (B=10, C=10, D=10, F=10)
        self.assertIn('EURUSD', self.builder.precomputed_market_features)
        self.assertEqual(self.builder.precomputed_market_features['EURUSD'].shape[1], 40)
        self.assertEqual(len(self.builder.precomputed_market_features['EURUSD']), 200)

    def test_feature_vector_schema(self):
        """Test that the generated feature vector has the correct size (60 features)"""
        portfolio_state = {
            'equity': 10000.0,
            'peak_equity': 10000.0,
            'total_exposure': 1000.0,
            'open_positions_count': 1,
            'recent_trades': [{'pnl': 0.01}, {'pnl': -0.005}],
            'asset_action_raw': 0.5,
            'asset_recent_actions': [0.5, 0.4, 0.6, 0.5, 0.5],
            'asset_signal_persistence': 1.0,
            'asset_signal_reversal': 0.0,
            'position_value': 500.0
        }
        
        trade_info = {
            'entry_price': 1.1000,
            'sl': 1.0900,
            'tp': 1.1200,
            'direction': 1
        }
        
        # Test for EURUSD at step 150
        features = self.builder.build_features('EURUSD', 150, trade_info, portfolio_state)
        
        # Schema verification:
        # Group A (Alpha Confidence): 10 features
        # Group B+C (Market Context): 30 features (slices of precomputed)
        # Group E (Execution): 10 features
        # Group D+F (Market Context): 10 features (remaining precomputed)
        # TOTAL: 60 features
        
        self.assertEqual(len(features), 60)
        self.assertIsInstance(features, list)
        self.assertTrue(all(isinstance(f, (int, float, np.float32, np.float64)) for f in features))

    def test_feature_consistency(self):
        """Test that precomputed features match for the same step"""
        portfolio_state = {
            'equity': 10000.0,
            'peak_equity': 10000.0,
            'total_exposure': 0,
            'open_positions_count': 0,
            'recent_trades': [],
            'asset_action_raw': 0,
            'asset_recent_actions': [0]*5,
            'asset_signal_persistence': 1.0,
            'asset_signal_reversal': 0.0,
            'position_value': 0
        }
        trade_info = {'entry_price': 1.1, 'sl': 1.09, 'tp': 1.12, 'direction': 1}
        
        f1 = self.builder.build_features('EURUSD', 100, trade_info, portfolio_state)
        f2 = self.builder.build_features('EURUSD', 100, trade_info, portfolio_state)
        
        self.assertEqual(f1, f2)

if __name__ == "__main__":
    unittest.main()

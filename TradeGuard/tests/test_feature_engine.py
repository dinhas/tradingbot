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
        except ImportError as e:
            print(f"ImportError: {e}")
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

    def test_risk_output_features(self):
        """Test calculation of Risk Model Output features."""
        if self.FeatureEngine is None:
            self.skipTest("FeatureEngine not implemented yet")
            
        fe = self.FeatureEngine()
        
        # Mock inputs
        risk_params = {
            'sl_mult': 1.5,
            'tp_mult': 2.5,
            'risk_factor': 0.8
        }
        
        features = fe.calculate_risk_output(risk_params)
        
        # We expect 3 features
        self.assertEqual(len(features), 3)
        self.assertEqual(features[0], 1.5)
        self.assertEqual(features[1], 2.5)
        self.assertEqual(features[2], 0.8)

    def test_news_proxies_features(self):
        """Test calculation of Synthetic News Proxies features (11-20)."""
        if self.FeatureEngine is None:
            self.skipTest("FeatureEngine not implemented yet")
            
        fe = self.FeatureEngine()
        
        # Create a mock historical DataFrame (200 bars)
        # Constant data for 199 bars, then an anomaly
        opens = [1.1000] * 200
        highs = [1.1010] * 199 + [1.1050]
        lows = [1.0990] * 199 + [1.0950]
        closes = [1.1005] * 199 + [1.1000]
        volumes = [1000.0] * 199 + [5000.0]
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        features = fe.calculate_news_proxies(df)
        
        # We expect 10 features
        self.assertEqual(len(features), 10)
        
        # Feature 11: volume_ratio (Current volume / 50-period average)
        # Avg of last 50: (49*1000 + 5000) / 50 = 1080
        # Ratio: 5000 / 1080 = 4.6296
        self.assertAlmostEqual(features[0], 4.6296, places=4)
        
        # Feature 12: volume_zscore
        # Mean = 1080, Std of [1000]*49 + [5000]
        vol_slice = pd.Series([1000.0]*49 + [5000.0])
        expected_z = (5000.0 - vol_slice.mean()) / vol_slice.std()
        self.assertAlmostEqual(features[1], expected_z, places=4)
        
        # Feature 13: range_ratio (Range / ATR)
        # Current Range = 1.1050 - 1.0950 = 0.0100
        # This one is tricky because ATR depends on previous bars.
        # But we can assert it's greater than 1.0 significantly.
        self.assertGreater(features[2], 1.0)
        
        # Feature 15: body_to_range
        # Body = abs(1.1000 - 1.1000) = 0
        # Range = 0.0100
        # Ratio = 0
        self.assertEqual(features[4], 0.0)
        
        # Feature 16: wick_ratio
        # Upper wick = 1.1050 - 1.1000 = 0.0050
        # Lower wick = 1.1000 - 1.0950 = 0.0050
        # Body = 0.0001 (min for division)
        # Ratio = (0.0050 + 0.0050) / 0.0001 = 100
        self.assertGreater(features[5], 10.0)

    def test_market_regime_features(self):
        """Test calculation of Market Regime features (21-30)."""
        if self.FeatureEngine is None:
            self.skipTest("FeatureEngine not implemented yet")
            
        fe = self.FeatureEngine()
        
        # Create a mock historical DataFrame (300 bars for long-term indicators)
        # Generate a trending market then a chop
        np.random.seed(42)
        n = 300
        
        # Trending part
        closes = np.linspace(1.1000, 1.2000, 200).tolist()
        # Choppy part
        closes += (np.random.rand(100) * 0.01 + 1.2000).tolist()
        
        opens = [c - 0.0005 for c in closes]
        highs = [c + 0.0010 for c in closes]
        lows = [c - 0.0010 for c in closes]
        volumes = [1000.0] * 300
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        features = fe.calculate_market_regime(df)
        
        # We expect 10 features
        self.assertEqual(len(features), 10)
        
        # Basic range checks for indicators
        
        # Feature 21: ADX (0-100)
        self.assertTrue(0 <= features[0] <= 100)
        
        # Feature 26: Hurst (0-1) - roughly
        # Since we added random chop at the end, hurst should be closer to 0.5 or <0.5
        self.assertTrue(0 <= features[5] <= 1.0)

    def test_session_edge_features(self):
        """Test calculation of Session Edge features (31-40)."""
        if self.FeatureEngine is None:
            self.skipTest("FeatureEngine not implemented yet")
            
        fe = self.FeatureEngine()
        
        # Create a mock timestamp (e.g., London Open - 8:00 UTC)
        timestamp = pd.Timestamp("2024-06-19 08:00:00")
        
        features = fe.calculate_session_edge(timestamp)
        
        # We expect 10 features
        self.assertEqual(len(features), 10)
        
        # Feature 31: hour_sin (8 am -> sin(2*pi*8/24))
        expected_sin = np.sin(2 * np.pi * 8 / 24)
        self.assertAlmostEqual(features[0], expected_sin, places=4)
        
        # Feature 35: is_london_open (Assume London is 7-16 UTC)
        # At 8:00, London is open -> 1.0
        self.assertEqual(features[4], 1.0)
        
        # Feature 36: is_ny_open (Assume NY is 12-21 UTC)
        # At 8:00, NY is closed -> 0.0
        self.assertEqual(features[5], 0.0)

    def test_execution_stats_features(self):
        """Test calculation of Execution Statistics features (41-50)."""
        if self.FeatureEngine is None:
            self.skipTest("FeatureEngine not implemented yet")
            
        fe = self.FeatureEngine()
        
        # Mock inputs
        # 20 bars of data
        closes = [1.1000] * 19 + [1.1050]
        highs = [c + 0.0010 for c in closes]
        lows = [c - 0.0010 for c in closes]
        opens = [c - 0.0005 for c in closes]
        df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes})
        
        trade_info = {
            'entry_price': 1.1050,
            'sl': 1.1000,
            'tp': 1.1150,
            'direction': 1 # Long
        }
        
        portfolio_state = {
            'equity': 10000,
            'peak_equity': 10500,
            'position_value': 1000
        }
        
        features = fe.calculate_execution_stats(df, trade_info, portfolio_state)
        
        self.assertEqual(len(features), 10)
        
        # Feature 41: entry_atr_distance
        self.assertGreater(features[0], 0)
        
        # Feature 44: risk_reward_ratio (100 pips / 50 pips = 2.0)
        self.assertAlmostEqual(features[3], 2.0, places=1)
        
        # Feature 45: position_size_pct (1000 / 10000 = 0.1)
        self.assertEqual(features[4], 0.1)

    def test_price_action_context_features(self):
        """Test calculation of Price Action Context features (51-60)."""
        if self.FeatureEngine is None:
            self.skipTest("FeatureEngine not implemented yet")
            
        fe = self.FeatureEngine()
        
        # 30 bars of data
        closes = [1.1000 + i*0.0001 for i in range(30)] # Uptrend
        highs = [c + 0.0005 for c in closes]
        lows = [c - 0.0005 for c in closes]
        opens = [c - 0.0002 for c in closes]
        volumes = [1000.0] * 30
        df = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes})
        
        features = fe.calculate_price_action_context(df)
        
        self.assertEqual(len(features), 10)
        
        # Feature 51: candle_direction (close > open)
        self.assertEqual(features[0], 1.0)
        
        # Feature 55: consecutive_direction
        self.assertGreaterEqual(features[4], 5)

if __name__ == '__main__':
    unittest.main()

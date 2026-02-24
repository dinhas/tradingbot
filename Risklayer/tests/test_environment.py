import unittest
import numpy as np
import pandas as pd
from Risklayer.execution_engine import ExecutionEngine
from Risklayer.peak_labeling import StructuralLabeler
from Risklayer.config import config
from Risklayer.trading_env import TradingEnv

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.execution = ExecutionEngine()
        self.labeler = StructuralLabeler()

        # Create Dummy Data
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='5min')
        self.dummy_df = pd.DataFrame({
            'EURUSD_open': np.linspace(1.1000, 1.1100, 1000),
            'EURUSD_high': np.linspace(1.1005, 1.1105, 1000),
            'EURUSD_low': np.linspace(1.0995, 1.1095, 1000),
            'EURUSD_close': np.linspace(1.1000, 1.1100, 1000),
            'EURUSD_volume': 100,
            'EURUSD_atr': 0.0010,
            'EURUSD_vol_percentile': 0.5,
            'meta_score': 0.6,
            'quality_score': 0.4
        }, index=dates)

        # Add labels
        self.dummy_df = self.labeler.label_data(self.dummy_df, 'EURUSD')

        # Add dummy features for other assets needed by FeatureEngine
        for asset in ['GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']:
            self.dummy_df[f"{asset}_close"] = 1.0
            self.dummy_df[f"{asset}_high"] = 1.05
            self.dummy_df[f"{asset}_low"] = 0.95
            self.dummy_df[f"{asset}_open"] = 1.0
            self.dummy_df[f"{asset}_volume"] = 100
            self.dummy_df[f"{asset}_atr"] = 0.0010

        # Add features
        from Risklayer.feature_engineering import FeatureEngine
        fe = FeatureEngine()
        feature_df = fe.calculate_features(self.dummy_df)
        self.dummy_df = pd.concat([self.dummy_df, feature_df], axis=1)

    def test_bid_ask_entry(self):
        mid = 1.1000
        atr = 0.0010
        asset = 'EURUSD'
        spread = config.SPREADS[asset]

        # Long entry
        entry_long = self.execution.get_entry_price(asset, mid, 'long', atr)
        self.assertGreaterEqual(entry_long, mid + spread/2)

        # Short entry
        entry_short = self.execution.get_entry_price(asset, mid, 'short', atr)
        self.assertLessEqual(entry_short, mid - spread/2)

    def test_sl_tp_triggering(self):
        asset = 'EURUSD'
        open_p = 1.1000
        sl = 1.0990
        tp = 1.1020

        # Hit SL
        exit_p, reason = self.execution.check_exit(asset, 'long', open_p, sl, tp, 1.1010, 1.0980, 1.0990, 0.0010)
        self.assertEqual(reason, 'sl')
        self.assertEqual(exit_p, sl)

        # Hit TP
        exit_p, reason = self.execution.check_exit(asset, 'long', open_p, sl, tp, 1.1025, 1.1005, 1.1020, 0.0010)
        self.assertEqual(reason, 'tp')
        self.assertEqual(exit_p, tp)

    def test_position_sizing(self):
        equity = 10000.0
        risk_pct = 0.01 # 1% = $100
        sl_dist = 0.0010 # 10 pips
        contract_size = 100000

        # Expected volume: 100 / (0.0010 * 100000) = 1 lot
        expected_vol = (equity * risk_pct) / (sl_dist * contract_size)
        self.assertEqual(expected_vol, 1.0)

    def test_drawdown_termination(self):
        # Create environment with low equity
        env = TradingEnv(self.dummy_df, asset='EURUSD')
        env.equity = 100 # 1% of initial

        # Try to step
        action = np.array([0, 0, 0]) # Mid values
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(terminated)
        self.assertLess(reward, 0) # Termination penalty

    def test_peak_labeling_consistency(self):
        # Check if peak_dist is always >= 0
        self.assertTrue((self.dummy_df['EURUSD_peak_dist'] >= 0).all())
        self.assertTrue((self.dummy_df['EURUSD_valley_dist'] >= 0).all())

if __name__ == '__main__':
    unittest.main()

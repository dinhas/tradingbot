import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import json
import tempfile
import shutil
import lightgbm as lgb

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.backtest_full_system import FullSystemBacktest

class MockBooster:
    def __init__(self, prob=0.6):
        self.prob = prob
    def predict(self, X):
        return np.array([self.prob])
    def save_model(self, path):
        with open(path, 'w') as f: f.write("mock")

class TestFullSystemLoop(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "guard_model.txt")
        self.metadata_path = os.path.join(self.test_dir, "model_metadata.json")
        
        # Create dummy data files for TradingEnv
        self.data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.data_dir)
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
            df.to_parquet(os.path.join(self.data_dir, f"{asset}_5m.parquet"))

        # Dummy models
        # Note: We need actual .zip files for SB3 load to work if we instantiate FullSystemBacktest normally,
        # but for testing the loop logic we might need to mock PPO.load too or just bypass it.
        # Let's mock the loading part of FullSystemBacktest.
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tradeguard_blocking_logic(self):
        """Test that trades are blocked when TradeGuard probability < threshold"""
        # We'll mock the internal components of FullSystemBacktest
        with (unittest.mock.patch('Alpha.backtest.backtest_full_system.PPO.load') as mock_ppo_load,
              unittest.mock.patch('Alpha.backtest.backtest_full_system.load_tradeguard_model') as mock_guard_load):
            
            mock_ppo_load.return_value = unittest.mock.Mock()
            
            # Guard model with 0.4 probability, threshold 0.5 -> Should block
            mock_guard_model = unittest.mock.Mock()
            mock_guard_model.predict.return_value = np.array([0.4])
            mock_guard_load.return_value = (mock_guard_model, {'threshold': 0.5})
            
            bt = FullSystemBacktest(
                "dummy_alpha", "dummy_risk", "dummy_guard", "dummy_meta",
                data_dir=self.data_dir
            )
            bt.env.reset()
            
            # Simulate an action from Alpha
            # 'direction' != 0
            act = {'direction': 1, 'size': 0.1, 'sl_mult': 1.5, 'tp_mult': 3.0}
            
            # Mock build_features
            bt.feature_builder = unittest.mock.Mock()
            bt.feature_builder.build_features.return_value = [0.0]*60
            
            # Evaluate
            is_approved, prob = bt.evaluate_tradeguard('EURUSD', act)
            
            self.assertFalse(is_approved)
            self.assertEqual(prob, 0.4)
            
            # Change prob to 0.6 -> Should approve
            mock_guard_model.predict.return_value = np.array([0.6])
            is_approved, prob = bt.evaluate_tradeguard('EURUSD', act)
            self.assertTrue(is_approved)
            self.assertEqual(prob, 0.6)

    def test_run_backtest_integration(self):
        """Test that run_backtest integration correctly captures blocked trades"""
        with (unittest.mock.patch('Alpha.backtest.backtest_full_system.PPO.load') as mock_ppo_load,
              unittest.mock.patch('Alpha.backtest.backtest_full_system.load_tradeguard_model') as mock_guard_load):
            
            # Mock Alpha model to always signal direction 1 for EURUSD
            mock_alpha = unittest.mock.Mock()
            # Alpha returns 5 outputs for Stage 1. Let's say [1.0, 0, 0, 0, 0]
            mock_alpha.predict.return_value = (np.array([1.0, 0.0, 0.0, 0.0, 0.0]), None)
            
            # Mock Risk model to return valid action
            mock_risk = unittest.mock.Mock()
            mock_risk.predict.return_value = (np.array([1.0, 1.0, 1.0]), None) # sl, tp, risk (0.8+0.2=1.0)
            
            mock_ppo_load.side_effect = [mock_alpha, mock_risk]
            
            # Mock TradeGuard to always block (prob 0.1, threshold 0.5)
            mock_guard_model = unittest.mock.Mock()
            mock_guard_model.predict.return_value = np.array([0.1])
            mock_guard_load.return_value = (mock_guard_model, {'threshold': 0.5})
            
            bt = FullSystemBacktest(
                "dummy_alpha", "dummy_risk", "dummy_guard", "dummy_meta",
                data_dir=self.data_dir
            )
            
            # Run for a few steps
            bt.env.max_steps = 205 # Env starts at 200 after warmup
            bt.run_backtest(episodes=1)
            
            # Check if blocked_trades has entries
            self.assertGreater(len(bt.blocked_trades), 0)
            self.assertEqual(bt.blocked_trades[0]['asset'], 'EURUSD')
            self.assertEqual(bt.blocked_trades[0]['prob'], 0.1)

if __name__ == "__main__":
    unittest.main()

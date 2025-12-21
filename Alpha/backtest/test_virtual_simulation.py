
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os
import unittest.mock

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.backtest_full_system import FullSystemBacktest

class TestVirtualSimulation(unittest.TestCase):
    def setUp(self):
        # We need a minimal environment setup to test _simulate_blocked_trade
        with (unittest.mock.patch('Alpha.backtest.backtest_full_system.PPO.load') as mock_ppo_load,
              unittest.mock.patch('Alpha.backtest.backtest_full_system.load_tradeguard_model') as mock_guard_load,
              unittest.mock.patch('Alpha.backtest.backtest_full_system.TradeGuardFeatureBuilder') as mock_fb):
            
            mock_guard_load.return_value = (unittest.mock.Mock(), {'threshold': 0.5})
            
            # Create a mock for the TradingEnv data that CombinedBacktest (parent) needs
            self.mock_env = unittest.mock.Mock()
            self.mock_env.assets = ['EURUSD']
            self.mock_env.data = {'EURUSD': pd.DataFrame()}
            self.mock_env.action_dim = 5
            self.mock_env.positions = {'EURUSD': None}
            
            # Instead of instantiating, let's mock __init__ or just create an object and set attributes
            # Actually, FullSystemBacktest inherits from CombinedBacktest which has logic we might need.
            # Let's try a more targeted mock of _simulate_blocked_trade
            self.bt = FullSystemBacktest.__new__(FullSystemBacktest)
            self.bt.env = self.mock_env
            self.bt.blocked_trades = []
            self.bt.guard_threshold = 0.5

    def test_simulate_blocked_trade_calculation(self):
        """Test that _simulate_blocked_trade correctly calculates PnL using future data"""
        asset = 'EURUSD'
        direction = 1 # Long
        act = {'sl_mult': 1.5, 'tp_mult': 3.0}
        prob = 0.1
        
        # Current state
        current_step = 100
        self.bt.env.current_step = current_step
        
        # Mock price data
        # Step 100: Price 1.1000, ATR 0.0010
        # SL = 1.1000 - 1.5 * 0.0010 = 1.0985
        # TP = 1.1000 + 3.0 * 0.0010 = 1.1030
        
        # We need _get_current_prices and _get_current_atrs to be available on env
        self.mock_env._get_current_prices.return_value = {asset: 1.1000}
        self.mock_env._get_current_atrs.return_value = {asset: 0.0010}
        self.bt.env.MIN_ATR_MULTIPLIER = 0.0001
        
        # Future data: let's say it hits TP at step 105
        # We need to mock how _simulate_blocked_trade looks ahead.
        # It should probably call a method on the environment that performs the peek-ahead.
        # TradingEnv already has _simulate_trade_outcome_with_timing (used in generate_dataset.py)
        
        # Let's mock the env._simulate_trade_outcome_with_timing
        self.mock_env._simulate_trade_outcome_with_timing.return_value = {
            'pnl': 0.0030 / 1.1000, # Profit
            'exit_step': 105,
            'reason': 'tp'
        }
        
        # Call the method
        self.bt._simulate_blocked_trade(asset, direction, act, prob)
        
        # Verify results
        self.assertEqual(len(self.bt.blocked_trades), 1)
        trade = self.bt.blocked_trades[0]
        self.assertAlmostEqual(trade['theoretical_pnl'], 0.0030 / 1.1000)
        self.assertEqual(trade['outcome'], 'tp')

if __name__ == "__main__":
    unittest.main()

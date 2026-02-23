import unittest
import pandas as pd
import numpy as np
from Risklayer.config import config
from Risklayer.execution_engine import ExecutionEngine
from Risklayer.peak_labeling import PeakLabeler
from Risklayer.trading_env import RiskTradingEnv
from Risklayer.feature_engineering import FeatureEngine

class TestRiskEnvironment(unittest.TestCase):
    def setUp(self):
        # Create dummy data for testing (need more for robustness)
        n = 2000
        data = {
            'open':  np.linspace(1.1000, 1.1100, n),
            'high':  np.linspace(1.1010, 1.1110, n),
            'low':   np.linspace(1.0990, 1.1090, n),
            'close': np.linspace(1.1005, 1.1105, n),
            'volume': np.random.randint(100, 200, n)
        }
        # Add a clear peak at index 500
        data['high'][500] = 1.1500
        # Reversal at index 510
        data['low'][510] = 1.1200 # 0.03 drop, which is plenty for 2*ATR

        self.raw_df = pd.DataFrame(data)
        self.raw_df.index = pd.date_range(start='2023-01-01', periods=n, freq='5min')

        # Preprocess
        fe = FeatureEngine()
        self.df = fe.compute_features(self.raw_df)
        labeler = PeakLabeler(reversal_mult=2.0)
        self.df = labeler.label_data(self.df)

    def test_peak_labeling(self):
        # Peak at 500 should be visible from earlier bars
        self.assertGreater(self.df.iloc[450]['peak_distance'], 0)

    def test_execution_bid_ask(self):
        config.SPREAD_PIPS = 2.0
        config.SLIPPAGE_STD = 0.0
        engine = ExecutionEngine()

        # Long trade
        result = engine.execute_trade(self.df, 100, 1, 1.0500, 1.2000, 100000)
        self.assertAlmostEqual(result.entry_price, self.df.iloc[100]['open'] + 0.0001, places=6)

    def test_position_sizing(self):
        env = RiskTradingEnv(self.df)
        env.equity = 100000

        # Should initialize without error now
        self.assertGreater(env.equity, 0)

    def test_drawdown_termination(self):
        config.INITIAL_EQUITY = 100000
        config.DRAWDOWN_LIMIT_PCT = 0.98
        env = RiskTradingEnv(self.df)
        env.equity = 1000 # Below 2%

        obs, reward, terminated, truncated, info = env.step(np.array([0, 0, 0]))
        self.assertTrue(terminated)

if __name__ == '__main__':
    unittest.main()

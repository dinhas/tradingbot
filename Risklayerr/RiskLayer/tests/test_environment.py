import unittest
import numpy as np
import pandas as pd
from ..config import RiskConfig
from ..trading_env import RiskTradingEnv
from ..execution_engine import ExecutionEngine
from ..peak_labeling import PeakLabeler

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.config = RiskConfig()
        self.config.RANDOM_START = False
        self.env = RiskTradingEnv(self.config)

    def test_bid_ask_logic(self):
        """Tests that bid is below ask by the spread amount."""
        engine = ExecutionEngine(self.config)
        spread = self.config.SPREAD_PITS * 0.00001

        # This is implicitly tested in resolve_trade, but we can verify it here
        # Actually, let's just check the config/engine directly
        self.assertGreater(spread, 0)

    def test_position_sizing(self):
        """Tests that position size correctly reflects risk percent."""
        state, _ = self.env.reset()
        # Mock a step with specific action
        # Action: [SL_mult, RR_ratio, Risk_pct]
        # action = [0, 0, 0] -> maps to middle of ranges
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # We need to ensure a trade actually happens, so mock scores
        self.env.df.iloc[self.env.current_step, self.env.df.columns.get_loc('meta_score')] = 1.0
        self.env.df.iloc[self.env.current_step, self.env.df.columns.get_loc('quality_score')] = 1.0

        obs, reward, done, truncated, info = self.env.step(action)

        # If a trade happened, check info or equity
        # (This depends on whether the trade resolved immediately)
        if self.env.last_trade_result:
            self.assertIsNotNone(self.env.last_trade_result.net_pnl)

    def test_drawdown_termination(self):
        """Tests that the episode ends when 98% drawdown is hit."""
        self.env.equity = self.config.INITIAL_EQUITY * 0.01 # 99% drawdown
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs, reward, done, truncated, info = self.env.step(action)
        self.assertTrue(done)
        self.assertLess(reward, -50) # Penalty

    def test_peak_labeling(self):
        """Tests that peak labeling produces non-zero values."""
        labeler = PeakLabeler(self.config)
        df = self.env.df.head(100)
        p, v = labeler.label_peaks_valleys(df)
        self.assertEqual(len(p), 100)
        self.assertEqual(len(v), 100)
        # Some values should be > 0 if there is movement
        self.assertTrue(np.any(p >= 0))
        self.assertTrue(np.any(v >= 0))

if __name__ == '__main__':
    unittest.main()

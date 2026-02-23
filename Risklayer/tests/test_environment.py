import unittest
import numpy as np
import pandas as pd
from Risklayer.trading_env import TradingEnv
from Risklayer.config import config
from Risklayer.execution_engine import ExecutionEngine

class TestEnvironment(unittest.TestCase):
    def setUp(self):
        # We need some dummy data for the environment to initialize
        self.assets = ["EURUSD"]
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="5min")
        df = pd.DataFrame({
            'open': np.linspace(1.10, 1.11, 1000),
            'high': np.linspace(1.105, 1.115, 1000),
            'low': np.linspace(1.095, 1.105, 1000),
            'close': np.linspace(1.10, 1.11, 1000),
            'volume': np.random.randint(100, 1000, 1000),
            'atr': np.full(1000, 0.0010),
            'vol_percentile': np.full(1000, 0.5),
            'peak_distance': np.full(1000, 0.0020),
            'valley_distance': np.full(1000, 0.0005),
            'meta_score': np.full(1000, 0.6),
            'quality_score': np.full(1000, 0.4),
            'ema9_dist': np.full(1000, 0.0001)
        }, index=dates)

        # Add 30 alpha features columns
        alpha_cols = [
            'ret_1', 'ret_12', 'ret_24', 'bb_pos', 'squeeze',
            'ema9_dist', 'ema21_dist', 'ema_align', 'rsi', 'macd_hist',
            'rsi_slope', 'macd_mom', 'vol_ratio', 'vol_shock', 'vwap_dist',
            'pressure', 'wick_rejection', 'breakout_vel', 'swing_prox', 'hour_sin',
            'hour_cos', 'day_sin', 'day_cos', 'session_asian', 'session_london',
            'session_ny', 'session_overlap', 'ret_240', 'mkt_vol', 'ret_120'
        ]
        for col in alpha_cols:
            if col not in df.columns:
                df[col] = 0.0

        self.data_dict = {"EURUSD": df}
        self.env = TradingEnv(data_dict=self.data_dict)

    def test_bid_ask_logic(self):
        ee = ExecutionEngine()
        mid = 1.1000
        spread = 0.0002
        self.assertEqual(ee.get_entry_price(mid, spread, 'long'), 1.1001)
        self.assertEqual(ee.get_entry_price(mid, spread, 'short'), 1.0999)

    def test_sl_tp_triggering(self):
        ee = ExecutionEngine()
        # Long trade
        entry = 1.1000
        sl = 1.0990
        tp = 1.1020
        atr = 0.0010
        spread = 0.0001

        # Case 1: SL hit
        window_sl = np.array([[1.1000, 1.1005, 1.0980, 1.0990]]) # open, high, low, close
        exit_price, closed, _ = ee.calculate_trade_outcome('long', entry, sl, tp, atr, spread, window_sl)
        self.assertTrue(closed)
        self.assertLessEqual(exit_price, sl)

        # Case 2: TP hit
        window_tp = np.array([[1.1000, 1.1030, 1.0995, 1.1020]])
        exit_price, closed, _ = ee.calculate_trade_outcome('long', entry, sl, tp, atr, spread, window_tp)
        self.assertTrue(closed)
        self.assertEqual(exit_price, tp)

    def test_drawdown_termination(self):
        self.env.reset()
        self.env.equity = config.INITIAL_EQUITY * 0.01 # 99% drawdown
        _, _, done, _, _ = self.env.step(np.array([0, 0, 0]))
        self.assertTrue(done)

    def test_position_sizing(self):
        # Manual check of position sizing in TradingEnv
        # Risk 1% of 100,000 = 1000
        # SL distance = 0.0010 (ATR) * 1.0 (SL_mult) = 0.0010
        # Volume = 1000 / 0.0010 = 1,000,000 units = 10 lots

        self.env.reset()
        # Force a trade by providing action
        action = np.array([0.0, 0.0, 0.0]) # rescales to SL_mult=1.75, RR=2.5, Risk=0.0105
        # Actually my rescale:
        # sl_mult: (-1 to 1) -> (0.5 to 3.0). 0 -> 1.75
        # rr_ratio: 0 -> 2.5
        # risk_pct: 0 -> 0.0105

        # We test the _open_trade directly if needed, but step() is easier.
        obs, reward, done, _, info = self.env.step(action)
        self.assertGreater(self.env.equity, 0)

if __name__ == '__main__':
    unittest.main()

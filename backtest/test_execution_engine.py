import numpy as np
import unittest
from numba import njit

@njit
def resolve_trade_fast(opens, highs, lows, closes, entry_mid, sl_price, tp_price, direction, spread):
    half_spread = spread / 2.0
    if direction == 1: actual_entry = entry_mid + half_spread
    else: actual_entry = entry_mid - half_spread

    sl_dist = abs(actual_entry - sl_price)
    if sl_dist < 1e-10: sl_dist = 1e-10

    for i in range(len(opens)):
        o, h, l = opens[i], highs[i], lows[i]
        if direction == 1:
            bid_o, bid_h, bid_l = o - half_spread, h - half_spread, l - half_spread
            if bid_o <= sl_price: return 0, (bid_o - actual_entry) / sl_dist, i + 1
            if bid_o >= tp_price: return 1, (bid_o - actual_entry) / sl_dist, i + 1
            if bid_l <= sl_price: return 0, (sl_price - actual_entry) / sl_dist, i + 1
            if bid_h >= tp_price: return 1, (tp_price - actual_entry) / sl_dist, i + 1
        else:
            ask_o, ask_h, ask_l = o + half_spread, h + half_spread, l + half_spread
            if ask_o >= sl_price: return 0, (actual_entry - ask_o) / sl_dist, i + 1
            if ask_o <= tp_price: return 1, (actual_entry - ask_o) / sl_dist, i + 1
            if ask_h >= sl_price: return 0, (actual_entry - sl_price) / sl_dist, i + 1
            if ask_l <= tp_price: return 1, (actual_entry - tp_price) / sl_dist, i + 1

    if direction == 1: pnl = (closes[-1] - half_spread) - actual_entry
    else: pnl = actual_entry - (closes[-1] + half_spread)
    return 2, pnl / sl_dist, len(closes)

class TestExecutionEngine(unittest.TestCase):
    def test_long_entry_and_exit_side(self):
        res_code, res_r, res_len = resolve_trade_fast(
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            1.1000, 1.0990, 1.1020, 1, 0.0002
        )
        # Expected PnL: -0.0002. SL Dist: 0.0011.
        self.assertAlmostEqual(res_r, -0.0002 / 0.0011, places=4)

    def test_sl_trigger_bid_ask(self):
        res_code, res_r, res_len = resolve_trade_fast(
            np.array([1.1000], dtype=np.float32),
            np.array([1.1010], dtype=np.float32),
            np.array([1.0996], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            1.1000, 1.0995, 1.1020, 1, 0.0002
        )
        self.assertEqual(res_code, 0)

    def test_gap_fill_realism(self):
        res_code, res_r, res_len = resolve_trade_fast(
            np.array([1.0990], dtype=np.float32),
            np.array([1.0990], dtype=np.float32),
            np.array([1.0980], dtype=np.float32),
            np.array([1.0990], dtype=np.float32),
            1.1000, 1.0995, 1.1020, 1, 0.0002
        )
        self.assertEqual(res_code, 0)
        self.assertAlmostEqual(res_r, -2.0, places=4)

    def test_short_side_correctness(self):
        res_code, res_r, res_len = resolve_trade_fast(
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            1.1000, 1.1010, 1.0980, -1, 0.0002
        )
        self.assertAlmostEqual(res_r, -0.0002 / 0.0011, places=4)

if __name__ == '__main__':
    unittest.main()

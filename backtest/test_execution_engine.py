import unittest

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - test fallback for lightweight environments
    def njit(fn):
        return fn


@njit
def resolve_trade_fast(opens, highs, lows, closes, entry_mid, sl_price, tp_price, direction, spread):
    """Reference execution logic mirrored from RiskLayer/src/risk_ppo_env.py."""
    half_spread = spread / 2.0
    if direction == 1:
        actual_entry = entry_mid + half_spread
    else:
        actual_entry = entry_mid - half_spread

    sl_dist = abs(actual_entry - sl_price)
    if sl_dist < 1e-10:
        sl_dist = 1e-10

    for i in range(len(opens)):
        o, h, l = opens[i], highs[i], lows[i]
        if direction == 1:
            bid_o, bid_h, bid_l = o - half_spread, h - half_spread, l - half_spread
            if bid_o <= sl_price:
                return 0, (bid_o - actual_entry) / sl_dist, i + 1
            if bid_o >= tp_price:
                return 1, (bid_o - actual_entry) / sl_dist, i + 1
            if bid_l <= sl_price:
                return 0, (sl_price - actual_entry) / sl_dist, i + 1
            if bid_h >= tp_price:
                return 1, (tp_price - actual_entry) / sl_dist, i + 1
        else:
            ask_o, ask_h, ask_l = o + half_spread, h + half_spread, l + half_spread
            if ask_o >= sl_price:
                return 0, (actual_entry - ask_o) / sl_dist, i + 1
            if ask_o <= tp_price:
                return 1, (actual_entry - ask_o) / sl_dist, i + 1
            if ask_h >= sl_price:
                return 0, (actual_entry - sl_price) / sl_dist, i + 1
            if ask_l <= tp_price:
                return 1, (actual_entry - tp_price) / sl_dist, i + 1

    if direction == 1:
        pnl = (closes[-1] - half_spread) - actual_entry
    else:
        pnl = actual_entry - (closes[-1] + half_spread)
    return 2, pnl / sl_dist, len(closes)


class TestExecutionEngine(unittest.TestCase):
    def test_long_entry_and_exit_side(self):
        _, res_r, _ = resolve_trade_fast(
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            1.1000,
            1.0990,
            1.1020,
            1,
            0.0002,
        )
        self.assertAlmostEqual(res_r, -0.0002 / 0.0011, places=4)

    def test_short_entry_and_exit_side(self):
        _, res_r, _ = resolve_trade_fast(
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            1.1000,
            1.1010,
            1.0980,
            -1,
            0.0002,
        )
        self.assertAlmostEqual(res_r, -0.0002 / 0.0011, places=4)

    def test_long_triggers_use_bid_stream(self):
        # Mid low dips below SL, but bid low does not -> SL must NOT trigger for long.
        # half spread = 0.0001 so bid low = 1.0996
        code, _, _ = resolve_trade_fast(
            np.array([1.1000], dtype=np.float32),
            np.array([1.1010], dtype=np.float32),
            np.array([1.0997], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            1.1000,
            1.0995,
            1.1020,
            1,
            0.0002,
        )
        self.assertNotEqual(code, 0)

    def test_short_triggers_use_ask_stream(self):
        # Mid high touches SL, ask high breaches SL -> SL should trigger for short.
        code, _, _ = resolve_trade_fast(
            np.array([1.1000], dtype=np.float32),
            np.array([1.1009], dtype=np.float32),
            np.array([1.0990], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            1.1000,
            1.1010,
            1.0980,
            -1,
            0.0002,
        )
        self.assertEqual(code, 0)

    def test_spread_widens_timeout_loss_for_flat_market(self):
        base_args = dict(
            opens=np.array([1.1000], dtype=np.float32),
            highs=np.array([1.1000], dtype=np.float32),
            lows=np.array([1.1000], dtype=np.float32),
            closes=np.array([1.1000], dtype=np.float32),
            entry_mid=1.1000,
            sl_price=1.0990,
            tp_price=1.1020,
            direction=1,
        )
        _, r_small, _ = resolve_trade_fast(spread=0.0001, **base_args)
        _, r_wide, _ = resolve_trade_fast(spread=0.0006, **base_args)
        self.assertLess(r_wide, r_small)

    def test_gap_fill_realism_uses_open_price(self):
        code, r_mult, bars = resolve_trade_fast(
            np.array([1.0990], dtype=np.float32),
            np.array([1.0990], dtype=np.float32),
            np.array([1.0980], dtype=np.float32),
            np.array([1.0990], dtype=np.float32),
            1.1000,
            1.0995,
            1.1020,
            1,
            0.0002,
        )
        self.assertEqual(code, 0)
        self.assertEqual(bars, 1)
        self.assertAlmostEqual(r_mult, -2.0, places=3)

    def test_same_candle_tp_sl_collision_is_conservative(self):
        # Both levels are reachable; engine should prioritize SL.
        code, _, _ = resolve_trade_fast(
            np.array([1.1000], dtype=np.float32),
            np.array([1.1030], dtype=np.float32),
            np.array([1.0970], dtype=np.float32),
            np.array([1.1000], dtype=np.float32),
            1.1000,
            1.0995,
            1.1020,
            1,
            0.0002,
        )
        self.assertEqual(code, 0)

    def test_high_volatility_stress_has_valid_outputs(self):
        rng = np.random.default_rng(7)
        n = 5_000
        opens = 1.1 + rng.normal(0, 0.0015, size=n).astype(np.float32)
        highs = opens + np.abs(rng.normal(0.0007, 0.0006, size=n)).astype(np.float32)
        lows = opens - np.abs(rng.normal(0.0007, 0.0006, size=n)).astype(np.float32)
        closes = opens + rng.normal(0, 0.0008, size=n).astype(np.float32)

        code, r_mult, bars = resolve_trade_fast(
            opens,
            highs,
            lows,
            closes,
            entry_mid=float(opens[0]),
            sl_price=float(opens[0] - 0.0015),
            tp_price=float(opens[0] + 0.0030),
            direction=1,
            spread=0.0003,
        )

        self.assertIn(code, (0, 1, 2))
        self.assertGreaterEqual(bars, 1)
        self.assertTrue(np.isfinite(r_mult))


if __name__ == '__main__':
    unittest.main()

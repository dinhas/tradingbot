import unittest
from datetime import datetime, timedelta

from backtest.rl_backtest import BacktestMetrics


class TestBacktestMetricsAccuracy(unittest.TestCase):
    def test_metrics_use_net_pnl_and_trade_timestamps(self):
        m = BacktestMetrics()
        t0 = datetime(2025, 1, 1, 0, 0)

        m.add_trade({
            'timestamp': t0,
            'asset': 'EURUSD',
            'pnl': 11.0,
            'net_pnl': 10.0,
            'fees': 1.0,
            'rr_ratio': 2.0,
            'sl_mult': 1.0,
            'spread_cost_est': 0.4,
            'realized_r': 1.1,
            'hold_time': 15,
        })
        m.add_trade({
            'timestamp': t0 + timedelta(days=1),
            'asset': 'EURUSD',
            'pnl': -5.0,
            'net_pnl': -6.0,
            'fees': 1.0,
            'rr_ratio': 1.0,
            'sl_mult': 1.4,
            'spread_cost_est': 0.6,
            'realized_r': -0.8,
            'hold_time': 10,
        })

        # intentionally different spacing to ensure trade_frequency is trade-timestamp based
        m.add_equity_point(t0, 1000)
        m.add_equity_point(t0 + timedelta(days=10), 1004)

        metrics = m.calculate_metrics()

        self.assertAlmostEqual(metrics['gross_profit'], 10.0)
        self.assertAlmostEqual(metrics['gross_loss'], 6.0)
        self.assertAlmostEqual(metrics['profit_factor'], 10.0 / 6.0)
        self.assertAlmostEqual(metrics['trade_frequency'], 2.0)  # 2 trades over 1 day span
        self.assertAlmostEqual(metrics['avg_spread_cost'], 0.5)
        self.assertAlmostEqual(metrics['avg_realized_r'], 0.15)
        self.assertAlmostEqual(metrics['pct_low_risk_sl'], 0.5)
        self.assertAlmostEqual(metrics['pct_rr_ge_1p2'], 0.5)

    def test_per_asset_breakdown_includes_realism_fields(self):
        m = BacktestMetrics()
        t0 = datetime(2025, 1, 1, 0, 0)

        m.add_trade({'timestamp': t0, 'asset': 'EURUSD', 'net_pnl': 4.0, 'pnl': 4.0, 'fees': 0.0, 'rr_ratio': 2.0, 'realized_r': 0.8, 'spread_cost_est': 0.1})
        m.add_trade({'timestamp': t0, 'asset': 'EURUSD', 'net_pnl': -2.0, 'pnl': -2.0, 'fees': 0.0, 'rr_ratio': 1.0, 'realized_r': -0.4, 'spread_cost_est': 0.2})

        asset = m.get_per_asset_metrics()['EURUSD']
        self.assertIn('avg_rr_ratio', asset)
        self.assertIn('avg_realized_r', asset)
        self.assertIn('avg_spread_cost', asset)
        self.assertAlmostEqual(asset['profit_factor'], 2.0)


if __name__ == '__main__':
    unittest.main()

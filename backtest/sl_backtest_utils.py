import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class BacktestMetrics:
    def __init__(self):
        self.trades = []
        self.equity_curve = []

    def add_trade(self, trade):
        self.trades.append(trade)

    def add_equity_point(self, timestamp, equity):
        self.equity_curve.append({'timestamp': timestamp, 'equity': equity})

    def calculate_metrics(self):
        if not self.trades:
            return {"Status": "No trades executed"}

        pnls = np.array([t['net_pnl'] for t in self.trades])
        total_pnl = pnls.sum()
        win_rate = (pnls > 0).mean()

        # Simple Sharpe (assuming 0 risk-free rate and 5m steps)
        # This is very rough
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-9) * np.sqrt(252 * 288)

        return {
            "Total Trades": len(self.trades),
            "Win Rate": win_rate,
            "Total Net PnL": total_pnl,
            "Avg PnL per Trade": pnls.mean(),
            "Rough Sharpe": sharpe,
            "Max Drawdown": self._calculate_max_drawdown()
        }

    def _calculate_max_drawdown(self):
        if not self.equity_curve: return 0
        equities = np.array([e['equity'] for e in self.equity_curve])
        peak = np.maximum.accumulate(equities)
        dd = (peak - equities) / peak
        return dd.max()

    def get_per_asset_metrics(self):
        if not self.trades: return {}
        df = pd.DataFrame(self.trades)
        asset_groups = df.groupby('asset')
        metrics = {}
        for asset, group in asset_groups:
            metrics[asset] = {
                'trades': len(group),
                'pnl': group['net_pnl'].sum(),
                'win_rate': (group['net_pnl'] > 0).mean()
            }
        return metrics

def generate_all_charts(metrics, per_asset, title, output_dir, timestamp):
    if not metrics.equity_curve: return

    plt.figure(figsize=(12, 6))
    df_equity = pd.DataFrame(metrics.equity_curve)
    plt.plot(df_equity['timestamp'], df_equity['equity'])
    plt.title(f"Equity Curve - {title}")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True)

    fname = os.path.join(output_dir, f"equity_curve_{timestamp}.png")
    plt.savefig(fname)
    plt.close()

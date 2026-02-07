
import pandas as pd
import numpy as np

file_path = 'backtest/results_alpha/trades_alpha_20260201_165828.csv'
df = pd.read_csv(file_path)

winners = df[df['pnl'] > 0].copy()
winners['is_tp'] = abs(winners['exit_price'] - winners['tp']) < 1e-8

print(f"Analysis for {file_path}")
print(f"Manual Exit Winners - Avg Hold Time: {winners.loc[~winners['is_tp'], 'hold_time'].mean():.2f} mins")
print(f"TP Hit Winners - Avg Hold Time: {winners.loc[winners['is_tp'], 'hold_time'].mean():.2f} mins")

losers = df[df['pnl'] < 0].copy()
losers['is_sl'] = abs(losers['exit_price'] - losers['sl']) < 1e-8
print(f"Manual Exit Losers - Avg Hold Time: {losers.loc[~losers['is_sl'], 'hold_time'].mean():.2f} mins")
print(f"SL Hit Losers - Avg Hold Time: {losers.loc[losers['is_sl'], 'hold_time'].mean():.2f} mins")

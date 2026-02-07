import pandas as pd
import numpy as np

file_path = 'backtest/results_alpha/trades_alpha_20260201_165828.csv'
df = pd.read_csv(file_path)

df['sl_dist'] = abs(df['entry_price'] - df['sl'])
df['achieved_rr'] = abs(df['exit_price'] - df['entry_price']) / df['sl_dist']
df['target_rr'] = abs(df['tp'] - df['entry_price']) / df['sl_dist']

winners = df[df['pnl'] > 0].copy()
winners['is_tp'] = abs(winners['exit_price'] - winners['tp']) < 1e-8

print(f"Analysis for {file_path}")
print(f"Total winning trades: {len(winners)}")
print(f"TP hits: {len(winners[winners['is_tp']])}")
print(f"Manual exits: {len(winners[~winners['is_tp']])}")

print("\nAchieved RR for manual exit winners:")
print(winners.loc[~winners['is_tp'], 'achieved_rr'].describe())

print("\nAchieved RR for TP winners:")
print(winners.loc[winners['is_tp'], 'achieved_rr'].describe())

losers = df[df['pnl'] < 0].copy()
losers['is_sl'] = abs(losers['exit_price'] - losers['sl']) < 1e-8
print(f"\nTotal losing trades: {len(losers)}")
print(f"SL hits: {len(losers[losers['is_sl']])}")
print(f"Manual exits: {len(losers[~losers['is_sl']])}")

if len(losers[~losers['is_sl']]) > 0:
    print("\nAchieved RR for manual exit losers:")
    print(losers.loc[~losers['is_sl'], 'achieved_rr'].describe())
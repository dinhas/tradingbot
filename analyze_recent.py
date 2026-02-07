
import pandas as pd
df = pd.read_csv('Alpha/backtest/results/trades_alpha_20260207_072540.csv')
print(f"Avg Hold Time: {df['hold_time'].mean():.2f} mins")
print(f"Win Rate: {len(df[df['pnl']>0])/len(df):.2%}")
print(f"Avg RR: {df['rr_ratio'].mean():.2f}")

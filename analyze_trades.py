import pandas as pd

df = pd.read_csv('Alpha/backtest/results/trades_alpha_20260207_072540.csv')

# Winners
winners = df[df['pnl'] > 0]

# Check how many winners hit TP exactly
hit_tp = winners[abs(winners['exit_price'] - winners['tp']) < 1e-8]
manual_exit_winners = winners[abs(winners['exit_price'] - winners['tp']) >= 1e-8]

print(f"Total trades: {len(df)}")
print(f"Total winners: {len(winners)}")
print(f"Winners hitting TP exactly: {len(hit_tp)}")
print(f"Winners exiting manually: {len(manual_exit_winners)}")

if not manual_exit_winners.empty:
    print("\nSample manual exit winners:")
    print(manual_exit_winners[['timestamp', 'asset', 'entry_price', 'exit_price', 'tp', 'sl', 'pnl']].head())
else:
    print("\nNo manual exit winners found in this file.")

# Also check losers
losers = df[df['pnl'] < 0]
hit_sl = losers[abs(losers['exit_price'] - losers['sl']) < 1e-8]
manual_exit_losers = losers[abs(losers['exit_price'] - losers['sl']) >= 1e-8]

print(f"\nTotal losers: {len(losers)}")
print(f"Losers hitting SL exactly: {len(hit_sl)}")
print(f"Losers exiting manually: {len(manual_exit_losers)}")
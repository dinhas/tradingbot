"""
Quick diagnostic script to analyze backtest trade data
"""
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("Usage: python backtest/diagnose_trades.py <trades_csv_file>")
    sys.exit(1)

trades_file = sys.argv[1]

print(f"Analyzing: {trades_file}\n")

df = pd.read_csv(trades_file)

print("="*60)
print("TRADE DATA SAMPLE (First 10 trades)")
print("="*60)
print(df[['asset', 'entry_price', 'exit_price', 'pnl', 'fees', 'hold_time']].head(10))

print("\n" + "="*60)
print("P&L STATISTICS")
print("="*60)
print(f"Total Trades: {len(df)}")
print(f"Min P&L: ${df['pnl'].min():.4f}")
print(f"Max P&L: ${df['pnl'].max():.4f}")
print(f"Mean P&L: ${df['pnl'].mean():.4f}")
print(f"Median P&L: ${df['pnl'].median():.4f}")
print(f"Std Dev P&L: ${df['pnl'].std():.4f}")

print("\n" + "="*60)
print("P&L DISTRIBUTION")
print("="*60)
print(f"Trades with P&L > 0: {len(df[df['pnl'] > 0])}")
print(f"Trades with P&L = 0: {len(df[df['pnl'] == 0])}")
print(f"Trades with P&L < 0: {len(df[df['pnl'] < 0])}")

print("\n" + "="*60)
print("PRICE MOVEMENT ANALYSIS")
print("="*60)
df['price_diff'] = df['exit_price'] - df['entry_price']
df['price_diff_pct'] = (df['price_diff'] / df['entry_price']) * 100

print(f"Min Price Diff: {df['price_diff'].min():.6f}")
print(f"Max Price Diff: {df['price_diff'].max():.6f}")
print(f"Mean Price Diff: {df['price_diff'].mean():.6f}")
print(f"Mean Price Diff %: {df['price_diff_pct'].mean():.4f}%")

print("\n" + "="*60)
print("SAMPLE TRADES WITH DETAILS")
print("="*60)
sample = df[['asset', 'action', 'entry_price', 'exit_price', 'pnl', 'fees', 'size']].head(5)
print(sample.to_string())

print("\n" + "="*60)
print("FEES ANALYSIS")
print("="*60)
print(f"Total Fees Paid: ${df['fees'].sum():.2f}")
print(f"Average Fee per Trade: ${df['fees'].mean():.4f}")
print(f"Total P&L (before fees): ${df['pnl'].sum():.2f}")
print(f"Total P&L (after fees): ${(df['pnl'] - df['fees']).sum():.2f}")

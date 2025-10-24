import pandas as pd
import sys
sys.path.append('.')

from src.backtest import backtest_method1_donchian, backtest_method2_atr, backtest_method3_volume

# Load data
print("Loading EUR/USD data...")
df = pd.read_csv('data/eurusd_5min_clean.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}\n")

print("="*80)
print("TESTING METHOD 1: Donchian Channel Breakout")
print("="*80)
results1 = backtest_method1_donchian(df, lookback_period=20)
print(f"Win Rate: {results1['win_rate']*100:.2f}%")
print(f"Total Return: {results1['total_return_pct']:.2f}%")
print(f"Profit Factor: {results1['profit_factor']:.2f}")
print(f"Number of Trades: {results1['num_trades']}")
print(f"Max Drawdown: {results1['max_drawdown_pct']:.2f}%")
print(f"Sharpe Ratio: {results1['sharpe_ratio']:.2f}")
print(f"\nFirst 5 trades:")
for i, trade in enumerate(results1['trades'][:5]):
    print(f"  {i+1}. {trade}")

print("\n" + "="*80)
print("TESTING METHOD 2: ATR Volatility Breakout")
print("="*80)
results2 = backtest_method2_atr(df, atr_period=14, atr_multiplier=1.5)
print(f"Win Rate: {results2['win_rate']*100:.2f}%")
print(f"Total Return: {results2['total_return_pct']:.2f}%")
print(f"Profit Factor: {results2['profit_factor']:.2f}")
print(f"Number of Trades: {results2['num_trades']}")
print(f"Max Drawdown: {results2['max_drawdown_pct']:.2f}%")
print(f"Sharpe Ratio: {results2['sharpe_ratio']:.2f}")
print(f"\nFirst 5 trades:")
for i, trade in enumerate(results2['trades'][:5]):
    print(f"  {i+1}. {trade}")

print("\n" + "="*80)
print("TESTING METHOD 3: Volume-Confirmed Breakout")
print("="*80)
results3 = backtest_method3_volume(df, lookback_period=20, volume_threshold=1.5, volume_ma_period=20)
print(f"Win Rate: {results3['win_rate']*100:.2f}%")
print(f"Total Return: {results3['total_return_pct']:.2f}%")
print(f"Profit Factor: {results3['profit_factor']:.2f}")
print(f"Number of Trades: {results3['num_trades']}")
print(f"Max Drawdown: {results3['max_drawdown_pct']:.2f}%")
print(f"Sharpe Ratio: {results3['sharpe_ratio']:.2f}")
print(f"\nFirst 5 trades:")
for i, trade in enumerate(results3['trades'][:5]):
    print(f"  {i+1}. {trade}")

print("\n" + "="*80)
print("✅ All backtest methods executed successfully!")
print("="*80)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.indicators import add_all_indicators

# Load the data
df = pd.read_csv('data/eurusd_5min_clean.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Loaded {len(df)} bars of EUR/USD 5-minute data")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print("\nFirst few rows of raw data:")
print(df.head())

# Add indicators with default periods
df_with_indicators = add_all_indicators(df, donchian_period=20, atr_period=14, volume_ma_period=20)

print("\n" + "="*80)
print("After adding indicators:")
print("="*80)
print(df_with_indicators.head(25))  # Show 25 rows to see after NaN period

print("\n" + "="*80)
print("Indicator Statistics:")
print("="*80)
print(df_with_indicators[['donchian_high', 'donchian_low', 'atr', 'volume_ma']].describe())

# Check for NaN values
print("\n" + "="*80)
print("NaN Check:")
print("="*80)
nan_counts = df_with_indicators.isna().sum()
print(nan_counts)

# Show a sample where signals might occur
print("\n" + "="*80)
print("Sample rows (100-110) - checking for potential breakout signals:")
print("="*80)
print(df_with_indicators.iloc[100:110][['timestamp', 'close', 'donchian_high', 'donchian_low', 'atr', 'volume', 'volume_ma']])

print("\n✅ Indicators calculated successfully!")

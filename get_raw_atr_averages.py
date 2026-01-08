import pandas as pd
import numpy as np
import os
import glob

def calculate_atr(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def analyze_raw_data():
    data_files = glob.glob("data/*.parquet")
    if not data_files:
        print("No parquet files found in data/ directory.")
        return

    results = []

    for file_path in data_files:
        pair_name = os.path.basename(file_path).replace("_5m.parquet", "")
        print(f"Processing {pair_name}...")
        
        df = pd.read_parquet(file_path)
        
        # Calculate ATR (Standard 14 period)
        df['atr'] = calculate_atr(df)
        
        avg_atr = df['atr'].mean()
        results.append({'Pair': pair_name, 'Average ATR': avg_atr})

    print("\n--- Raw OHLCV Average ATR (14-period) ---")
    stats_df = pd.DataFrame(results).sort_values(by='Average ATR', ascending=False)
    print(stats_df.to_string(index=False))

if __name__ == "__main__":
    analyze_raw_data()


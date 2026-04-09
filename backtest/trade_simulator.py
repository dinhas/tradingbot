
import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse

# SPREAD_PIPS and PIP_SIZE for spread calculations
SPREAD_PIPS = {
    "EURUSD": 1.2,
    "GBPUSD": 1.5,
    "USDJPY": 1.0,
    "USDCHF": 1.8,
    "XAUUSD": 45.0,
}

PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "USDCHF": 0.0001,
    "XAUUSD": 0.1,
}

def simulate_trade(asset, timestamp, direction, sl_mult, tp_mult, data_dir):
    file_path = Path(data_dir) / f"{asset}_5m_2025.parquet"
    if not file_path.exists():
        file_path = Path(data_dir) / f"{asset}_5m.parquet"
        
    if not file_path.exists():
        print(f"Error: Data for {asset} not found.")
        return

    df = pd.read_parquet(file_path)
    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    try:
        entry_idx = df.index.get_indexer([pd.to_datetime(timestamp)], method='nearest')[0]
    except:
        print(f"Error: Timestamp {timestamp} not found.")
        return

    actual_time = df.index[entry_idx]
    entry_price = df.iloc[entry_idx]['close']
    
    # Calculate ATR (rough estimation for simulation)
    # Using 14-period range
    if entry_idx < 14:
        atr = entry_price * 0.001
    else:
        atr = (df.iloc[entry_idx-14:entry_idx]['high'] - df.iloc[entry_idx-14:entry_idx]['low']).mean()
    
    sl_dist = sl_mult * atr
    tp_dist = tp_mult * atr
    
    sl = entry_price - (direction * sl_dist)
    tp = entry_price + (direction * tp_dist)
    
    print(f"\nSimulating {asset} {'BUY' if direction == 1 else 'SELL'} at {actual_time}")
    print(f"Entry: {entry_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f} (ATR: {atr:.5f})")
    
    forward_df = df.iloc[entry_idx+1 : entry_idx+1000]
    
    outcome = "Timed Out"
    exit_price = entry_price
    exit_time = actual_time
    bars = 0
    
    for i in range(len(forward_df)):
        bars += 1
        h, l = forward_df.iloc[i]['high'], forward_df.iloc[i]['low']
        t = forward_df.index[i]
        
        if direction == 1:
            if l <= sl:
                outcome = "STOP LOSS"
                exit_price = sl
                exit_time = t
                break
            elif h >= tp:
                outcome = "TAKE PROFIT"
                exit_price = tp
                exit_time = t
                break
        else:
            if h >= sl:
                outcome = "STOP LOSS"
                exit_price = sl
                exit_time = t
                break
            elif l <= tp:
                outcome = "TAKE PROFIT"
                exit_price = tp
                exit_time = t
                break
                
    pnl_pct = (exit_price - entry_price) / entry_price * direction
    print(f"Outcome: {outcome}")
    print(f"Exit Price: {exit_price:.5f} at {exit_time} ({bars} bars)")
    print(f"PnL: {pnl_pct:.2%}")
    
    # What-if analysis
    if outcome == "STOP LOSS":
        # Check if it would have hit TP eventually
        future_df = df.iloc[entry_idx+bars+1 : entry_idx+2000]
        for i in range(len(future_df)):
            h, l = future_df.iloc[i]['high'], future_df.iloc[i]['low']
            if (direction == 1 and h >= tp) or (direction == -1 and l <= tp):
                print(f"ADVICE: Price eventually hit TP at {future_df.index[i]}. A wider SL would have won.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, required=True)
    parser.add_argument("--time", type=str, required=True, help="YYYY-MM-DD HH:MM:SS")
    parser.add_argument("--dir", type=int, choices=[1, -1], required=True, help="1 for BUY, -1 for SELL")
    parser.add_argument("--sl", type=float, default=2.0)
    parser.add_argument("--tp", type=float, default=4.0)
    parser.add_argument("--data_dir", type=str, default="backtest/data")
    args = parser.parse_args()
    
    simulate_trade(args.asset, args.time, args.dir, args.sl, args.tp, args.data_dir)

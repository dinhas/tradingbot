
import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse

# SPREAD_PIPS and PIP_SIZE for spread calculations (matching backtest_combined.py)
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

def analyze_trades(trades_csv, data_dir):
    print(f"Loading trades from {trades_csv}...")
    df_trades = pd.read_csv(trades_csv)
    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
    
    assets = df_trades['asset'].unique()
    data_dict = {}
    
    print("Loading asset data...")
    for asset in assets:
        file_path = Path(data_dir) / f"{asset}_5m_2025.parquet"
        if not file_path.exists():
            file_path = Path(data_dir) / f"{asset}_5m.parquet"
        
        if file_path.exists():
            data_dict[asset] = pd.read_parquet(file_path)
            if 'timestamp' in data_dict[asset].columns:
                data_dict[asset].set_index('timestamp', inplace=True)
            data_dict[asset].sort_index(inplace=True)
        else:
            print(f"Warning: Could not find data for {asset}")

    results = []

    print("Analyzing trades...")
    for idx, trade in tqdm(df_trades.iterrows(), total=len(df_trades)):
        asset = trade['asset']
        if asset not in data_dict:
            continue
            
        data = data_dict[asset]
        entry_time = trade['timestamp'] - pd.Timedelta(minutes=5 * trade['hold_bars'])
        
        # Find entry index in data
        try:
            entry_idx = data.index.get_indexer([entry_time], method='nearest')[0]
        except:
            continue
            
        # Get forward data (next 500 bars)
        forward_data = data.iloc[entry_idx : entry_idx + 500]
        if len(forward_data) < 10:
            continue
            
        direction = trade['direction']
        entry_price = trade['entry_price']
        sl = trade['sl']
        tp = trade['tp']
        
        # Calculate MFE (Maximum Favorable Excursion)
        if direction == 1: # Long
            mfe_price = forward_data['high'].max()
            mae_price = forward_data['low'].min()
            mfe_dist = mfe_price - entry_price
            mae_dist = entry_price - mae_price
        else: # Short
            mfe_price = forward_data['low'].min()
            mae_price = forward_data['high'].max()
            mfe_dist = entry_price - mfe_price
            mae_dist = mae_price - entry_price
            
        tp_dist = abs(tp - entry_price)
        sl_dist = abs(sl - entry_price)
        
        # 1. Reversed Winner Test (Patience)
        # Did it hit TP after hitting SL?
        hit_sl = False
        hit_tp_after_sl = False
        tp_hit_idx = -1
        sl_hit_idx = -1
        
        for i in range(len(forward_data)):
            h, l = forward_data.iloc[i]['high'], forward_data.iloc[i]['low']
            if direction == 1:
                if not hit_sl and l <= sl:
                    hit_sl = True
                    sl_hit_idx = i
                if hit_sl and h >= tp:
                    hit_tp_after_sl = True
                    tp_hit_idx = i
                    break
                if not hit_sl and h >= tp: # Hit TP first
                    break
            else:
                if not hit_sl and h >= sl:
                    hit_sl = True
                    sl_hit_idx = i
                if hit_sl and l <= tp:
                    hit_tp_after_sl = True
                    tp_hit_idx = i
                    break
                if not hit_sl and l <= tp: # Hit TP first
                    break
        
        # 2. Cascading Failure (Good Stop)
        # Price continues significantly against us after SL
        continued_against = False
        if hit_sl:
            max_against_after_sl = mae_dist - sl_dist
            if max_against_after_sl > sl_dist * 0.5:
                continued_against = True

        # 3. Profit Evaporation (Greed)
        # MFE > 75% of TP but trade lost
        greed_issue = False
        if trade['pnl'] < 0 and mfe_dist > (tp_dist * 0.75):
            greed_issue = True

        # 4. Stale Signal (Time-Out)
        time_out_issue = trade['hold_bars'] > 150 and trade['pnl'] < 0

        # 5. Volatility Trap (ATR Buffer)
        # Assuming ATR is available or can be calculated simply
        # For now, just check if SL hit by a tiny margin
        spread = SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)
        tight_sl_hit = False
        if hit_sl:
            # Did it hit SL and then immediately recover?
            # Check price 10 bars after SL hit
            if sl_hit_idx + 10 < len(forward_data):
                recovery_price = forward_data.iloc[sl_hit_idx + 10]['close']
                if direction == 1 and recovery_price > sl:
                    tight_sl_hit = True
                elif direction == -1 and recovery_price < sl:
                    tight_sl_hit = True

        # 6. Directional Mismatch (Alpha Quality)
        alpha_fail = mfe_dist < (sl_dist * 0.2)

        # 7. Spread Squeeze (Cost)
        cost_issue = False
        if hit_sl:
            # If we hit SL by less than 1 spread
            if direction == 1:
                if (sl - forward_data.iloc[sl_hit_idx]['low']) < spread:
                    cost_issue = True
            else:
                if (forward_data.iloc[sl_hit_idx]['high'] - sl) < spread:
                    cost_issue = True

        # Categorize
        category = "Unknown"
        if trade['pnl'] > 0:
            category = "Winner"
        elif hit_tp_after_sl:
            category = "Bad SL (Reversed Winner)"
        elif continued_against:
            category = "Good SL (Trend Against)"
        elif greed_issue:
            category = "Greed (No Trailing/BE)"
        elif alpha_fail:
            category = "Alpha Fail (Wrong Direction)"
        elif time_out_issue:
            category = "Stale Signal (Time-out)"
        elif tight_sl_hit:
            category = "Volatility Trap (Need Buffer)"
        elif cost_issue:
            category = "Spread Squeeze (Thin Edge)"

        results.append({
            'timestamp': trade['timestamp'],
            'asset': asset,
            'pnl': trade['pnl'],
            'category': category,
            'mfe_pct_tp': mfe_dist / tp_dist if tp_dist > 0 else 0,
            'mae_pct_sl': mae_dist / sl_dist if sl_dist > 0 else 0,
            'hold_bars': trade['hold_bars']
        })

    results_df = pd.DataFrame(results)
    print("\n" + "="*40)
    print("DEBUG JOURNEY SUMMARY")
    print("="*40)
    summary = results_df['category'].value_counts()
    print(summary)
    print("\n" + "="*40)
    
    # Recommendations
    print("RECOMMENDATIONS:")
    if "Bad SL (Reversed Winner)" in summary and summary["Bad SL (Reversed Winner)"] > len(df_trades) * 0.1:
        print("- HIGH PRIORITY: Loosen SL multipliers. Many trades survive and hit TP eventually.")
    if "Greed (No Trailing/BE)" in summary and summary["Greed (No Trailing/BE)"] > len(df_trades) * 0.1:
        print("- HIGH PRIORITY: Implement Trailing Stop or Breakeven at 75% TP distance.")
    if "Alpha Fail (Wrong Direction)" in summary and summary["Alpha Fail (Wrong Direction)"] > len(df_trades) * 0.2:
        print("- HIGH PRIORITY: Alpha model direction is often wrong. Retrain Alpha or tighten Meta-Threshold.")
    if "Stale Signal (Time-out)" in summary and summary["Stale Signal (Time-out)"] > len(df_trades) * 0.1:
        print("- MEDIUM PRIORITY: Implement a time-based exit (e.g., exit after 150 bars).")
    if "Spread Squeeze (Thin Edge)" in summary and summary["Spread Squeeze (Thin Edge)"] > len(df_trades) * 0.1:
        print("- MEDIUM PRIORITY: Trading edge is too thin. Increase R:R or avoid high-spread assets.")

    # Save detailed results
    output_path = trades_csv.replace(".csv", "_debug_analysis.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed analysis saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades", type=str, required=True, help="Path to trades CSV")
    parser.add_argument("--data_dir", type=str, default="backtest/data", help="Path to OHLC data")
    args = parser.parse_args()
    
    analyze_trades(args.trades, args.data_dir)

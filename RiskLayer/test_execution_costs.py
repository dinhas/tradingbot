import sys
import os
import numpy as np
import pandas as pd

# Add project root to path to import Shared module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Shared.execution_engine import ExecutionEngine

def run_test():
    print("--- Starting Execution Cost Analysis (50 Data Points) ---")
    
    # Initialize Engine in 'Standard' mode (Retail settings)
    engine = ExecutionEngine(mode='standard')
    print(f"Engine Mode: {engine.mode}")
    print(f"Base Spread: {engine.base_spread_pips} pips")
    print(f"Commission: {engine.commission_pct * 100}%")
    print("-" * 60)

    # --- Generate 50 Mock Data Points ---
    # 25 EURUSD (Liquid, Non-JPY)
    # 25 USDJPY (Liquid, JPY - verifies bug fix)
    data = []
    
    # Generate EURUSD Data (Price ~1.1000, ATR ~0.0010 i.e., 10 pips)
    for i in range(25):
        data.append({
            'asset': 'EURUSD',
            'price': 1.1000 + (np.random.normal(0, 0.005)), 
            'atr': 0.0010 + (np.random.normal(0, 0.0002)),
            'lots': 1.0
        })
        
    # Generate USDJPY Data (Price ~150.00, ATR ~0.15 i.e., 15 pips)
    for i in range(25):
        data.append({
            'asset': 'USDJPY',
            'price': 150.00 + (np.random.normal(0, 0.5)),
            'atr': 0.15 + (np.random.normal(0, 0.02)),
            'lots': 1.0
        })

    results = []
    
    print(f"{ 'Asset':<8} | { 'Price':<8} | { 'ATR (pips)':<10} | { 'Spread (pips)':<13} | { 'Slip (pips)':<11} | { 'Comm ($)':<8} | { 'Total Cost (pips)':<15}")
    print("-" * 90)

    for row in data:
        cost_data = engine.calculate_total_cost(
            asset=row['asset'],
            price=row['price'],
            atr=row['atr'],
            direction=1, # Buy
            lots=row['lots']
        )
        
        # Calculate pip size for display conversion
        pip_size = engine.get_pip_scaler(row['asset'])
        atr_pips = row['atr'] / pip_size
        
        results.append({
            'asset': row['asset'],
            'spread_pips': cost_data['spread_pips'],
            'slippage_pips': cost_data['slippage_pips'],
            'commission_usd': cost_data['commission_usd'],
            'total_cost_pips': cost_data['total_cost_pips']
        })
        
        # Print first 2 and last 2 of each block to save space, or just first 10
        if len(results) <= 5 or (len(results) > 25 and len(results) <= 30):
            print(f"{row['asset']:<8} | {row['price']:<8.4f} | {atr_pips:<10.1f} | {cost_data['spread_pips']:<13.2f} | {cost_data['slippage_pips']:<11.2f} | {cost_data['commission_usd']:<8.2f} | {cost_data['total_cost_pips']:<15.2f}")

    print("-" * 90)
    
    # --- Analysis ---
    df = pd.DataFrame(results)
    
    # Group by Asset to show JPY fix vs Normal
    summary = df.groupby('asset').agg({
        'spread_pips': 'mean',
        'slippage_pips': 'mean',
        'commission_usd': 'mean',
        'total_cost_pips': 'mean'
    })
    
    print("\n--- SUMMARY STATISTICS (Averages) ---")
    print(summary)
    
    print("\n--- INTERPRETATION ---")
    eur_spread = summary.loc['EURUSD', 'spread_pips']
    jpy_spread = summary.loc['USDJPY', 'spread_pips']
    
    if 1.0 < eur_spread < 2.5:
        print(f"[PASS] EURUSD Spread ({eur_spread:.2f} pips) is realistic (Target: 1.2 base + ATR).")
    else:
        print(f"[FAIL] EURUSD Spread ({eur_spread:.2f} pips) is abnormal.")
        
    if 1.0 < jpy_spread < 2.5:
        print(f"[PASS] USDJPY Spread ({jpy_spread:.2f} pips) is realistic. JPY Bug is FIXED.")
    elif jpy_spread < 0.5:
        print(f"[FAIL] USDJPY Spread ({jpy_spread:.2f} pips) is too low. JPY Bug persists (0.01 vs 0.0001 scalar issue).")
        
    avg_comm = df['commission_usd'].mean()
    if avg_comm == 0.0:
        print(f"[INFO] Commission is $0.00 (Standard/Retail Mode).")
    else:
        print(f"[INFO] Commission is ${avg_comm:.2f}/lot (Raw/ECN Mode).")

if __name__ == "__main__":
    run_test()

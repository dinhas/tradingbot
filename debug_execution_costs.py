import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Ensure we can import Shared.execution
sys.path.append(os.getcwd())

from Shared.execution import ExecutionEngine, TradeConfig

# Setup a deterministic execution engine for testing
# We want to see the "random" slippage, but maybe for analysis we can mock it or just observe it.
# We will use the default config but maybe force some values if needed.
config = TradeConfig()
engine = ExecutionEngine(config)

# 1. Create 5 lines of mock data (EURUSD context)
# Columns: Time, Open, High, Low, Close, ATR
data_points = [
    {"time": "10:00", "mid_open": 1.05000, "mid_high": 1.05100, "mid_low": 1.04900, "mid_close": 1.05050, "atr": 0.0010}, # Volatile
    {"time": "10:05", "mid_open": 1.05050, "mid_high": 1.05060, "mid_low": 1.05040, "mid_close": 1.05045, "atr": 0.0005}, # Quiet
    {"time": "10:10", "mid_open": 1.05045, "mid_high": 1.05200, "mid_low": 1.05000, "mid_close": 1.05150, "atr": 0.0015}, # High Volatility
    {"time": "10:15", "mid_open": 1.05150, "mid_high": 1.05155, "mid_low": 1.05145, "mid_close": 1.05150, "atr": 0.0002}, # Very low vol
    {"time": "10:20", "mid_open": 1.05150, "mid_high": 1.05300, "mid_low": 1.04800, "mid_close": 1.05000, "atr": 0.0020}, # Extreme
]

print(f"{ '='*100}")
print(f"{ 'EXECUTION ENGINE DEBUG & FEE ANALYSIS':^100}")
print(f"{ '='*100}")
print(f"Config:")
print(f"  Contract Size: {config.CONTRACT_SIZE}")
print(f"  Spread Min (Pips): {config.SPREAD_MIN_PIPS}")
print(f"  Spread ATR Factor: {config.SPREAD_ATR_FACTOR}")
print(f"  Slippage (Pips): {config.SLIPPAGE_MIN_PIPS} - {config.SLIPPAGE_MAX_PIPS}")
print(f"{'-'*100}")

results = []

for i, row in enumerate(data_points):
    print(f"\nProcessing Data Point {i+1} | Time: {row['time']} | ATR: {row['atr']:.5f}")
    
    # We will simulate a LONG trade entering at 'mid_close' (current price)
    mid_price = row['mid_close']
    atr = row['atr']
    direction = 1 # LONG
    
    # 1. Get Spread
    spread_price_units = engine.get_spread(mid_price, atr)
    spread_pips = spread_price_units * 10000
    
    # 2. Get Entry Execution
    # Note: get_entry_price adds spread + random slippage for LONG
    # LONG Entry = Mid + Spread + Slippage
    # We run it multiple times to see slippage range or just once? Just once per row.
    entry_price_exec = engine.get_entry_price(mid_price, direction, atr)
    
    # Decompose execution cost
    # Theoretical Ask (Mid + Spread)
    theoretical_ask = mid_price + spread_price_units
    # Slippage is the remainder
    slippage_paid = entry_price_exec - theoretical_ask
    slippage_pips = slippage_paid * 10000
    
    # Total Entry Cost (vs Mid)
    total_entry_cost_price = entry_price_exec - mid_price
    total_entry_cost_pips = total_entry_cost_price * 10000
    
    # 3. Position Sizing (Assume $10,000 equity, risk 1% for test or use calculate_position_size)
    # We'll use the engine's sizing to be authentic
    # Let's say we target a wide SL for sizing calculation to get a reasonable lot size
    sl_mult = 1.0
    sl_dist_price = sl_mult * atr
    
    lots = engine.calculate_position_size(
        equity=10000,
        initial_equity=10000,
        entry_price=entry_price_exec,
        sl_dist_price=sl_dist_price,
        atr=atr,
        is_usd_quote=True # EURUSD
    )
    
    # 4. Simulate Exits (TP and SL)
    # For TP/SL prices, we use the helper
    tp_mult = 2.0
    sl_price_target, tp_price_target, _, _ = engine.get_exit_prices(entry_price_exec, direction, sl_mult, tp_mult, atr)
    
    # -- Scenario A: TP Hit --
    # Exit at TP Target price. 
    # Logic in engine: Close LONG = Sell at Bid (Target - Slippage)
    # Wait, usually TP is a limit order, so no slippage?
    # The engine's `get_close_price` applies spread/slippage logic indiscriminately if used directly.
    # In `backtest.py`, it says: "if exit_reason == 'TIME': get_close_price ... else: exit_execution_price = exit_price"
    # This implies Limit Orders (SL/TP) execute EXACTLY at the price in the backtester? 
    # Let's verify `backtest.py`:
    #   if low <= sl_price: exit_price = sl_price ...
    #   if high >= tp_price: exit_price = tp_price ...
    #   if exit_reason == 'TIME': ... get_close_price
    #   else: exit_execution_price = exit_price
    # So the backtester assumes GUARANTEED EXECUTION at SL/TP levels (no slippage on limits).
    # However, for this "deep analysis", user might want to see what happens if we DID pay spread/slip on exit (Market Exit).
    # We will simulate a "Market Exit" at those levels to show the costs involved if closing manually.
    
    exit_exec_tp_market = engine.get_close_price(tp_price_target, direction, atr) # Sell at Bid
    # Cost to close LONG = (Mid - Execution). Mid is approx TP target? 
    # Actually if Mid hits TP target, Bid is (TP - spread). 
    # So we pay spread on exit.
    
    # Let's stick to the PnL calculation provided by the engine.
    pnl_tp = engine.calculate_pnl(entry_price_exec, tp_price_target, lots, direction) # Assuming limit exit
    pnl_sl = engine.calculate_pnl(entry_price_exec, sl_price_target, lots, direction) # Assuming limit exit
    
    # Dollar Costs
    # Spread Cost ($) = Spread (price) * Lots * Contract Size
    cost_spread_usd = spread_price_units * lots * config.CONTRACT_SIZE
    cost_slippage_usd = slippage_paid * lots * config.CONTRACT_SIZE
    
    print(f"  Entry Mid: {mid_price:.5f}")
    print(f"  Exec Price: {entry_price_exec:.5f}")
    print(f"  Lots: {lots:.2f}")
    print(f"  --- Costs ---")
    print(f"  Spread: {spread_pips:.2f} pips (${cost_spread_usd:.2f})")
    print(f"  Slippage: {slippage_pips:.2f} pips (${cost_slippage_usd:.2f})")
    print(f"  Total Entry Cost: {total_entry_cost_pips:.2f} pips (${cost_spread_usd + cost_slippage_usd:.2f})")
    
    print(f"  --- Scenarios ---")
    print(f"  TP Target: {tp_price_target:.5f} | PnL: ${pnl_tp:.2f}")
    print(f"  SL Target: {sl_price_target:.5f} | PnL: ${pnl_sl:.2f}")
    
    results.append({
        "ID": i+1,
        "Lots": round(lots, 2),
        "Spread(pips)": round(spread_pips, 2),
        "Spread($)": round(cost_spread_usd, 2),
        "Slip(pips)": round(slippage_pips, 2),
        "Slip($)": round(cost_slippage_usd, 2),
        "EntryCost($)": round(cost_spread_usd + cost_slippage_usd, 2),
        "TP_PnL": round(pnl_tp, 2),
        "SL_PnL": round(pnl_sl, 2)
    })

print("\n" + "="*100)
print("SUMMARY TABLE")
print("="*100)
df = pd.DataFrame(results)
print(df.to_string(index=False))
print("="*100)

print("\nAnalysis Notes:")
print("1. Spread is dynamic based on ATR (Config: SPREAD_ATR_FACTOR). Higher ATR = Wider Spread.")
print("2. Slippage is random (Uniform dist) added to the spread for Market Orders.")
print("3. Entry Price is always WORSE than Mid Price by (Spread + Slippage).")
print("4. PnL calculations assume Limit Exits (SL/TP) execute exactly at the level (no exit slip/spread in this generic test).")
print("   * If Market Exit was used, additional spread/slippage costs would apply on exit.")

# Save results to CSV
output_csv = "execution_costs.csv"
df.to_csv(output_csv, index=False)
print(f"\n[Saved] Detailed data saved to {output_csv}")

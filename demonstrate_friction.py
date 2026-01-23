import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from Shared.execution import ExecutionEngine, TradeConfig

def test_friction():
    print("--- Friction Analysis on $10 Account ---")
    
    # 1. Setup
    config = TradeConfig()
    engine = ExecutionEngine(config)
    
    equity = 10.0
    entry_price_mid = 1.10000 # EURUSD-ish
    atr = 0.0010 # 10 pips ATR (low volatility)
    direction = 1 # Long
    
    print(f"Initial Equity: ${equity:.2f}")
    print(f"Asset: EURUSD (Mid: {entry_price_mid})")
    print(f"ATR: {atr:.5f}")
    print(f"Friction Settings:")
    print(f"  Spread: {config.SPREAD_MIN_PIPS} pips + {config.SPREAD_ATR_FACTOR*100}% ATR")
    print(f"  Slippage: {config.SLIPPAGE_MIN_PIPS}-{config.SLIPPAGE_MAX_PIPS} pips")
    
    # 2. Calculate Costs - USING ENGINE LOGIC (Source of Truth)
    # Simulate an immediate Buy and Sell to see the spread cost
    entry_price = engine.get_entry_price(entry_price_mid, direction, atr)
    exit_price = engine.get_close_price(entry_price_mid, direction, atr) # Close immediately
    
    # Calculate Friction
    # For Long: Buy at Ask, Sell at Bid. 
    # Cost = Entry - Exit
    friction_price = entry_price - exit_price
    friction_pips = friction_price * 10000
    
    print(f"\n--- Cost Calculation (Engine Actual) ---")
    print(f"Entry Price: {entry_price:.5f}")
    print(f"Exit Price:  {exit_price:.5f}")
    print(f"Total Friction: {friction_pips:.2f} pips")
    
    # 3. Impact on Min Lot Trade
    lots = 0.01
    contract_size = 100000
    pip_value = 10.0 # Per lot for EURUSD
    
    cost_usd = friction_pips * (pip_value * lots)
    cost_pct = (cost_usd / equity) * 100
    
    print(f"\n--- Trade Impact (0.01 Lots) ---")
    print(f"Cost in USD: ${cost_usd:.4f}")
    print(f"Cost % of Equity: {cost_pct:.2f}%")
    
    print(f"\nCONCLUSION:")
    if cost_pct > 1.0:
        print(f"CRITICAL: You are losing {cost_pct:.2f}% of your account per trade just on fees/spread.")
        print("No AI can overcome a 2%+ hurdle per trade without massive Alpha.")
    else:
        print("Friction is manageable.")

if __name__ == "__main__":
    test_friction()

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from Shared.execution import ExecutionEngine, TradeConfig

def test_friction_loop():
    print("--- Friction Simulation (Spread Only) ---")
    
    # 1. Setup
    config = TradeConfig()
    engine = ExecutionEngine(config)
    
    initial_equity = 10000.0
    equity = initial_equity
    
    # Market Conditions
    entry_price_mid = 1.10000 
    atr = 0.0010 
    direction = 1 # Long
    lots = 0.01
    pip_value = 10.0 # Standard Lot EURUSD
    
    print(f"Start Equity: ${equity:.2f}")
    print(f"Simulating 5000 trades (Entry -> Immediate Exit)")
    
    # 2. Loop
    print("\nStep | Equity | Friction Cost | Cost %")
    print("-" * 40)
    
    total_cost = 0.0
    
    for i in range(1, 5001):
        # Calculate Costs - USING ENGINE LOGIC
        entry_price = engine.get_entry_price(entry_price_mid, direction, atr)
        exit_price = engine.get_close_price(entry_price_mid, direction, atr)
        
        # Friction per trade (Spread)
        price_diff = entry_price - exit_price # Buying higher, selling lower
        cost_usd = price_diff * (lots * config.CONTRACT_SIZE)
        
        equity -= cost_usd
        total_cost += cost_usd
        
        if i % 1000 == 0:
            cost_pct = (cost_usd / equity) * 100
            print(f"{i:3d} | ${equity:.2f} | -${cost_usd:.4f}     | {cost_pct:.4f}%")

    print("-" * 40)
    print(f"Final Equity: ${equity:.2f}")
    print(f"Total Friction Paid: ${total_cost:.2f}")
    drop_pct = ((initial_equity - equity) / initial_equity) * 100
    print(f"Total Account Drop: {drop_pct:.2f}%")

if __name__ == "__main__":
    test_friction_loop()

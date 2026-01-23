# Position Sizing Cleanup Summary

## Problem Identified
The codebase had **misleading code** that suggested dynamic position sizing based on model confidence, but this was broken and causing training instability:

1. **Training reward inconsistency**: With a total PnL-based reward, dynamic position sizing made a -25% loss feel the same as a -50% loss during drawdown periods
2. **Broken orchestrator code**: LiveExecution was trying to access `risk_action[2]` (3rd output) that didn't exist in the model
3. **Confusing comments**: Multiple references to "dynamic sizing" when risk was actually hardcoded

## Changes Made

### 1. Shared/execution.py
- **Removed** `risk_pct` from `decode_action()` return value (now returns only `sl_mult, tp_mult`)
- **Removed** `risk_pct` parameter from `calculate_position_size()` 
- **Simplified** logic to always use `config.DEFAULT_RISK_PCT` (25%)
- **Updated** docstrings to clarify FIXED 25% risk per trade

### 2. RiskLayer/src/risk_env.py  
- **Updated** `step()` to unpack only 2 values from `decode_action()`
- **Clarified** docstring: model learns SL/TP placement, NOT position sizing
- **Removed** misleading "dynamic sizing removed" comment

### 3. backtest/backtest_combined_v2.py
- **Updated** to match new `decode_action()` signature (2 outputs)
- **Removed** unused `risk_pct` variable

### 4. LiveExecution/src/orchestrator.py
- **Imported** `ExecutionEngine` for consistency
- **Removed** manual SL/TP calculation and `risk_action[2]` access attempt  
- **Replaced** with engine calls: `decode_action()`, `get_exit_prices()`, `calculate_position_size()`
- **Eliminated** broken dynamic risk scaling logic

## Current Behavior
✅ **Risk Percentage**: FIXED at 25% of account equity per trade  
✅ **Lot Size**: DYNAMIC (calculated from equity, SL distance, and ATR)  
✅ **Model Control**: Only SL/TP multipliers (0.2-2.0 ATR for SL, 0.5-4.0 ATR for TP)

## Why This is Better
1. **Consistent training signal**: A $100 loss always feels like a $100 loss to the model
2. **No broken code**: Orchestrator no longer tries to access non-existent model outputs
3. **Clear intent**: Code now accurately reflects what the system does
4. **Shared logic**: All environments (train/backtest/live) use identical position sizing math

## What Still Scales Dynamically
- **Lot size** scales with:
  - Current equity (more money = larger positions)
  - ATR (higher volatility = smaller positions for same $ risk)
  - SL distance (wider stops = smaller positions)
  - Leverage limits (max 400x, 80% margin per trade)

The key insight: **lot sizing is dynamic, but risk percentage is fixed at 25%**.

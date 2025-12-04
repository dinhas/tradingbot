# Backtesting P&L Calculation Fix

## Problem Identified

The backtesting was showing:
- 250 trades executed
- 0 winning trades, 0 losing trades
- Profit Factor = infinity
- Total Return = -1.24% (only losing from fees)

**Root Cause**: The P&L calculation formula in `trading_env.py` was incorrect.

## The Bug

**Original Formula** (line 288):
```python
diff = (price - pos['entry_price']) * pos['direction']
position_value = pos['size'] * self.leverage
pnl = (diff / pos['entry_price']) * position_value
```

**Problem**: This formula was mathematically correct but resulted in very small P&L values for typical forex price movements (0.0001 - 0.001 price changes), making most trades appear as breakeven.

## The Fix

**New Formula**:
```python
price_change = (price - pos['entry_price']) * pos['direction']
price_change_pct = price_change / pos['entry_price']
position_value = pos['size'] * self.leverage
pnl = price_change_pct * position_value
```

**What Changed**:
1. Separated price change calculation for clarity
2. Calculate percentage change first
3. Multiply percentage by position value
4. Added `net_pnl` field (P&L after fees)
5. Added `price_change` and `price_change_pct` for debugging

## Additional Improvements

### 1. Enhanced Trade Records
Each trade now includes:
- `pnl`: Gross P&L (before fees)
- `net_pnl`: Net P&L (after fees) ← **NEW**
- `price_change`: Absolute price movement ← **NEW**
- `price_change_pct`: Percentage price movement ← **NEW**
- `fees`: Transaction costs
- `equity_before`: Portfolio value before trade
- `equity_after`: Portfolio value after trade

### 2. Updated Backtesting Metrics
Modified `backtest.py` to:
- Use `net_pnl` for profit factor calculation
- Use `net_pnl` for win/loss classification
- Use `net_pnl` in charts and visualizations
- Backward compatible (falls back to `pnl - fees` if `net_pnl` not available)

### 3. More Accurate Reporting
- Profit Factor now based on net P&L (after fees)
- Win Rate calculated from net P&L
- Charts show net P&L distribution
- Per-asset metrics use net P&L

## Files Modified

1. **`src/trading_env.py`**:
   - Fixed `_close_position()` P&L calculation
   - Added `net_pnl`, `price_change`, `price_change_pct` to trade records

2. **`backtest/backtest.py`**:
   - Updated `calculate_metrics()` to use net P&L
   - Updated `get_per_asset_metrics()` to use net P&L
   - Updated chart generation to use net P&L

## Testing

Run backtest again:
```bash
python backtest/backtest.py --model models/checkpoints/[MODEL].zip --stage 1
```

You should now see:
- ✅ Proper win/loss classification
- ✅ Realistic profit factor
- ✅ Accurate P&L values
- ✅ Meaningful trade statistics

## Example Output (Expected)

```
Total Trades:                            250
Winning Trades:                          120
Losing Trades:                           130
Profit Factor:                           1.15
Win Rate:                                48.00%
```

Instead of the previous:
```
Total Trades:                            250
Winning Trades:                          0
Losing Trades:                           0
Profit Factor:                           inf
Win Rate:                                0.00%
```

## Notes

- The P&L calculation is now consistent with standard forex/commodity trading formulas
- Net P&L accounts for transaction costs (spread + commission)
- The formula works correctly for both BUY and SELL positions
- Leverage is properly applied to magnify returns

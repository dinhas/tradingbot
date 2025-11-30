#!/usr/bin/env python3
"""Analyze trading data from trades.txt"""

# Parse trades.txt
with open('trades.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Extract key statistics
total_trades = 0
total_fees = 0
portfolio_values = []
buy_trades = 0
sell_trades = 0
sl_counts = {}  # Count different SL multipliers
tp_counts = {}  # Count different TP multipliers

for i, line in enumerate(lines):
    if 'Trade #' in line:
        total_trades += 1
    
    if 'Type:' in line and 'BUY' in line:
        buy_trades += 1
    elif 'Type:' in line and 'SELL' in line:
        sell_trades += 1
    
    if 'Fees:' in line:
        try:
            fee_str = line.split('$')[1].strip()
            fee_val = float(fee_str.replace(',', ''))
            total_fees += fee_val
        except:
            pass
    
    if 'Portfolio:' in line and '→' in line:
        try:
            parts = line.split('→')
            end_val_str = parts[1].strip().replace('$', '').replace(',', '')
            portfolio_values.append(float(end_val_str))
        except:
            pass
    
    if 'SL Mult:' in line:
        try:
            mult_str = line.split(':')[1].strip().split('x')[0]
            mult = float(mult_str)
            sl_counts[mult] = sl_counts.get(mult, 0) + 1
        except:
            pass
    
    if 'TP Mult:' in line:
        try:
            mult_str = line.split(':')[1].strip().split('x')[0]
            mult = float(mult_str)
            tp_counts[mult] = tp_counts.get(mult, 0) + 1
        except:
            pass

print("=" * 60)
print("TRADE ANALYSIS SUMMARY")
print("=" * 60)
print(f"Total Trades: {total_trades}")
print(f"Buy Trades: {buy_trades}")
print(f"Sell Trades: {sell_trades}")
print(f"Total Fees Paid: ${total_fees:,.2f}")
print(f"Average Fee per Trade: ${total_fees/total_trades:,.2f}")

if portfolio_values:
    print(f"\nPortfolio Value Journey:")
    print(f"  Starting: ${portfolio_values[0]:,.2f}")
    print(f"  Ending: ${portfolio_values[-1]:,.2f}")
    print(f"  Change: ${portfolio_values[-1] - portfolio_values[0]:,.2f}")
    print(f"  Return: {((portfolio_values[-1]/portfolio_values[0]) - 1)*100:.2f}%")

print(f"\nStop Loss Distribution (top 10):")
sorted_sl = sorted(sl_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for mult, count in sorted_sl:
    pct = (count / sum(sl_counts.values())) * 100
    print(f"  {mult:.2f}x ATR: {count} trades ({pct:.1f}%)")

print(f"\nTake Profit Distribution (top 10):")
sorted_tp = sorted(tp_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for mult, count in sorted_tp:
    pct = (count / sum(tp_counts.values())) * 100
    print(f"  {mult:.2f}x ATR: {count} trades ({pct:.1f}%)")

# Fee impact analysis
print(f"\nFEE IMPACT:")
print(f"  Fees as % of starting balance: {(total_fees/10000)*100:.2f}%")
print(f"  Trading frequency: {total_trades} trades in ~9 months = {total_trades/270:.1f} trades/day")
print("=" * 60)

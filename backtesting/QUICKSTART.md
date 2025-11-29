# Quick Start: Running the Backtest

## Prerequisites
Ensure you have these files in the project root (`e:\tradingbot\`):
- ✅ `best_model.zip`
- ✅ `final_model_vecnormalize.pkl`

## Run the Backtest

```bash
# From the project root
python backtesting/backtest_runner.py
```

## What You'll Get

### 1. Trade Logs
- **`backtesting/trades.txt`** → All 2025 trades in readable format
- **`backtesting/trades_all.csv`** → All trades (all years) in CSV

### 2. Visualizations
- **`backtest_portfolio_value.png`** → Portfolio growth chart
- **`backtest_drawdown.png`** → Risk/drawdown chart
- **`backtest_BTC_trades.png`** → BTC price + buy/sell dots
- **`backtest_ETH_trades.png`** → ETH price + buy/sell dots
- **`backtest_SOL_trades.png`** → SOL price + buy/sell dots
- **`backtest_EUR_trades.png`** → EUR price + buy/sell dots
- **`backtest_GBP_trades.png`** → GBP price + buy/sell dots
- **`backtest_JPY_trades.png`** → JPY price + buy/sell dots
- **`backtest_all_assets_overview.png`** → All 6 assets in one chart

### 3. Metrics
- **`backtest_metrics.json`** → Performance statistics

## Example Output

```
Loading model from best_model.zip...
Loading normalization stats from final_model_vecnormalize.pkl...

Starting Backtest Loop...
================================================================================
Step 500: 2024-03-15 14:30:00 | Portfolio=$10234.56 | Return=0.0023 | Trades=45
Step 1000: 2024-06-20 09:15:00 | Portfolio=$10567.89 | Return=-0.0012 | Trades=92
...
================================================================================
✅ Backtest Finished!
Total Steps: 35040
Total Trades: 1250
Final Portfolio Value: $12345.67
Total Return: 23.46%
================================================================================

💾 Saving Trade Logs...
✅ Saved 450 trades from 2025 to backtesting/trades.txt
✅ Saved all 1250 trades to backtesting/trades_all.csv

📊 Generating Visualizations...
✅ Saved backtest_portfolio_value.png
✅ Saved backtest_drawdown.png
✅ Saved backtest_BTC_trades.png (150 trades)
✅ Saved backtest_ETH_trades.png (120 trades)
✅ Saved backtest_SOL_trades.png (100 trades)
✅ Saved backtest_EUR_trades.png (80 trades)
✅ Saved backtest_GBP_trades.png (70 trades)
✅ Saved backtest_JPY_trades.png (60 trades)
✅ Saved backtest_all_assets_overview.png

📈 Calculating Final Metrics...
✅ Saved metrics to backtesting/backtest_metrics.json

================================================================================
📊 BACKTEST SUMMARY
================================================================================
{
  "model_path": "best_model.zip",
  "total_trades": 1250,
  "trades_2025": 450,
  "final_value": 12345.67,
  "total_return_pct": 23.46,
  "max_drawdown_pct": 12.34
}
================================================================================
```

## Understanding the Charts

### Price Charts with Trading Markers
- **Green Line**: Asset price over time
- **Green Triangles (▲)**: Buy orders
- **Red Triangles (▼)**: Sell orders

### No Stop-Loss/Take-Profit Markers
As requested, charts only show actual buy/sell actions taken by the model, not automatic SL/TP levels.

## Analyzing Results

1. **Check `trades.txt`** to see what the model did in 2025
2. **Open the PNG charts** to visualize trading patterns
3. **Import `trades_all.csv`** into Excel for deeper analysis
4. **Review `backtest_metrics.json`** for key performance indicators

## Common Issues

### Missing Model Files
```
❌ Model file 'best_model.zip' not found in current directory.
```
**Solution**: Copy `best_model.zip` to `e:\tradingbot\`

### Missing Normalizer
```
⚠️ Warning: Normalization stats not found! Model performance may be degraded.
```
**Solution**: Copy `final_model_vecnormalize.pkl` to `e:\tradingbot\`

### Missing Data Files
```
FileNotFoundError: Data for BTC not found.
```
**Solution**: Run `backtesting/backtest_data_fetcher.py` first to get 2025 data

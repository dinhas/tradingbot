# Backtesting Instructions

This directory contains scripts to backtest the trained AI model on unseen data (2025).

## Prerequisites
- Python 3.10+
- Dependencies listed in `../requirements.txt`
- cTrader account credentials (configured in `backtest_data_fetcher.py`)

## Steps

1. **Fetch Data**:
   Run the data fetcher to download historical data (2020-Present) and generate normalization statistics.
   ```bash
   python backtest_data_fetcher.py
   ```
   This will create `backtest_data_{asset}.parquet` files and `volatility_baseline.json` in this directory.

2. **Run Backtest**:
   Run the backtest runner to execute the model on the fetched data.
   ```bash
   python backtest_runner.py
   ```
   This will:
   - Load the model from `../final_model.zip`
   - Load normalization stats from `../final_model_vecnormalize.pkl`
   - Run the trading environment
   - Generate `backtest_portfolio_value.png`, `backtest_drawdown.png`, and `backtest_metrics.json`.

## Output
- **backtest_metrics.json**: Summary of performance (Return, Drawdown, Fees).
- **backtest_portfolio_value.png**: Chart of portfolio value over time.
- **backtest_drawdown.png**: Chart of drawdown over time.

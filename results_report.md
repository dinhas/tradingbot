# Alpha Model Training and Backtest Results (30M Data)

## Training Summary
- **Data Period**: 2022-04-11 to 2025-12-31 (approx. 45,000 bars per asset)
- **Timeframe**: 30M
- **Assets**: EURUSD, GBPUSD, XAUUSD, USDCHF, USDJPY
- **Labeling Method**: Triple Barrier Method
  - **Take Profit**: 2.0x ATR
  - **Stop Loss**: 1.0x ATR
  - **Time Barrier**: 7 candles
  - **Filters**: None (No ADX or HTF Trend filters used)
- **Target Distribution**:
  - Buy (+1): 34,515 (16.1%)
  - Sell (-1): 35,318 (16.5%)
  - Neutral (0): 144,582 (67.4%)
- **Training Epochs**: 16 (Early stopping triggered)
- **Best Validation Loss**: 0.6238

## Backtest Summary
- **Backtester**: Ultra-Fast Vectorized Alpha LSTM Backtester
- **Initial Equity**: $10,000.00
- **Final Equity**: $667,356.50
- **Total Return**: 6,573.57%
- **Sharpe Ratio**: 6.11
- **Profit Factor**: 1.108
- **Win Rate**: 45.60%
- **Max Drawdown**: -41.47%
- **Total Trades**: 3,219
- **Avg Hold Time**: 168.04 minutes (approx. 5.6 bars)

## Key Findings
- Removing regime filters and using a purely technical approach on 30M data with the specified TBM parameters yielded a highly aggressive and profitable model in this backtest.
- The 1.0x SL and 2.0x TP configuration provides a positive expectancy, which the LSTM model was able to exploit effectively across the tested assets.
- The Sharper Ratio of 6.11 indicates high risk-adjusted performance, though the Max Drawdown of 41% suggests significant volatility in equity growth.

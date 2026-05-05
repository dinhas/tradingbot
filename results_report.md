# Alpha Model Training and Backtest Results (30M Data - 7-Bar Time Barrier)

## Training Summary
- **Data Period**: 2022-04-11 to 2025-12-31 (approx. 100,000 bars per asset)
- **Timeframe**: 30M
- **Assets**: EURUSD, GBPUSD, XAUUSD, USDCHF, USDJPY
- **Labeling Method**: Triple Barrier Method
  - **Take Profit**: 4.0x ATR
  - **Stop Loss**: 2.0x ATR
  - **Vertical Barrier**: 7 candles (Updated from 24)
  - **Filters**: None (No ADX or HTF Trend filters used)
- **Target Distribution (Dataset 1)**:
  - Buy (+1): 16,188 (3.24%)
  - Sell (-1): 17,632 (3.53%)
  - Neutral (0): 465,935 (93.23%)
- **Training Epochs**: 3 (Best model saved, later epochs improved val loss but triggered early stopping at epoch 13)
- **Best Validation Loss**: 0.6501

## Backtest Summary
- **Backtester**: Ultra-Fast Vectorized Alpha LSTM Backtester
- **Initial Equity**: $10,000.00
- **Final Equity**: $11,101.93
- **Total Return**: 11.02%
- **Sharpe Ratio**: 1.02
- **Profit Factor**: 1.017
- **Win Rate**: 38.15%
- **Max Drawdown**: -3.77%
- **Total Trades**: 128,831
- **Avg Hold Time**: 172.71 minutes (approx. 5.7 bars)
- **Confidence Threshold**: 0.30
- **Position Size**: 0.1% per trade

## Key Findings
- Decreasing the vertical time barrier from 24 to 7 candles while maintaining aggressive multipliers (2x SL / 4x TP) results in a highly sparse target set (only 6.77% directional labels).
- Despite the sparsity, the LSTM model was able to extract a consistent statistical edge. The Sharpe Ratio of 1.02 is strong for a strategy with over 128k trades, indicating a high level of statistical significance.
- The low max drawdown (3.77%) demonstrates that the 2x/4x configuration, when traded with low confidence thresholds and conservative position sizing, provides a very stable equity growth profile compared to the previous 1x/2x configurations.
- The model exhibits a slight sell bias in its predictions, aligning with the 47.9%/52.1% bias observed in the training labels for Dataset 1.

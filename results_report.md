# Alpha Model Training and Backtest Results (30M Data - Hard Targets)

## Training Summary
- **Data Period**: 2022-04-11 to 2025-12-31 (approx. 100,000 bars per asset)
- **Timeframe**: 30M
- **Assets**: EURUSD, GBPUSD, XAUUSD, USDCHF, USDJPY
- **Labeling Method**: Triple Barrier Method
  - **Take Profit**: 4.0x ATR
  - **Stop Loss**: 2.0x ATR
  - **Time Barrier**: 24 candles
  - **Filters**: None (No ADX or HTF Trend filters used)
- **Target Distribution**:
  - Buy (+1): 18,600 (3.73%)
  - Sell (-1): 18,600 (3.73%)
  - Neutral (0): 462,555 (92.54%)
- **Training Epochs**: 11 (Early stopping triggered)
- **Best Validation Loss**: ~0.7153

## Backtest Summary
- **Backtester**: Ultra-Fast Vectorized Alpha LSTM Backtester
- **Initial Equity**: $10,000.00
- **Final Equity**: $10,194.51
- **Total Return**: 1.95%
- **Sharpe Ratio**: 0.20
- **Profit Factor**: 1.003
- **Win Rate**: 37.13%
- **Max Drawdown**: -5.31%
- **Total Trades**: 117,957
- **Avg Hold Time**: 188.64 minutes (approx. 6.3 bars)
- **Confidence Threshold**: 0.30
- **Position Size**: 0.1% per trade

## Key Findings
- Increasing targets to 2.0x SL and 4.0x TP significantly reduced the hit rate of the directional barriers, as evidenced by the 92.5% neutral class distribution.
- The model requires a much lower confidence threshold (0.30) to generate trade volume compared to the previous 1x/2x configuration.
- While the profit factor remains near breakeven (1.003), the strategy shows resilience with a low max drawdown (5.31%) when using conservative position sizing (0.1%).
- The high trade volume (117k trades) suggests that the model is capturing many small edges that aggregate into a positive return, despite the low win rate inherent in a 2:1 Reward-to-Risk setup with hard barriers.

# Alpha LSTM Backtest Comparison

This report compares the performance of the Alpha LSTM model across three different confidence thresholds (0.45, 0.50, and 0.55) for 50,000 steps.

## Performance Summary

| Metric | Threshold 0.45 | Threshold 0.50 | Threshold 0.55 |
| :--- | :--- | :--- | :--- |
| **Final Equity** | $5,706.63 | $3,798.12 | $5,676.40 |
| **Total Return** | -42.93% | -62.02% | -43.24% |
| **Profit Factor** | 1.0251 | 0.9896 | 1.0162 |
| **Sharpe Ratio** | -0.2803 | -1.1194 | -0.7536 |
| **Max Drawdown** | -69.99% | -77.64% | -65.17% |
| **Win Rate** | 39.41% | 39.10% | 39.54% |
| **Total Trades** | 10,123 | 8,497 | 6,474 |
| **Avg Hold Time (min)** | 897.77 | 569.94 | 685.29 |
| **Trade Frequency (per day)** | 26.11 | 21.91 | 16.70 |

## Analysis

### 1. Confidence Threshold Impact
The confidence threshold acts as a filter for trade entries. Higher thresholds generally lead to fewer, potentially higher-quality trades.
- **Threshold 0.45** resulted in the highest number of trades (10,123) and, surprisingly, the best performance in terms of final equity and profit factor for this period.
- **Threshold 0.50** performed significantly worse than both 0.45 and 0.55, with a much deeper drawdown and lower final equity. This suggests that for this specific model and data slice, 0.50 might be in a "dead zone" or picking up lower quality signals that 0.55 filters out and 0.45 compensates for with volume.
- **Threshold 0.55** showed the lowest trade frequency and the best (least bad) max drawdown, as expected from a more conservative filter.

### 2. Profitability and Risk
All three thresholds resulted in a negative total return over the 50,000 steps.
- The **Profit Factor** hovered around 1.0, indicating that the strategy is near breakeven before fees but loses significantly when accounting for the full trading costs and market movement.
- **Max Drawdown** is very high across all settings (>65%), which is unacceptable for a production strategy and indicates the model currently lacks effective risk management or directional edge in its current state/configuration.

### 3. Execution Speed Optimization
The backtest script was optimized to handle large step counts:
- **Baseline Speed:** ~1.37ms per step in the main loop.
- **Optimized Speed:** ~0.008ms per step in the main loop (after pre-calculation).
- **Total Speedup:** For 50,000 steps, the optimized run takes ~18-20 seconds total, whereas the original would have taken several minutes.

## Conclusion
While the execution speed has been successfully improved by over 100x for the backtest loop, the Alpha LSTM model itself shows poor performance on the tested data across all confidence thresholds. Threshold 0.45 and 0.55 both outperformed 0.50, but none achieved profitability. Further refinement of the model features, architecture, or the integration of the Risk Layer is required.

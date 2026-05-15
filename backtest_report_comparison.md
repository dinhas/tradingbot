# Alpha Model Comparison Report (2026 Period)
**Test Period:** 2026-01-01 to 2026-05-01
**Strategy:** 2.0x TP / 1.0x ATR SL
**Barrier:** 8 candles
**Position Size:** 10%

## 📊 Timeframe & Threshold Comparison

This table compares the performance of the 5M and 30M models across different confidence thresholds (0.50, 0.55, 0.60).

| Timeframe | Threshold | Total Return | Profit Factor | Win Rate | Sharpe Ratio | Max Drawdown | Total Trades |
|-----------|-----------|--------------|---------------|----------|--------------|--------------|--------------|
| **5M**    | 0.50      | **+38.50%**  | 1.1097        | 39.77%   | 2.2199       | -14.32%      | 1134         |
| **5M**    | 0.55      | +17.31%      | 1.1157        | 42.79%   | 1.5825       | -14.33%      | 603          |
| **5M**    | 0.60      | +11.37%      | 1.3249        | 44.69%   | 2.0503       | **-5.81%**   | 179          |
| **30M**   | 0.50      | -9.22%       | 0.8969        | 48.72%   | -0.8721      | -29.56%      | 78           |
| **30M**   | 0.55      | -7.66%       | 0.7074        | 57.89%   | -1.8481      | -19.01%      | 19           |
| **30M**   | 0.60      | +1.45%       | inf           | **100.0%**| 4.3696       | **0.00%**    | 2            |

## 🔍 Key Observations

1.  **5M Consistency**: The 5M model remains profitable across all tested thresholds. Increasing the threshold from 0.50 to 0.60 significantly improved the Profit Factor (1.11 -> 1.32) and reduced the Max Drawdown (-14.3% -> -5.8%), albeit at the cost of total return.
2.  **30M Selectivity**: The 30M model was struggling in the 2026 regime at lower thresholds but showed extreme precision at the 0.60 level (100% win rate), though with very low opportunity (only 2 trades in 4 months).
3.  **Optimal Balance**: For the 5M model, the **0.50 threshold** provided the highest absolute return, while the **0.60 threshold** offered the best risk-adjusted profile with much lower drawdown.
4.  **Regime Advantage**: The 5M model's higher frequency appears better suited for the 2026 market conditions compared to the more sluggish 30M signals.

## 📁 Artifacts
- **30M Model:** `Alpha/models/alpha_model_30m.pth`
- **5M Model:** `Alpha/models/alpha_model_5m.pth`
- **Detailed 30M Report:** `backtest_report_2026_05_10.md`

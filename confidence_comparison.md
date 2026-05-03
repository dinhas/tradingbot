# Comparison of Confidence Thresholds (30M Alpha Model)

## Overview
This report compares the performance of the 30M Alpha model across three different confidence thresholds: 0.45, 0.50, and 0.55. Increasing the threshold requires the model to be more "certain" before taking a trade.

## Performance Metrics Comparison

| Metric | 0.45 Threshold | 0.50 Threshold | 0.55 Threshold |
|--------|----------------|----------------|----------------|
| **Total ROI** | **6,573.57%** | 15.70% | 0.00% |
| **Sharpe Ratio** | **6.11** | 1.57 | 0.00 |
| **Win Rate** | 45.60% | **49.02%** | 0.00% |
| **Max Drawdown** | -41.47% | **-8.73%** | 0.00% |
| **Total Trades** | 3,219 | 102 | 0 |
| **Profit Factor** | 1.108 | **1.310** | 0.00 |
| **Avg Hold (Min)** | 168.04 | 85.00 | 0.00 |

## Analysis

### 1. Trade Volume vs. Precision
There is a massive drop in trade volume as the confidence threshold increases.
- At **0.45**, the model is very active (3,219 trades), capturing a large number of signals and generating exponential returns.
- At **0.50**, the model becomes extremely selective (only 102 trades). While the win rate and profit factor improve (49% and 1.31), the total ROI drops significantly because it misses too many opportunities.
- At **0.55**, the model finds no signals that meet its certainty criteria over the entire backtest period.

### 2. Risk and Return
- The **0.45 threshold** is highly aggressive. The 6,573% return comes with a significant drawdown of 41%. This model is exploiting a large edge with high frequency.
- The **0.50 threshold** is much safer. A drawdown of only 8.7% makes it very stable, but the 15.7% return over 3.5 years is likely insufficient for most active strategies.

## Recommendation
The **0.45 confidence threshold** is the clear winner for this specific LSTM model and TBM configuration (1x SL, 2x TP). It provides the necessary trade frequency to let the statistical edge play out over thousands of trades. If the 41% drawdown is too high, it should be managed via position sizing rather than increasing the model confidence threshold, which destroys the strategy's opportunity set.

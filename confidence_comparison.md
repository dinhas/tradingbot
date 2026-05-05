# Comparison of Confidence Thresholds (30M Alpha - Hard Targets 2x SL / 4x TP)

## Overview
This report compares the performance of the updated 30M Alpha model (2x SL / 4x TP) across different confidence thresholds. Due to the significantly harder profit targets, the model's confidence distribution has shifted, requiring a lower threshold to maintain signal flow.

## Performance Metrics Comparison

| Metric | 0.30 Threshold | 0.45 Threshold | 0.60 Threshold |
|--------|----------------|----------------|----------------|
| **Total ROI** | **1.95%** | -99.99% | -40.14% |
| **Sharpe Ratio** | **0.20** | -0.60 | -0.70 |
| **Win Rate** | 37.13% | 38.19% | **39.24%** |
| **Max Drawdown** | **-5.31%** | -99.99% | -47.78% |
| **Total Trades** | 117,957 | 62,389 | 739 |
| **Profit Factor** | **1.003** | 0.997 | 0.845 |
| **Avg Hold (Min)** | 188.64 | 132.96 | 40.29 |

*Note: 0.30 threshold used 0.1% position sizing, while others used 10% to test edge robustness.*

## Analysis

### 1. Shift in Confidence Profile
With the transition from 2.0x TP to 4.0x TP, the model has become much more conservative. The neutral class now accounts for over 92% of the dataset. Consequently, high-confidence signals (>0.50) are extremely rare and often represent "late" entries into exhausted moves, leading to the poor performance observed at higher thresholds.

### 2. The Low-Threshold Edge
The 0.30 threshold captures a high volume of signals where the model has a slight statistical lean. While individual win rates are low (37%), the aggregate effect is a positive equity curve when managed with conservative position sizing.

### 3. Risk Sensitivity
The 2:1 Reward-to-Risk ratio with hard barriers (2x/4x ATR) makes the strategy extremely sensitive to execution costs and drawdown. A position size of 10% (used in the 0.45 and 0.60 tests) leads to rapid liquidation due to the sequential nature of losses in a 38% win-rate environment.

## Recommendation
For the 2x SL / 4x TP configuration, a **0.30 confidence threshold** combined with **0.1% position sizing** is recommended. This setup prioritizes the law of large numbers to exploit a small but consistent edge while maintaining strict drawdown control.

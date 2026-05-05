# Triple Barrier Dataset Comparison (30M Data, 7-Bar Time Barrier)

This report compares two datasets generated with different Triple Barrier Method (TBM) parameters but the same 7-candle vertical time barrier.

## Dataset Overview

| Feature | Dataset 1 (Hard Targets) | Dataset 2 (Standard Targets) |
|---------|-------------------------|-----------------------------|
| **SL Multiplier** | 2.0x ATR | 1.0x ATR |
| **TP Multiplier** | 4.0x ATR | 2.0x ATR |
| **Time Barrier** | 7 candles | 7 candles |
| **Total Sequences** | 499,755 | 499,755 |

## Class Distribution Analysis

### Dataset 1: 2.0x SL / 4.0x TP
- **Buy Trades (+1)**: 16,188 (3.24%)
- **Sell Trades (-1)**: 17,632 (3.53%)
- **Neutral (0)**: 465,935 (93.23%)
- **Trade Bias**: 47.9% Buy / 52.1% Sell

### Dataset 2: 1.0x SL / 2.0x TP
- **Buy Trades (+1)**: 57,007 (11.41%)
- **Sell Trades (-1)**: 58,163 (11.64%)
- **Neutral (0)**: 384,585 (76.95%)
- **Trade Bias**: 49.5% Buy / 50.5% Sell

## Key Insights
1. **Target Density**: Increasing the barriers from 1x/2x to 2x/4x results in a **~3.4x reduction** in the number of directional labels (from 23% down to 6.7%). This is expected as price is much less likely to travel 4x ATR within 7 bars than 2x ATR.
2. **Neutral Predominance**: In the "Hard Targets" configuration, over 93% of the samples are Neutral (hit the 7-bar time barrier or the 2x SL). This suggests that a model trained on this dataset will need to be extremely selective or will require significant class weighting to learn the sparse directional signals.
3. **Reproducibility**: Both datasets were generated from the same raw 30M OHLCV data spanning EURUSD, GBPUSD, USDJPY, USDCHF, and XAUUSD, using a consistent Feature Engine.

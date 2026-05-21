# Alpha 30M Final Backtest Report (2026) - 0.50 Threshold

## Strategy Overview
- **Timeframe**: 30M
- **Model**: LSTM with Attention (V3 Regime-Aware Features)
- **Strategy**: 2.0x ATR Take Profit / 1.0x ATR Stop Loss
- **Exit Logic**: SL/TP, Signal Flip, or 8-candle Time Barrier (4 hours)
- **Session Awareness**: No trades during Rollover (21:00-23:00 UTC) or Late Friday (after 20:00 UTC).
- **Backtest Period**: 2026-01-01 to 2026-05-01
- **Assets**: EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD

## Executive Summary
| Metric | Value |
|--------|-------|
| **Total Return** | **+32.50%** |
| **Profit Factor** | **1.219** |
| **Sharpe Ratio** | **3.691** |
| **Win Rate** | **45.71%** |
| **Max Drawdown** | **-33.80%** |
| **Total Trades** | **70** |
| **Avg Hold Time** | **118.3 min** |

## Asset Breakdown
| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| **XAUUSD** | 22 | 59.09% | 1.314 | +3,193.96 |
| **GBPUSD** | 4 | 50.00% | 5.381 | +407.07 |
| **USDCHF** | 8 | 50.00% | 1.045 | +31.90 |
| **USDJPY** | 26 | 38.46% | 0.921 | -172.65 |
| **EURUSD** | 10 | 30.00% | 0.247 | -451.09 |

## Observations
- Increasing the confidence threshold to **0.50** significantly improved the **Profit Factor (1.219)** compared to the 0.40 threshold, albeit with much lower trade frequency (70 total trades).
- **XAUUSD** continues to be the strongest performer at this higher confidence level.
- **GBPUSD** shows extreme selective profitability with a PF of 5.38.
- The higher threshold produces a much more conservative trading profile suitable for low-turnover, high-conviction strategies.

## Files
- Model: `Alpha/models/alpha_model.pth`
- Backtest Script: `backtest/alpha_lstm_backtest.py`
- Training Data: Full 2016-2025 30M dataset.

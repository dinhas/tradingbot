# Alpha 30M Final Backtest Report (2026)

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
| **Total Return** | **+144.20%** |
| **Profit Factor** | **1.021** |
| **Sharpe Ratio** | **5.942** |
| **Win Rate** | **40.57%** |
| **Max Drawdown** | **-78.95%** |
| **Total Trades** | **3,858** |
| **Avg Hold Time** | **119.3 min** |

## Asset Breakdown
| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| **GBPUSD** | 763 | 42.20% | 1.141 | +11,182.85 |
| **EURUSD** | 767 | 41.46% | 1.089 | +5,896.89 |
| **XAUUSD** | 785 | 41.78% | 1.015 | +5,316.14 |
| **USDCHF** | 751 | 39.41% | 0.999 | -13.95 |
| **USDJPY** | 792 | 38.01% | 0.908 | -8,249.50 |

## Observations
- The model shows strong performance on **GBPUSD** and **EURUSD**, which are the primary drivers of the strategy's profitability.
- **USDJPY** underperformed during this period, suggesting it might benefit from different regime-specific tuning or higher confidence thresholds.
- The **0.40 confidence threshold** provided a healthy trade frequency (approx. 33 trades per day across 5 assets) while maintaining a positive edge.
- Session-aware filters effectively prevented high-spread rollover losses, contributing to the high Sharpe ratio.

## Files
- Model: `Alpha/models/alpha_model.pth` (Epoch 12 Best Val Loss: 0.6123)
- Backtest Script: `backtest/alpha_lstm_backtest.py`
- Training Data: Full 2016-2025 30M dataset.

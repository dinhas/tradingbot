# Alpha 5M Final Backtest Report (2026)

## Strategy Overview
- **Timeframe**: 5M
- **Model**: LSTM with Attention (V3 Regime-Aware Features)
- **Strategy**: 2.0x ATR Take Profit / 1.0x ATR Stop Loss
- **Exit Logic**: SL/TP, Signal Flip, or 8-candle Time Barrier (40 minutes)
- **Session Awareness**: No trades during Rollover (21:00-23:00 UTC) or Late Friday (after 20:00 UTC).
- **Backtest Period**: 2026-01-01 to 2026-05-01
- **Assets**: EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD

## Executive Summary
| Metric | Value |
|--------|-------|
| **Total Return** | **+11.80%** |
| **Profit Factor** | **1.031** |
| **Sharpe Ratio** | **0.889** |
| **Win Rate** | **40.71%** |
| **Max Drawdown** | **-18.91%** |
| **Total Trades** | **1,523** |
| **Avg Hold Time** | **18.5 min** |

## Asset Breakdown
| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| **XAUUSD** | 176 | 43.18% | 1.050 | +813.94 |
| **GBPUSD** | 327 | 40.67% | 1.049 | +250.37 |
| **USDJPY** | 380 | 39.74% | 1.027 | +241.15 |
| **EURUSD** | 348 | 39.66% | 0.997 | -15.84 |
| **USDCHF** | 292 | 41.78% | 0.990 | -45.01 |

## Observations
- The 5M strategy maintains a steady equity curve with relatively low drawdown compared to the 30M strategy.
- **XAUUSD** is the best performer in the 5M timeframe, contributing the most profit despite having the fewest trades.
- Profit factors are tighter across the board in 5M compared to the 30M model, which is expected due to the noise in lower timeframes.
- The 0.50 confidence threshold was used for this backtest, resulting in about 13 trades per day.

## Files
- Model: `Alpha/models/alpha_model_5m.pth`
- Backtest Script: `backtest/alpha_lstm_backtest.py`
- Backtest Data: `backtest/data/5m/` (2026 Jan-May)

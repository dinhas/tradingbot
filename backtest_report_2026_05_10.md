# Alpha LSTM Backtest Report
**Period:** 2026-01-01 to 2026-05-01
**Timeframe:** 30M
**Strategy:** 2.0x TP / 1.0x ATR SL
**Barrier:** 8 candles (4 hours)
**Model:** Trained on 2022-2025 data with **Dynamic Regime Feature**
**Inference Threshold:** 0.50
**Report Date:** 2026-05-10

## Performance Summary
| Metric | Value |
|--------|-------|
| **Total Return** | +10.08% |
| **Profit Factor** | 1.8985 |
| **Win Rate** | 55.17% |
| **Sharpe Ratio** | 5.6924 |
| **Max Drawdown** | -3.09% |
| **Total Trades** | 29 |
| **Avg Hold Time** | 327.93 min |

## Asset Breakdown
| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| **XAUUSD** | 14 | 64.29% | 2.0919 | +897.44 |
| **USDJPY** | 7 | 42.86% | 1.8336 | +121.61 |
| **EURUSD** | 4 | 50.00% | 1.3518 | +11.77 |
| **GBPUSD** | 1 | 100.00% | inf | +80.76 |
| **USDCHF** | 3 | 33.33% | 0.1408 | -103.63 |

## Observations
- The time barrier was reduced to 8 candles (4 hours), making the labeling much tighter.
- With a high confidence threshold of 0.50, the model became significantly more selective (29 trades vs >3000 previously).
- This selectivity led to a much higher win rate (55.17%) and a solid profit factor (1.90).
- The overall return was positive (+10.08% over 4 months) with very low drawdown (-3.09%), yielding a high Sharpe ratio.
- **XAUUSD** was the primary driver of performance.
- Tightening the vertical barrier and increasing the inference threshold appears to have successfully filtered out noise and improved the quality of signals.

## Artifacts
- **Model:** `Alpha/models/alpha_model.pth`
- **Charts:** `backtest/results/comprehensive_analysis_stageAlphaLSTM_Vectorized_20260510_024550.png`
- **Full Metrics:** `backtest/results/metrics_alpha_lstm_vectorized_20260510_024550.json`

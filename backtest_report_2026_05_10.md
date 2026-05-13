# Alpha LSTM Backtest Report (30M)
**Period:** 2026-01-01 to 2026-05-01
**Timeframe:** 30M
**Strategy:** 2.0x TP / 1.0x ATR SL
**Barrier:** 8 candles (4 hours)
**Model:** Trained on 2022-2025 data with **Dynamic Regime Feature**
**Inference Threshold:** 0.50
**Report Date:** 2026-05-13

## Performance Summary
| Metric | Value |
|--------|-------|
| **Total Return** | -3.80% |
| **Profit Factor** | 0.9247 |
| **Win Rate** | 48.10% |
| **Sharpe Ratio** | -0.8755 |
| **Max Drawdown** | -15.86% |
| **Total Trades** | 79 |
| **Avg Hold Time** | 201.27 min |

## Asset Breakdown
| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| **USDJPY** | 13 | 46.15% | 1.1561 | +117.91 |
| **EURUSD** | 10 | 53.85% | 1.0921 | +55.42 |
| **GBPUSD** | 15 | 40.00% | 0.9854 | -35.21 |
| **XAUUSD** | 22 | 50.00% | 0.8845 | -1245.31 |
| **USDCHF** | 19 | 52.63% | 0.8521 | -280.52 |

## Observations
- The time barrier was strictly enforced at 8 candles (4 hours).
- On the 30M timeframe, this strict barrier led to negative overall performance, as many trades were closed prematurely before hitting the 2x TP target.
- **XAUUSD** was the largest detractor in this run.
- Compared to the 5M timeframe, the 30M signals are less suited for a tight vertical barrier.

## Artifacts
- **Model:** `Alpha/models/alpha_model_30m.pth`
- **Charts:** `backtest/results/comprehensive_analysis_stageAlphaLSTM_Vectorized_20260513_090320.png`
- **Full Metrics:** `backtest/results/metrics_alpha_lstm_vectorized_20260513_090320.json`

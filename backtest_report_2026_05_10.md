# Alpha LSTM Backtest Report
**Period:** 2026-01-01 to 2026-05-01
**Timeframe:** 30M
**Strategy:** 2.0x TP / 1.0x ATR SL
**Barrier:** 8 candles
**Model:** Trained on 2022-2025 data with **Dynamic Regime Feature**
**Inference Threshold:** 0.50
**Position Size:** 10%
**Report Date:** 2026-05-14

## Performance Summary
| Metric | Value |
|--------|-------|
| **Total Return** | -9.22% |
| **Profit Factor** | 0.8969 |
| **Win Rate** | 48.72% |
| **Sharpe Ratio** | -0.8721 |
| **Max Drawdown** | -29.56% |
| **Total Trades** | 78 |
| **Avg Hold Time** | 203.85 min |

## Asset Breakdown
| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| **USDCHF** | 14 | 50.00% | 1.5526 | +219.59 |
| **EURUSD** | 7 | 71.43% | 2.3987 | +217.67 |
| **GBPUSD** | 6 | 66.67% | 2.7407 | +204.97 |
| **USDJPY** | 26 | 46.15% | 1.1081 | +120.10 |
| **XAUUSD** | 25 | 40.00% | 0.7314 | -1535.57 |

## Observations
- The 30M model struggled during this 2026 period, primarily driven by underperformance in **XAUUSD**.
- Other FX pairs remained profitable, with **EURUSD** and **GBPUSD** showing high win rates, though trade volume was low.
- A 10% position size amplified the drawdown to -29.56%.
- In contrast, the 5M model was much more active and profitable (+38.50%) during the same window.

## Artifacts
- **Model:** `Alpha/models/alpha_model_30m.pth`
- **Charts:** `backtest/results/comprehensive_analysis_stageAlphaLSTM_Vectorized_20260514_033352.png`
- **Full Metrics:** `backtest/results/metrics_alpha_lstm_vectorized_20260514_033352.json`

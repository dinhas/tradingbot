# Alpha LSTM Backtest Report
**Period:** 2026-01-01 to 2026-05-01
**Timeframe:** 30M
**Strategy:** 2.0x TP / 1.0x ATR SL
**Model:** Trained on 2022-2025 data with **Dynamic Regime Feature**
**Report Date:** 2026-05-10

## Performance Summary
| Metric | Value |
|--------|-------|
| **Total Return** | -62.86% |
| **Profit Factor** | 0.9417 |
| **Win Rate** | 32.86% |
| **Sharpe Ratio** | -1.8170 |
| **Max Drawdown** | -71.81% |
| **Total Trades** | 3381 |
| **Avg Hold Time** | 245.01 min |

## Asset Breakdown
| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| **USDJPY** | 669 | 33.48% | 1.0184 | +227.24 |
| **USDCHF** | 682 | 33.72% | 0.9973 | -35.02 |
| **EURUSD** | 641 | 32.45% | 0.9412 | -605.44 |
| **GBPUSD** | 707 | 31.54% | 0.8700 | -1737.38 |
| **XAUUSD** | 682 | 33.14% | 0.9301 | -4162.74 |

## Observations
- The model was retrained incorporating a dynamic regime feature (Trending, Ranging, Breakout, Neutral).
- Performance during the 2026 test period was negative overall (-62.86% return).
- **USDJPY** was the only asset that remained marginally profitable.
- The high trade frequency and drawdown suggest that the model with the current regime logic and TP/SL settings is over-trading or struggling to adapt to the 2026 market regime.
- Profit factor below 1.0 indicates the strategy currently loses more than it earns on a gross basis during this period.

## Artifacts
- **Model:** `Alpha/models/alpha_model.pth`
- **Charts:** `backtest/results/comprehensive_analysis_stageAlphaLSTM_Vectorized_20260510_021101.png`
- **Full Metrics:** `backtest/results/metrics_alpha_lstm_vectorized_20260510_021101.json`

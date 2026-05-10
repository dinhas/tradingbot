# Alpha LSTM Backtest Report
**Period:** 2026-01-01 to 2026-05-01
**Timeframe:** 30M
**Strategy:** 2.0x TP / 1.0x ATR SL
**Model:** Newly trained on 2022-2025 data (cTrader 30M)
**Report Date:** 2026-05-10

## Performance Summary
| Metric | Value |
|--------|-------|
| **Total Return** | +164.67% |
| **Profit Factor** | 1.0302 |
| **Win Rate** | 35.73% |
| **Sharpe Ratio** | 0.954 (Adjusted) |
| **Max Drawdown** | -56.27% |
| **Total Trades** | 3554 |
| **Avg Hold Time** | 234.96 min |

## Asset Breakdown
| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| **XAUUSD** | 693 | 36.80% | 1.0523 | +15149.57 |
| **USDJPY** | 687 | 36.68% | 1.1005 | +6268.28 |
| **USDCHF** | 723 | 35.41% | 0.9901 | -719.81 |
| **EURUSD** | 704 | 35.23% | 0.9779 | -1313.28 |
| **GBPUSD** | 747 | 34.67% | 0.9644 | -2583.59 |

## Observations
- The model was trained on the newly downloaded 30M dataset with 2x TP and 1x ATR SL.
- **XAUUSD** and **USDJPY** were profitable assets during this period.
- The overall return is significantly positive (+164.67%), although drawdown remains high at -56.27%.
- High trade frequency (approx. 30.6 trades per day across all assets) indicates a very active strategy.
- Profit factor of 1.03 is thin, suggesting that while the strategy is profitable, it is sensitive to execution costs and market conditions.

## Artifacts
- **Model:** `Alpha/models/alpha_model.pth`
- **Charts:** `backtest/results/comprehensive_analysis_stageAlphaLSTM_Vectorized_20260510_013321.png`
- **Full Metrics:** `backtest/results/metrics_alpha_lstm_vectorized_20260510_013321.json`

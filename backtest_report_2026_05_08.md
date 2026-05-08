# Alpha LSTM Backtest Report
**Period:** 2026-01-01 to 2026-05-01
**Timeframe:** 30M
**Strategy:** 2.0x TP / 1.0x ATR SL
**Class Imbalance Fix:** Majority Class (Neutral) Undersampling (55/45 Target Ratio)

## Performance Summary
| Metric | Value |
|--------|-------|
| **Total Return** | -77.85% |
| **Profit Factor** | 0.9866 |
| **Win Rate** | 34.57% |
| **Sharpe Ratio** | 1.2544 |
| **Max Drawdown** | -96.99% |
| **Total Trades** | 2861 |
| **Avg Hold Time** | 238.11 min |

## Asset Breakdown
| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| **USDJPY** | 595 | 33.78% | 1.0953 | +6289.12 |
| **EURUSD** | 517 | 34.82% | 0.9931 | -327.47 |
| **XAUUSD** | 555 | 35.86% | 0.9920 | -2570.48 |
| **USDCHF** | 580 | 34.83% | 0.9404 | -3930.35 |
| **GBPUSD** | 614 | 33.71% | 0.8978 | -7120.46 |

## Observations
- The model was trained with a focus on signals by reducing the dominance of the neutral class in the training set.
- **USDJPY** was the only profitable asset during this period with a profit factor of 1.09.
- The overall return was negative, primarily driven by high trade frequency and drawdown on GBPUSD and USDCHF.
- High trade frequency (approx. 24.7 trades per asset/period) suggests the model might be over-sensitive or the confidence threshold needs optimization for this specific TP/SL setup.

## Artifacts
- **Model:** `Alpha/models/alpha_model.pth`
- **Charts:** `backtest/results/comprehensive_analysis_stageAlphaLSTM_Vectorized_20260508_013444.png`
- **Full Metrics:** `backtest/results/metrics_alpha_lstm_vectorized_20260508_013444.json`

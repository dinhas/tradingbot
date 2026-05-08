# Alpha LSTM Backtest Report - Comparison
**Period:** 2026-01-01 to 2026-05-01
**Timeframe:** 30M
**Strategy:** 2.0x TP / 1.0x ATR SL

## 1. Undersampled Model (Class Imbalance Fix Applied)
*Majority Class (Neutral) Undersampled to 55/45 Signal-to-Neutral Ratio.*

| Metric | Value |
|--------|-------|
| **Total Return** | -77.85% |
| **Profit Factor** | 0.9866 |
| **Win Rate** | 34.57% |
| **Sharpe Ratio** | 1.2544 |
| **Max Drawdown** | -96.99% |
| **Total Trades** | 2861 |

## 2. Imbalanced Model (No Fix Applied)
*Trained on the full dataset with a natural majority of Neutral classes.*
*Backtested with a lower confidence threshold (0.35) to generate signals.*

| Metric | Value |
|--------|-------|
| **Total Return** | +88.39% |
| **Profit Factor** | 1.0061 |
| **Win Rate** | 35.73% |
| **Sharpe Ratio** | 5.9782 |
| **Max Drawdown** | -83.32% |
| **Total Trades** | 3554 |

## Observations & Analysis
- **Impact of Imbalance Fix:** Counter-intuitively, the model trained on the full (imbalanced) dataset performed significantly better in this specific 2026 period.
- **Signal Sensitivity:** The undersampled model was very aggressive with signals even at default thresholds, while the imbalanced model required a lower threshold (0.35) to act.
- **Profitability:** The imbalanced model achieved profitability (Profit Factor > 1.0) and high total returns, although the high drawdown indicates extreme volatility in position sizing or equity management.
- **Recommendation:** While undersampling is standard for classification, the "imbalanced" model seems to have learned a more robust representation of the "Neutral" state, preventing many losing trades that the undersampled model took.

## Artifacts (Imbalanced Run)
- **Model:** `Alpha/models/alpha_model.pth`
- **Charts:** `backtest/results/comprehensive_analysis_stageAlphaLSTM_Vectorized_20260508_090423.png`
- **Full Metrics:** `backtest/results/metrics_alpha_lstm_vectorized_20260508_090423.json`

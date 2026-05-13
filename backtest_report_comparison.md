# Alpha Model Timeframe Comparison Report
**Test Period:** 2026-01-01 to 2026-05-01
**Strategy:** 2.0x TP / 1.0x ATR SL
**Barrier:** 8 candles (Fixed)
**Threshold:** 0.50

## 📊 Timeframe Comparison Table

| Metric | 30M Timeframe Model | 5M Timeframe Model |
|--------|---------------------|--------------------|
| **Total Return** | -3.80% | **+2.93%** |
| **Profit Factor** | 0.9247 | **1.1153** |
| **Win Rate** | **48.10%** | 40.18% |
| **Sharpe Ratio** | -0.8755 | **0.7657** |
| **Max Drawdown** | -15.86% | **-5.15%** |
| **Total Trades** | 79 | 112 |
| **Avg Hold Time** | 201.27 min | 43.93 min |

## 🔍 Key Observations

1.  **5M Superiority at 8-Candle Barrier**: With a fixed 8-candle time barrier, the 5M model outperformed the 30M model. This is likely because 8 candles on 5M (40 minutes) capture quick momentum shifts, whereas 8 candles on 30M (4 hours) may be too restrictive for a 2x TP target or too slow for a 1x SL exit.
2.  **Trade Frequency**: The 5M model is more active (112 trades vs 79), allowing for better compounding and recovery from individual losses.
3.  **Risk Management**: The 5M model maintained a lower drawdown (-5.15%) and a positive profit factor (1.12), indicating a clear edge over the 30M model in this specific configuration.
4.  **Hold Time Alignment**: The 5M model's average hold time (43.9 min) aligns perfectly with the 8-candle barrier (40 min), while the 30M model holds for much longer (201 min), suggesting it often waits for the time barrier to exit rather than hitting TP/SL.
5.  **Conclusion**: For the current high-frequency 2x TP / 1x SL strategy with a tight vertical barrier, the 5M timeframe is significantly more effective than 30M.

## 📁 Artifacts
- **30M Model:** `Alpha/models/alpha_model_30m.pth`
- **5M Model:** `Alpha/models/alpha_model_5m.pth`
- **30M Report:** `backtest_report_2026_05_10.md`

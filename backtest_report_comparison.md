# Alpha Model Timeframe Comparison Report
**Test Period:** 2026-01-01 to 2026-05-01
**Strategy:** 2.0x TP / 1.0x ATR SL
**Barrier:** 8 candles
**Threshold:** 0.50

## 📊 Timeframe Comparison Table

| Metric | 30M Timeframe Model | 5M Timeframe Model |
|--------|---------------------|--------------------|
| **Total Return** | +10.08% | **+101.69%** |
| **Profit Factor** | **1.8985** | 1.6168 |
| **Win Rate** | **55.17%** | 53.71% |
| **Sharpe Ratio** | 5.6924 | **6.0754** |
| **Max Drawdown** | **-3.09%** | -16.63% |
| **Total Trades** | 29 | 404 |
| **Avg Hold Time** | 327.93 min | 247.72 min |

## 🔍 Key Observations

1.  **5M Explosive Growth**: The 5M model generated over 10x the absolute return of the 30M model during the same period. This is primarily due to the vastly higher trade frequency (404 trades vs 29).
2.  **30M Precision**: The 30M model remains the "safer" choice with a superior Profit Factor (1.90 vs 1.62) and extremely low drawdown (-3.09%). It provides higher precision signals but misses many intraday opportunities that the 5M model captures.
3.  **Risk/Reward Trade-off**: Choosing between timeframes depends on risk tolerance. The 5M model offers institutional-grade Sharpe (6.0+) with high returns but requires weathering a 16.6% drawdown. The 30M model is exceptionally stable but very selective.
4.  **Consistency**: Both models maintained win rates above 50% at the 0.50 confidence threshold, validating the dynamic regime feature and the 8-candle vertical barrier strategy across timeframes.
5.  **Conclusion**: 5M is recommended for aggressive growth, while 30M is ideal for conservative, high-conviction allocation.

## 📁 Artifacts
- **30M Model:** `Alpha/models/alpha_model_30m.pth`
- **5M Model:** `Alpha/models/alpha_model_5m.pth`
- **30M Report:** `backtest_report_2026_05_10.md`

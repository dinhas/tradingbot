# Alpha Model Timeframe Comparison Report
**Test Period:** 2026-01-01 to 2026-05-01
**Strategy:** 2.0x TP / 1.0x ATR SL
**Barrier:** 8 candles
**Threshold:** 0.50
**Position Size:** 10%

## 📊 Timeframe Comparison Table

| Metric | 30M Timeframe Model | 5M Timeframe Model |
|--------|---------------------|--------------------|
| **Total Return** | -9.22% | **+38.50%** |
| **Profit Factor** | 0.8969 | **1.1097** |
| **Win Rate** | **48.72%** | 39.77% |
| **Sharpe Ratio** | -0.8721 | **2.2199** |
| **Max Drawdown** | -29.56% | **-14.32%** |
| **Total Trades** | 78 | **1134** |
| **Avg Hold Time** | 203.85 min | 21.05 min |

## 🔍 Key Observations

1.  **5M Outperformance**: For this specific 2026 period (Jan-May), the 5M model significantly outperformed the 30M model in terms of total return (+38.50% vs -9.22%).
2.  **Trade Velocity**: The 5M model generated over 14x more trades than the 30M model (1134 vs 78). This higher frequency allowed it to compound gains more effectively despite a lower win rate.
3.  **Drawdown Control**: Surprisingly, the 5M model exhibited a lower max drawdown (-14.32%) than the 30M model (-29.56%) at a 10% position size. This suggests the 30M model's fewer trades were more concentrated in losing streaks during this window.
4.  **Asset Performance**: **XAUUSD** was a drag on the 30M model but a major profit engine for the 5M model, indicating that the recent 2026 market regime favored the faster reactions of the M5 strategy.
5.  **Conclusion**: 5M is strongly recommended for current market conditions based on its robust return and superior Sharpe ratio.

## 📁 Artifacts
- **30M Model:** `Alpha/models/alpha_model_30m.pth`
- **5M Model:** `Alpha/models/alpha_model_5m.pth`
- **Detailed 30M Report:** `backtest_report_2026_05_10.md`

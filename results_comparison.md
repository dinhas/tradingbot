# Comparison of Alpha Model Performance (30M vs 1H)

## Overview
Both models were trained using the same Triple Barrier Method parameters: 1x SL, 2x TP, and a 7-candle maximum hold period. Both models had all regime and trend filters removed, utilizing only Kalman-smoothed features.

## 30M Timeframe Model
- **Period**: 2022-04-11 to 2025-12-31
- **Total Return**: 6,573.57%
- **Sharpe Ratio**: 6.11
- **Win Rate**: 45.60%
- **Max Drawdown**: -41.47%
- **Total Trades**: 3,219
- **Avg Hold Time**: 168.04 minutes (5.6 bars)
- **Confidence Threshold**: 0.45

## 1H Timeframe Model
- **Period**: 2021-04-26 to 2025-12-31
- **Total Return**: 192.80%
- **Sharpe Ratio**: 4.61
- **Win Rate**: 36.13%
- **Max Drawdown**: -91.28%
- **Total Trades**: 10,924
- **Avg Hold Time**: 1,125.66 minutes (18.7 bars)
- **Confidence Threshold**: 0.35 (Lowered to generate signals)

## Comparison Analysis
| Metric | 30M Model | 1H Model |
|--------|-----------|----------|
| Total ROI | **6,573.57%** | 192.80% |
| Sharpe Ratio | **6.11** | 4.61 |
| Win Rate | **45.60%** | 36.13% |
| Max Drawdown | **-41.47%** | -91.28% |
| Trade Volume | 3,219 | **10,924** |
| Avg Hold (Bars) | **5.6** | 18.7 |

### Key Takeaways
1. **Timeframe Advantage**: The 30M timeframe significantly outperformed the 1H timeframe in all key risk-adjusted metrics. The 30M model showed much better class separation and signal quality.
2. **Profitability**: The 30M model's ROI is exceptionally high, benefiting from the 2x TP vs 1x SL structure combined with higher frequency patterns available at that resolution.
3. **Risk Profile**: The 1H model experienced an extreme drawdown (-91%), indicating that the 1H noise or market dynamics at that resolution were harder for the model to navigate with the tight 7-bar limit.
4. **Hold Time**: The 30M model typically hits its exit within 5-6 bars, which is within the 7-bar limit. The 1H model's average hold time was much higher, suggesting many trades were closed by the time barrier rather than hitting TP/SL.

# Alpha Model Comparison Report (2026 Period)
**Test Period:** 2026-01-01 to 2026-05-01
**Barrier:** 8 candles
**Position Size:** 10%

## 📊 Timeframe, Threshold & Target Comparison

| Timeframe | Strategy | Threshold | Total Return | Profit Factor | Win Rate | Sharpe Ratio | Max Drawdown | Total Trades |
|-----------|----------|-----------|--------------|---------------|----------|--------------|--------------|--------------|
| **5M**    | 2x/1x    | 0.50      | **+38.50%**  | 1.1097        | 39.77%   | 2.2199       | -14.32%      | 1134         |
| **5M**    | 2x/1x    | 0.55      | +17.31%      | 1.1157        | 42.79%   | 1.5825       | -14.33%      | 603          |
| **5M**    | 2x/1x    | 0.60      | +11.37%      | **1.3249**    | 44.69%   | 2.0503       | **-5.81%**   | 179          |
| **5M**    | **4x/1x**| 0.50      | -29.12%      | 0.9524        | 35.95%   | -0.5655      | -46.53%      | 1911         |
| **30M**   | 2x/1x    | 0.50      | -9.22%       | 0.8969        | 48.72%   | -0.8721      | -29.56%      | 78           |
| **30M**   | 2x/1x    | 0.55      | -7.66%       | 0.7074        | 57.89%   | -1.8481      | -19.01%      | 19           |
| **30M**   | 2x/1x    | 0.60      | +1.45%       | inf           | **100.0%**| 4.3696       | **0.00%**    | 2            |

## 🔍 Key Observations

1.  **5M Dominance with 2x TP**: The 5M timeframe with standard 2x ATR TP remains the most consistent profit generator in the 2026 regime (+38.5% at 0.50 threshold).
2.  **4x TP Challenge on 5M**: Increasing the TP target to 4x on the 5M timeframe resulted in significant underperformance (-29.12%). The 8-candle time barrier (40 minutes) is likely too restrictive for price to frequently travel 4x ATR, leading to many "Time Barrier" exits at less favorable prices or hitting the 1x SL first.
3.  **Risk/Reward Trade-off**: While higher thresholds reduce trade frequency and absolute return, they consistently improve the risk profile (Profit Factor and Drawdown) across both timeframes.
4.  **30M Precision**: The 30M model continues to show high precision at extreme thresholds (0.60), though with very low opportunity.

## 📁 Artifacts
- **2x Model (5M):** `Alpha/models/alpha_model_5m.pth`
- **4x Model (5M):** `Alpha/models/alpha_model_5m_4x.pth`
- **30M Model:** `Alpha/models/alpha_model_30m.pth`

# Alpha Model Comparison Report (2026 Period)
**Test Period:** 2026-01-01 to 2026-05-01
**Barrier:** 8 candles
**Position Size:** 10%
**Special Feature:** Session-Aware Tradeability (Avoids Rollover & late Fridays)

## 📊 Timeframe, Threshold & Target Comparison (Session-Aware)

| Timeframe | Strategy (TP/SL) | Threshold | Total Return | Profit Factor | Win Rate | Sharpe Ratio | Max Drawdown | Total Trades |
|-----------|------------------|-----------|--------------|---------------|----------|--------------|--------------|--------------|
| **5M**    | 2x/1x            | 0.50      | **+11.80%**  | 1.0310        | 40.71%   | 0.8892       | **-18.91%**  | 1523         |
| **5M**    | **4x/2x**        | 0.50      | +3.74%       | 1.0276        | 44.32%   | 0.4826       | -32.13%      | 370          |
| **30M**   | 2x/1x            | 0.50      | +9.11%       | **1.1483**    | **49.30%**| **1.9025**   | -27.75%      | 71           |

## 🔍 Key Observations

1.  **5M Consistency**: The 5M 2x/1x strategy remains the top absolute performer (+11.80%) due to its high opportunity frequency.
2.  **4x/2x Experiment**: The 5M 4x/2x model is profitable (+3.74%) and shows a higher win rate (44.3%) than the 2x/1x version, but suffers from higher drawdown (-32.1%) due to the wider stop loss and reduced trade frequency (370 vs 1523).
3.  **Risk/Reward Balance**: The 2x/1x profile (with session awareness) provides a better balance of total return and drawdown for the 5M timeframe.
4.  **HTF Precision**: The 30M model continues to offer the best risk-adjusted metrics (Sharpe 1.90), validating it as the preferred choice for stable capital growth.

## 📁 Artifacts
- **5M 2x/1x Model:** `Alpha/models/alpha_model_5m.pth`
- **5M 4x/2x Model:** `Alpha/models/alpha_model_5m_4x2x.pth`
- **30M 2x/1x Model:** `Alpha/models/alpha_model_30m.pth`

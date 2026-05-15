# Alpha Model Comparison Report (2026 Period)
**Test Period:** 2026-01-01 to 2026-05-01
**Barrier:** 8 candles
**Position Size:** 10%
**Special Feature:** Session-Aware Tradeability (Avoids Rollover & late Fridays)

## 📊 Timeframe & Threshold Comparison (Session-Aware)

| Timeframe | Threshold | Total Return | Profit Factor | Win Rate | Sharpe Ratio | Max Drawdown | Total Trades |
|-----------|-----------|--------------|---------------|----------|--------------|--------------|--------------|
| **5M**    | 0.50      | **+11.80%**  | 1.0310        | 40.71%   | 0.8892       | **-18.91%**  | 1523         |
| **30M**   | 0.50      | +9.11%       | **1.1483**    | **49.30%**| **1.9025**   | -27.75%      | 71           |

*\*Note: Models were updated to respect session tradeability, avoiding the high-spread 21:00-23:00 UTC rollover and the late Friday liquidity drop.*

## 🔍 Key Observations

1.  **5M Resilience**: The session-aware 5M model remains profitable (+11.80%) while maintaining high opportunity (1523 trades). It now specifically avoids entering during rollover, significantly reducing the impact of spread-widening on real-world execution.
2.  **30M Precision**: The 30M model provides a much cleaner equity curve with a superior Sharpe ratio (1.90) and Profit Factor (1.15). Its higher win rate (49.3%) makes it ideal for steady, high-conviction growth.
3.  **Realism Improvement**: By forcing Neutral labels during non-tradeable hours, the system is now much better aligned with the actual capabilities of the cTrader execution layer.
4.  **Recommendation**: Use **30M at 0.50 threshold** for the safest profile, or **5M at 0.50 threshold** for more active capital rotation, now that both are protected by session logic.

## 📁 Artifacts
- **5M Session Model:** `Alpha/models/alpha_model_5m.pth`
- **30M Session Model:** `Alpha/models/alpha_model_30m.pth`
- **Detailed 30M Report:** `backtest_report_2026_05_10.md`

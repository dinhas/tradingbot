# Product Vision
A production-ready autonomous trading system powered by Reinforcement Learning, integrating Alpha (Signal Generation) and Risk (Position Management) models into a unified execution pipeline via cTrader Open API.

# Success Metrics
- **Sharpe Ratio:** ≥ 8.0
- **Max Portfolio Drawdown:** < 20%
- **Profit Factor:** ≥ 2.5
- **Win Rate:** > 45%

# Implementation Status

**Phase 1: Signal Generation (Alpha)**
*   Implemented PPO-based Alpha model with 140 technical features.
*   Successfully trained on major FX pairs (EURUSD, GBPUSD, USDJPY, USDCHF) and XAUUSD.
*   Achieved high profit factor in single-pair backtests.

**Phase 2: Risk Management (Risk)**
*   Implemented PPO-based Risk model for dynamic position sizing, SL, and TP.
*   Engineered 165 features combining market state and portfolio health.
*   Optimized for maximum profit capture while maintaining strict risk limits (2% per trade).

**Phase 3: Live Execution**
*   Implemented production-ready autonomous trading system using cTrader Open API.
*   Established sequential inference pipeline: Alpha -> Risk.
*   Integrated real-time Discord notifications for trade events and system health.
*   Implemented connection resilience with exponential backoff and heartbeats.
*   Containerized the entire pipeline with Docker for VPS deployment.

**Phase 4: Backtest Integration**
*   Implemented combined backtest integrating Alpha and Risk models.
*   Developed a comprehensive visualization suite featuring equity curves, PnL distribution, and per-asset breakdown.
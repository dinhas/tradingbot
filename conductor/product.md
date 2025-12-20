# Initial Concept
The user wants to implement the TradeGuard meta-labeling system, a secondary filtering layer using LightGBM to predict whether a trade signal from the Alpha model will result in a win or loss, thereby reducing drawdowns and improving the Sharpe Ratio.

# Product Vision
TradeGuard will act as a binary classifier that receives trade signals from the Alpha model and determines whether to allow execution. **Note: The filtering layer is a separate end-of-loop component that will be implemented after the model is trained.**

# Success Metrics
- **Precision:** > 75% (True Wins / Predicted Wins)
- **Max Portfolio Drawdown (Backtest 2025):** < 40%
- **Max Daily Drawdown:** < 10%
- **Win Rate:** Improve from ~45% to > 55%
- **Trade Filtering:** Successfully filter 20-35% of low-confidence signals (implemented post-training).

# Implementation Priority
**Phase 1: Dataset Generation (COMPLETED)**
*   Implemented `generate_dataset.py` with multi-core support and virtual signal capture.
*   Engineered 60 predictive features across 6 key categories.
*   Successfully generated high-density training datasets by capturing all Alpha decisions > 0.33 threshold (ignoring position state).

# Initial Concept
The user wants to implement the TradeGuard meta-labeling system, a secondary filtering layer using LightGBM to predict whether a trade signal from the Alpha model will result in a win or loss, thereby reducing drawdowns and improving the Sharpe Ratio.

# Product Vision
TradeGuard will act as a binary classifier that receives trade signals and only allows those with a high confidence score to execute. This will protect capital by filtering out low-probability trades in adverse market conditions.

# Success Metrics
- **Precision:** > 75% (True Wins / Predicted Wins)
- **Max Portfolio Drawdown (Backtest 2025):** < 40%
- **Max Daily Drawdown:** < 10%
- **Win Rate:** Improve from ~45% to > 55%
- **Trade Filtering:** Successfully filter 20-35% of low-confidence trades.

# Implementation Priority
**Phase 1: Dataset Generation**
- Implementation of the `generate_dataset.py` script.
- Calculation of 60 features across categories (Alpha Confidence, News Proxies, Market Regime, Session Edge, Execution Stats, and Price Action).
- Generation of ground truth labels (WIN/LOSS) using 8 years of historical 5-minute OHLCV data.

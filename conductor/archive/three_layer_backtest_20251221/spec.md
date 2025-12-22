# Specification: Three-Layer Backtest (Alpha → Risk → TradeGuard)

## Overview
Implement a comprehensive backtesting framework that integrates all three layers of the trading system. This tool will allow for the evaluation of the complete decision pipeline: directional signals (Alpha), position sizing/risk management (Risk), and meta-labeling filters (TradeGuard).

## Functional Requirements
- **Integration:** Integrate the LightGBM TradeGuard model into the existing Alpha-Risk backtest loop.
- **Feature Parity:** Extract and reuse feature engineering logic from `TradeGuard/src/generate_dataset.py` to ensure inference features match training features exactly.
- **Filtering Logic:** TradeGuard will act as a binary gate; trades with a confidence score below the threshold (loaded from metadata) will be blocked.
- **Virtual Simulation:** For every trade blocked by TradeGuard, the system must simulate its "virtual outcome" (what would have happened) to calculate the "Block Quality" and "Opportunity Cost".
- **Comparative Analysis:** Maintain a parallel "Shadow Portfolio" that represents performance without TradeGuard filtering to provide a direct baseline comparison.
- **Fail-Safe:** The script must "Fail Fast" and exit if the TradeGuard model or metadata files are missing or invalid.

## Non-Functional Requirements
- **Performance:** TradeGuard inference overhead should be minimized; aim for <2x runtime compared to the combined Alpha+Risk backtest.
- **Auditability:** Generate comprehensive logs for both executed and blocked trades, including confidence scores and reasoning.

## Metrics & Visualizations
- **Core Metrics:** Total Return, Sharpe Ratio, Max Drawdown, Profit Factor, Win Rate.
- **TradeGuard Specifics:** Approval Rate, Block Accuracy (Good Blocks vs. Bad Blocks), Net Value-Add.
- **Visualizations:**
    - Equity curve comparison (Full System vs. Baseline).
    - TradeGuard probability distribution (Approved vs. Blocked).
    - Block quality timeline.
    - Approval rate and Profit Factor by asset.
    - Time-of-day decision analysis.

## Acceptance Criteria
- [ ] Backtest runs to completion using all three layers.
- [ ] Blocked trades are correctly simulated and logged with theoretical PnL.
- [ ] Performance metrics accurately reflect the benefit (or cost) of the TradeGuard layer.
- [ ] Visualizations are generated and saved to the results directory.
- [ ] CLI accepts paths for all three models and the training dataset (for schema verification).

## Out of Scope
- Real-time trading execution.
- Retraining models during the backtest.

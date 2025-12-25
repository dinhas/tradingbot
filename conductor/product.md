# Initial Concept
The user wants to implement the TradeGuard meta-labeling system, a secondary filtering layer using Reinforcement Learning (PPO) to decide whether a trade signal from the Alpha model should be executed or blocked. This agent aims to maximize total portfolio profit by learning to filter out low-quality trades.

# Product Vision
TradeGuard will act as a secondary RL agent that receives trade signals from the Alpha model and determines whether to allow execution. **Note: The filtering layer is a separate end-of-loop component that will be implemented after the model is trained.**

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

**Phase 2: Model Training (COMPLETED)**
*   Implemented `TradeGuardEnv` (Gymnasium) with hybrid PnL reward logic.
*   Created `train_guard.py` for automated PPO training using Stable-Baselines3.
*   Integrated configurable hyperparameter management via YAML.
*   Automated model artifact saving and TensorBoard logging support.

**Phase 4: Live Execution (COMPLETED)**
*   Implemented production-ready autonomous trading system using cTrader Open API.
*   Established sequential inference pipeline: Alpha -> Risk -> TradeGuard.
*   Integrated real-time Discord notifications for trade events and system health.
*   Implemented connection resilience with exponential backoff and heartbeats.
*   Containerized the entire pipeline with Docker for VPS deployment.

**Phase 3: Three-Layer Backtest Integration (COMPLETED)**
*   Implemented `backtest_full_system.py` integrating Alpha (PPO), Risk (PPO), and TradeGuard (PPO).
*   Established strict feature parity between training and inference using `TradeGuardFeatureBuilder`.
*   Integrated a "Shadow Portfolio" (Baseline) system to simulate performance without filtering for direct value-add comparison.
*   Implemented automated virtual simulation for blocked trades to calculate Opportunity Cost and Block Quality.
*   Developed a comprehensive visualization suite featuring comparative equity curves, probability distributions, and block quality timelines.

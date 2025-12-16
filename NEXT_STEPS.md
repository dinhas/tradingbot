# Next Steps: Risk Management Layer

Now that the `risk_dataset.parquet` has been generated (and a sample downloaded), we will proceed to training the Risk Management Model.

## 1. Data Verification & Analysis (EDA)
**Goal:** Ensure the dataset is high quality before training.
- [ ] **Load Sample**: Load the 500-row `risk_dataset.parquet`.
- [ ] **Distributions**: Check Buy/Sell balance and Asset distribution.
- [ ] **Potential**: Analyze `max_profit_pct` vs `max_loss_pct`. This tells us the theoretical potential of every trade.
- [ ] **Integrity**: Verify features are correctly shaped (140 floats) and contain no NaNs.

## 2. Implement Risk Management Environment (Fast Sim)
**Goal:** Create a high-speed Gym environment strictly for optimizing SL/TP.
- [ ] **File**: `src/risk_env.py`
- [ ] **Logic**: Unlike the Alpha environment, this will **NOT** step through time candle-by-candle.
- [ ] **Mechanism**:
    1. Env resets to a random trade from the dataset.
    2. Agent sees the Alpha features + Trade context.
    3. Agent picks SL and TP multipliers.
    4. Env looks at `max_profit_pct` and `max_loss_pct` to instantly determine if TP or SL was hit first.
    5. Returns result (PnL) as reward.
- **Why?**: This "One-Step" or "Episodic" approach is 1000x faster than tick-based training.

## 3. Train the Risk Model
**Goal:** Train a PPO/SAC agent to find the optimal dynamic SL/TP.
- [ ] **File**: `train_risk_layer.py`
- [ ] **Input**: The full `risk_dataset.parquet` (on cloud).
- [ ] **Output**: A small, efficient RL policy (`risk_policy.zip`).

## 4. Validation & Integration
**Goal:** Verify the improvement.
- [ ] **Backtest Comparison**:
    - **Baseline**: Alpha Model with Fixed SL (1.5x) / TP (2.5x).
    - **Test**: Alpha Model + Trained Risk Agent.
- [ ] **Metrics**: Look for higher Sharpe/Sortino ratios and lower Max Drawdown.

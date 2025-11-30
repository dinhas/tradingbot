# Implementation Plan - TradingEnv Refactoring

## Goal
Refactor `trading_env.py` to implement a robust R-multiple based reward system and a stable Risk-Reward (RR) based Take-Profit/Stop-Loss mechanism. This aims to fix reward sparsity, overfitting, and encourage healthy trading behavior.

## User Review Required
> [!IMPORTANT]
> **Action Space Re-interpretation**: The last two dimensions of the action space will be re-mapped.
> - `action[7]` (was raw SL mult): Now maps to **SL Multiplier (2x - 4x ATR)**.
> - `action[8]` (was raw TP mult): Now maps to **Reward Ratio (1.5x - 3.0x)**.
>
> **Reward Scaling**: The new reward function uses R-multiples. The magnitude of rewards will change significantly from the previous version. PPO hyperparameters (like learning rate or clip range) might need adjustment if the reward scale is vastly different, though normalization usually handles this.

## Proposed Changes

### `backtesting/trading_env.py` AND `trading_env.py` (Root)

Both files will be updated with identical logic to ensure consistency between training and backtesting.

#### [MODIFY] `TradingEnv` Class (in both files)

1.  **`__init__` & `reset`**:
    *   Initialize `self.sl_distances = {asset: 0.0}` to track the SL distance (risk) for each position. This is crucial for calculating R-multiples.

2.  **`step` Method - SL/TP Logic**:
    *   **SL Calculation**: `sl_mult = action[7] * 2.0 + 2.0` (Range: 2.0 to 4.0 ATR).
    *   **TP Calculation**: `reward_ratio = action[8] * 1.5 + 1.5` (Range: 1.5 to 3.0).
    *   **TP Distance**: `tp_dist = sl_dist * reward_ratio`.
    *   **Trailing Logic**: Implement a "Break Even" move. If `current_price` moves > 50% of the way to TP, move SL to `entry_price`.

3.  **`step` Method - Reward Calculation**:
    *   **Realized R-multiple**: When a trade closes (SL or TP), calculate `(exit_price - entry_price) / sl_distance_at_entry`.
    *   **Unrealized R-multiple**: For open positions, calculate `(current_price - entry_price) / sl_distance_at_entry`.
    *   **Drawdown Penalty**: Penalize if `portfolio_value` drops from peak.
    *   **Frequency Bonus**: Small positive reward for having active positions (encourages participation).
    *   **Formula**:
        ```python
        reward = (
            (unrealized_r_change * 0.1) +       # Smooth progress signal
            (realized_r * 2.0) -                # Strong signal for closing
            (drawdown_pct * 10.0) +             # Penalty for risk
            (holding_bonus * 0.001)             # Tiny bias to trade
        )
        ```

4.  **`step` Method - Position Management**:
    *   Update `self.sl_distances[asset]` whenever a *new* position is opened or size is increased significantly.

## Impact Analysis & Explanation

### Why apply to both?
*   **Consistency**: If the training environment (`root/trading_env.py`) and backtesting environment (`backtesting/trading_env.py`) differ, the model will learn one behavior but be evaluated on another. This leads to "sim-to-real" gaps where backtest results are invalid.
*   **Validation**: Backtesting is only useful if it accurately reflects the conditions the agent was trained in.

### How this affects the Agent
1.  **Risk-Awareness**: By rewarding "R-multiples" (Profit / Risk) instead of raw dollars, the agent learns that making \$100 risking \$10 is better than making \$100 risking \$1000.
2.  **Patience**: The "Risk-Reward Ratio" constraint (TP must be > 1.5x SL) forces the agent to hunt for setups where it can reasonably expect a larger move, rather than scalping tiny profits with huge risks.
3.  **Stability**: The "Drawdown Penalty" directly discourages the agent from holding onto losing trades hoping they turn around (bag-holding).
4.  **Activity**: The small "Frequency Bonus" prevents the agent from getting stuck in a local optimum of "doing nothing" to avoid losing money.

## Verification Plan

### Automated Tests
*   **Run `backtest_runner.py`**: Verify the environment runs without errors and produces a `trades.txt` (or similar log) showing trades with the new SL/TP logic.
*   **Check Reward Output**: Print sample rewards during a few steps to ensure they are not NaN and are within a reasonable range (e.g., -5 to +5).

### Manual Verification
*   **Inspect Logic**: Verify that TP is always > SL distance (Reward Ratio >= 1.5).
*   **Check "Cheating"**: Ensure the previous "0 SL" exploit is impossible due to the new 2x minimum ATR constraint.

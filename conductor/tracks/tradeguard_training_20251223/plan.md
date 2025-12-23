# Implementation Plan - TradeGuard RL Training

## Phase 1: Configuration & Environment Setup [checkpoint: a3b65cd]
- [x] Task: Create `TradeGuard/config/ppo_config.yaml` 0e3c82e
    - Define default PPO hyperparameters (lr, n_steps, batch_size, gamma, gae_lambda, ent_coef).
    - Define environment parameters (reward_scaling, penalty_factors).
- [x] Task: Create `TradeGuard/src/trade_guard_env.py` d8f9ec4
    - **Sub-task:** Implement `__init__` to load the full Parquet dataset.
    - **Sub-task:** Implement `reset()` to initialize the state pointer.
    - **Sub-task:** Implement `step(action)`:
        - Retrieve current trade data (features & outcome).
        - Calculate Reward based on the "Hybrid PnL" logic.
        - Advance to the next trade.
        - Return `obs, reward, terminated, truncated, info`.
    - **Sub-task:** Write unit tests for `TradeGuardEnv` (state transitions, reward calculation, termination).
- [ ] Task: Conductor - User Manual Verification 'Configuration & Environment Setup' (Protocol in workflow.md)

## Phase 2: Training Pipeline Implementation
- [ ] Task: Create `TradeGuard/src/train_guard.py`
    - **Sub-task:** Implement config loading logic.
    - **Sub-task:** Setup `DummyVecEnv` wrapping `TradeGuardEnv`.
    - **Sub-task:** Initialize PPO model with loaded config.
    - **Sub-task:** Implement the training loop with `model.learn()`.
    - **Sub-task:** Save the final model to `TradeGuard/models/`.
- [ ] Task: Conductor - User Manual Verification 'Training Pipeline Implementation' (Protocol in workflow.md)

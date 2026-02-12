# Risk Model Training Analysis & Finetuning

## 1. Objectives
- **Budget**: 5,000,000 Timesteps.
- **Goal**: Maximize survival on a $10 account while capturing Alpha model signals efficiently.
- **Constraints**: 95% Drawdown termination threshold.

## 2. Configuration & Hyperparameters
We have centralized training parameters in `RiskLayer/config.py`.

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Learning Rate** | `1e-4` | Slow and stable convergence for noisy financial data. |
| **Batch Size** | `4096` | High batch size to reduce gradient variance. |
| **N Steps** | `8192` | Large rollout window per update. |
| **N Epochs** | `4` | Prevent overfitting to local batch noise. |
| **Ent Coef** | `0.05` -> `0.005` | High initial exploration, decaying to exploitation over 4M steps. |
| **Network Arch**| `[256, 128]` | Balanced depth for 45 features; prevents memorization. |
| **Gamma** | `0.98` | Shorter horizon focus for risk management. |

## 3. Environment Refinements
- **Position Sizing**: Fixed at 25% of equity (max) with a **Risk Capacitor** that scales down exposure as drawdown increases.
- **Leverage**: Standardized at 1:100.
- **Reward System**: 
  - **Efficiency Reward**: Normalizes PnL by ATR ratio. 1 ATR of profit = +2.0 Reward.
  - **Bullet Dodger**: +3.0 Bonus for placing stops that prevent crashes (>20% extension beyond SL).
  - **Clipping**: All rewards clipped to `[-10, 10]` for stability.

## 4. Training Strategy
For the 5M step run, we utilize `N_CPU` parallel environments. Observation normalization is enabled with `VecNormalize` to handle the varying scales of market features (ATR, RSI, Macd) vs account state.

## 5. Recent Fixes
- Moved hardcoded paths and params to `config.py`.
- Fixed missing `lots` variable in `risk_env.py` causing potential crashes.
- Synced SL/TP multiplier ranges with expert recommendations (`0.5x` to `2.5x` for SL).

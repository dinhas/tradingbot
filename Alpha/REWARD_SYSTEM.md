# Alpha Model Reward System

## Overview

The Alpha Model employs a specialized reward system designed to solve the "temporal credit assignment" problem inherent in trading. Instead of waiting for a trade to close to receive a reward (which could be hundreds of steps later), the environment uses a **Peek & Label** mechanism during training to provide immediate feedback.

## Training Mode: "Peek & Label"

During training (`is_training=True`), the environment looks into the future immediately upon trade entry to determine the outcome.

### 1. Primary Signal: Future P&L
When a position is opened:
1.  The environment simulates the trade forward up to 1000 candles.
2.  It checks if the Take Profit (TP) or Stop Loss (SL) is hit first.
3.  The P&L of that future outcome is calculated **including execution friction** (spread + slippage).
4.  This future P&L is awarded **immediately** in the current step.

**Normalization:**
```python
Normalized PnL = (Future PnL / Starting Equity) * 2.0
```

### 2. Tiered Speed Bonuses (Scalping Focus)
To encourage capital efficiency and faster turnover, "fast wins" receive significant multipliers.

| Hold Time (Bars) | Multiplier | Description |
| :--- | :--- | :--- |
| **≤ 3** (15 min) | **3.0x** | Ultra-Fast Win |
| **≤ 5** (25 min) | **2.0x** | Fast Win |
| **≤ 12** (1 hour) | **1.5x** | Moderate Win |
| **> 12** | **1.0x** | Standard Win |

### 3. Fast SL Penalty (Precision Focus)
To discourage "bad entries" that immediately go wrong:
- If a trade hits **Stop Loss within 3 bars** (15 mins), an **extra penalty** of **1% of equity** is applied.

### 4. Loss Aversion (Prospect Theory)
To induce risk-averse behavior and mimic human psychology:
- Negative rewards (Losses) are multiplied by **2.25x**.
- *Example:* A 1% loss becomes a -2.25 reward signal, creating a strong deterrent against losing trades.

### 5. Progressive Drawdown Penalty
To prevent the agent from taking excessive risks that lead to deep drawdowns:
- **Trigger:** Drawdown > 5%
- **Penalty:** Scales exponentially with severity.
- **Formula:** `Penalty = -0.15 * ((Drawdown - 0.05) / 0.20) ^ 1.5`
- **Terminal Penalty:** If Drawdown > 25%, the episode ends immediately with a fixed **-0.5** penalty.

---

## Backtesting Mode

During backtesting (`is_training=False`), the "Peek & Label" system is disabled to ensure realistic reporting.
- **Reward:** Sum of **Realized P&L** from trades that actually closed in the current step.
- **Normalization:** Same as training (`(Step PnL / Start Equity) * 2.0`).

---

## Execution Friction

Crucially, the reward system uses the shared `ExecutionEngine` to calculate prices.
- **Entry Price:** `Mid + Spread + Slippage` (for Longs)
- **Exit Price:** `Mid - Slippage` (for Longs)
- **Impact:** The agent is "charged" the spread and slippage immediately in its reward calculation. It learns that it must capture a move *larger* than the spread+slippage to generate a positive reward.

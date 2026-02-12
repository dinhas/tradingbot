# Risk Management Environment - Reward System Documentation

## Overview
The reward system focuses on **trade quality (efficiency)** and **capital preservation**. All rewards are clipped to the range **[-10, 10]** to ensure training stability and prevent gradient explosions.

---

## Reward Components

Action Space (2):
*   **0: Stop Loss Multiplier**: Maps to 0.5x – 2.5x ATR.
*   **1: Take Profit Multiplier**: Maps to 0.5x – 5.0x ATR.

### 1. **PnL Efficiency Reward** (Primary Component)
**Formula:**
```python
atr_ratio = atr / entry_price
efficiency = realized_pct / atr_ratio
pnl_reward = clip(efficiency * 2.0, -5.0, 5.0)
```

**Calculation Steps:**
1. **ATR Normalization:**
   - Normalizes the realized profit/loss by the current **ATR ratio**.
   - This ensures the reward scale is consistent across different volatility regimes.

2. **Scaling:**
   - Multiplied by 2.0 to provide a strong signal.
   - Clipped to **[-5.0, 5.0]**.

**Purpose:** Rewards capturing moves relative to the asset's volatility. It prevents "lucky" volatile trades from dominating the signal while ensuring small, high-quality moves in low-volatility environments are properly incentivized.

**Range:** [-5.0, 5.0]

---

### 2. **Bullet Dodger Bonus** (Capital Preservation)
**Formula:**
```python
if exited_on == 'SL':
    if avoided_loss_dist > (sl_pct_dist * 1.2):
        saved_ratio = (avoided_loss_dist - sl_pct_dist) / sl_pct_dist
        bullet_bonus = min(saved_ratio * 1.5, 3.0)
```

**Details:**
- Triggered when the agent hits a Stop Loss, and the price continues to move against the position by more than **20%** of the SL distance.
- **Avoided Loss:** The distance from the SL price to the eventual worst price reached.
- **Bonus Scaling:** 1.5x the "saved ratio" (how much extra loss was avoided).
- **Maximum Bonus:** +3.0
- **Purpose:** Specifically incentivizes the model to place stops that protect against catastrophic "tail risk" events.

**Range:** [0.0, 3.0]

---

### 3. **Step Penalty** (Optional/Currently Disabled)
- Can be used to encourage faster trade resolutions.
- Typically a small constant like `-0.01` per step.

---

## Final Reward Calculation

```python
reward = clip(pnl_reward + bullet_bonus, -10.0, 10.0)
```

**Final Range:** [-10.0, 10.0]

---

## Data Generation Look-Ahead
- **Window:** 150 Minutes (30 steps for 5m candles).
- The model evaluates the quality of its trade decisions based on the price action occurring within 2.5 hours of the signal.

---

## Reward Scaling Rationale

### Why Clip to [-10, 10]?
1. **RL Stability:** Standard PPO and SAC architectures perform best when rewards are within a small, predictable range.
2. **Gradient Consistency:** Prevents a single catastrophic trade or one massive win from drowning out the lessons learned from hundreds of average trades.

### Why Efficiency vs ATR?
1. **Volatility Agnostic:** 1 ATR of profit is treated the same whether the market is quiet or wild.
2. **Quality Focus:** The agent learns to seek trades that offer high reward-to-risk relative to the market's current "noise" level.

---

## Training Implications

### What the Agent Learns:
1. **Volatility-Adjusted Sizing:** Because rewards are normalized by ATR, the agent learns to respect the current market volatility.
2. **Defensive Discipline:** The Bullet Dodger bonus makes "getting stopped out" a positive event if the stop was correctly placed before a crash.
3. **Consistent Performance:** The tight clipping forces the agent to optimize for the *median* outcome rather than chasing extreme outliers.

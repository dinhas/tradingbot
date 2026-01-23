# Risk Management Environment - Reward System Documentation

## Overview
The reward system focuses on **trade quality (efficiency)** and **capital preservation**. All rewards are clipped to the range **[-100, 100]** to ensure training stability and prevent gradient explosions.

---

## Reward Components

### 1. **PnL Efficiency Reward** (Primary Component)
**Formula:**
```python
denom = max(max_favorable_pct, 1e-5)
realized_pct = (exit_price - entry_price) / entry_price * direction
pnl_efficiency = (realized_pct / denom) * 10.0
```

**Calculation Steps:**
1. **Max Available Profit (Denominator):**
   - Uses the distance from entry to the best price reached during the trade (`max_favorable_pct`).
   - Ensures the agent is rewarded for capturing a high percentage of the *available* move.

2. **Realized Performance:**
   - Calculates the actual percentage change captured by the trade.

3. **Normalized Reward:**
   - Ratio of realized performance to available performance, multiplied by 10.
   - Target range: [-10.0, 10.0] (roughly).

**Purpose:** Rewards capturing the bulk of a move and punishes poor realization (e.g., getting stopped out early on a move that eventually went in your favor).

**Range:** [~-10.0, 10.0]

---

### 2. **Whipsaw Penalty** (Premature Stop Loss)
**Condition:**
- Exited on **SL**
- Price eventually went **> 2.0 * ATR** in the favorable direction (from entry)

**Modification:**
- `pnl_efficiency = pnl_efficiency * 1.5`

**Purpose:** Increases the penalty (since efficiency is negative on loss) when the agent sets a Stop Loss that is too tight, causing it to miss a significant profitable move.

---

### 3. **Bullet Dodger Bonus** (Capital Preservation)
**Formula:**
```python
if exited_on == 'SL':
    if avoided_loss_dist > (sl_pct_dist * 1.5):
        saved_ratio = avoided_loss_dist / sl_pct_dist
        bullet_bonus = min(saved_ratio, 3.0) * 2.0
```

**Details:**
- Triggered when the agent hits a Stop Loss, but the price continues to crash significantly further.
- **Avoided Loss:** The distance from the SL price to the eventual "worst" price.
- **Bonus Scaling:** Scales with how much loss was avoided compared to the SL distance.
- **Maximum Bonus:** +6.0
- **Purpose:** Encourages placing protective stops that save capital during catastrophic moves.

**Range:** [0.0, 6.0]

---

### 4. **Risk Violation Penalty**
**Formula:**
```python
if actual_risk_cash > intended_risk_cash * 2.0 and intended_risk > 1e-9:
    excess_risk_pct = (actual_risk_cash - intended_risk_cash) / equity
    if excess_risk_pct > 0.05:  # 5% of account
        risk_violation_penalty = -2.0 * excess_risk_pct
        risk_violation_penalty = clip(risk_violation_penalty, -10.0, 0.0)
```

**Details:**
- Triggered when actual risk exceeds intended risk by **2x**
- Only applies if excess risk is **> 5% of account equity**
- Penalty scales with excess risk percentage
- Clipped to [-10.0, 0.0] range

**Purpose:** Prevents agent from accidentally taking excessive risk due to rounding or leverage constraints.

**Range:** [-10.0, 0.0]

---

## Final Reward Calculation

```python
reward = pnl_efficiency + bullet_bonus + risk_violation_penalty
reward = clip(reward, -20.0, 20.0)  # Final safety clip
```

**Final Range:** [-100.0, 100.0]

---

## Special Cases

### 1. **Skipped Trades** (Low Risk Request)
**Condition:** `risk_raw < 1e-3` (agent requests < 0.1% risk)
**Reward:** `0.0`

### 2. **Skipped Small Positions**
**Condition:** Calculated lots < MIN_LOTS (0.01)
**Reward:** `0.0`

### 3. **Terminal Penalty** (Equity Drop)
**Condition:** `equity < initial_equity * 0.3` (70% drawdown)
**Penalty:** `-20.0`
**Result:** Episode terminates immediately
**Final Reward:** `clip(reward - 20.0, -100.0, 100.0)`

---

## Configuration Constants

```python
TRADING_COST_PCT = 0.0002      # 2 pips/ticks roundtrip cost
MIN_LOTS = 0.01                # Minimum position size
CONTRACT_SIZE = 100000         # Standard lot size
DEFAULT_RISK_PCT = 0.25        # Fixed 25% risk per trade
MAX_RISK_PER_TRADE = 0.40      # 40% max risk per trade
MAX_MARGIN_PER_TRADE_PCT = 0.80  # 80% max margin usage
MAX_LEVERAGE = 400.0           # 1:400 leverage
EPISODE_LENGTH = 100           # Fixed episode length
```

---

## Reward Scaling Rationale

### Why Clip to [-100, 100]?
1. **Value Function Stability:** Extreme rewards break value function learning (causes negative explained variance).
2. **Gradient Stability:** Prevents gradient explosions in PPO.
3. **Training Stability:** Keeps loss curves smooth and predictable.

### Why Efficiency instead of raw PnL?
1. **Scale Independence:** Same efficiency score regardless of account size or asset price.
2. **Quality Focus:** Forces the agent to optimize for the *best* exit, not just any profit.
3. **ATR Normalization:** Prevents rewarding "luck" in high-volatility environments while maintaining signal strength in low-volatility ones.

---

## Training Implications

### What the Agent Learns:
1. **Maximize Efficiency:** Primary objective is to capture as much of the favorable move as possible.
2. **Capital Preservation:** Bullet Dodger bonus teaches that hitting a stop loss is better than riding a crash.
3. **Respect Risk Limits:** Risk violation penalty prevents over-leveraging.

### Potential Issues Addressed:
- ✅ **Reward Stability:** Tight clipping prevents extreme values.
- ✅ **Risk Control:** Focus on avoiding large crashes via SL.
- ✅ **No Drawdown Obsession:** Removal of the absolute drawdown penalty prevents the agent from becoming "too scared" to trade during normal market pullbacks.
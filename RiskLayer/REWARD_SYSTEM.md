# Risk Management Environment - Reward System Documentation

## Overview
The reward system consists of **4 main components** that are combined to form the final reward signal. All rewards are clipped to the range **[-100, 100]** to prevent extreme values that break value function learning.

---

## Reward Components

### 1. **PnL Reward** (Primary Component)
**Formula:**
```python
pnl_ratio = net_pnl / max(prev_equity, 1e-6)
pnl_reward = clip(pnl_ratio * 10.0, -50.0, 50.0)
```

**Calculation Steps:**
1. **Gross PnL (Quote Currency):**
   ```python
   price_change = exit_price - entry_price
   gross_pnl_quote = price_change * lots * CONTRACT_SIZE * direction
   ```

2. **Currency Conversion to USD:**
   - **USD Quote pairs** (EURUSD, GBPUSD, XAUUSD): `gross_pnl_usd = gross_pnl_quote`
   - **USD Base pairs** (USDJPY, USDCHF): `gross_pnl_usd = gross_pnl_quote / exit_price`
   - **Cross pairs**: Uses fallback conversion

3. **Net PnL (After Costs):**
   ```python
   costs = position_val * TRADING_COST_PCT  # 0.0002 (2 pips/ticks)
   net_pnl = gross_pnl_usd - costs
   ```

4. **Normalized Reward:**
   - Normalized by previous equity to make rewards scale-independent
   - Multiplied by 10.0 for scaling
   - Clipped to [-50.0, 50.0] range

**Range:** [-50.0, 50.0]

---

### 2. **Take Profit (TP) Bonus**
**Formula:**
```python
if exited_on == 'TP':
    tp_bonus = 0.5 * (tp_mult / 8.0)
else:
    tp_bonus = 0.0
```

**Details:**
- Only applied when trade exits via Take Profit
- Scales with TP multiplier (0.5x to 4.0x ATR)
- Maximum bonus: `0.5 * (4.0 / 8.0) = 0.25`
- Encourages agent to set appropriate TP levels

**Range:** [0.0, 0.25]

---

### 3. **Drawdown Penalty** (Risk Control)
**Formula:**
```python
prev_dd = 1.0 - (prev_equity / prev_peak_equity)
new_dd = 1.0 - (current_equity / prev_peak_equity)
dd_increase = max(0.0, new_dd - prev_dd)
dd_penalty = -(dd_increase ** 2) * 50.0
```

**Details:**
- **Quadratic penalty** on increasing drawdown
- Only penalizes **increases** in drawdown (not absolute drawdown)
- Multiplier: **50.0** (reduced from 2000.0 to prevent extreme values)
- Strongly discourages actions that worsen drawdown

**Example:**
- If drawdown increases from 5% to 10%:
  - `dd_increase = 0.05`
  - `dd_penalty = -(0.05²) * 50.0 = -0.125`

**Range:** [-∞, 0.0] (but effectively bounded by final clipping)

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

**Purpose:** Prevents agent from accidentally taking excessive risk due to:
- Rounding up to MIN_LOTS
- Leverage/margin constraints
- Position size clamping

**Range:** [-10.0, 0.0]

---

## Final Reward Calculation

```python
reward = pnl_reward + tp_bonus + dd_penalty + risk_violation_penalty
reward = clip(reward, -100.0, 100.0)  # Final safety clip
```

**Final Range:** [-100.0, 100.0]

---

## Special Cases

### 1. **Skipped Trades** (Low Risk Request)
**Condition:** `risk_raw < 1e-3` (agent requests < 0.1% risk)
**Reward:** `0.0`
**Info:** `{'exit': 'SKIPPED', 'pnl': 0.0}`

### 2. **Skipped Small Positions**
**Condition:** Calculated lots < MIN_LOTS (0.01)
**Reward:** `0.0`
**Info:** `{'exit': 'SKIPPED_SMALL', 'pnl': 0.0}`

### 3. **Terminal Penalty** (Equity Drop)
**Condition:** `equity < initial_equity * 0.3` (70% drawdown)
**Penalty:** `-50.0` (reduced from -200.0)
**Result:** Episode terminates immediately
**Final Reward:** `clip(reward - 50.0, -100.0, 100.0)`

---

## Configuration Constants

```python
TRADING_COST_PCT = 0.0002      # 2 pips/ticks roundtrip cost
MIN_LOTS = 0.01                # Minimum position size
CONTRACT_SIZE = 100000         # Standard lot size
MAX_RISK_PER_TRADE = 0.40     # 40% max risk per trade
MAX_MARGIN_PER_TRADE_PCT = 0.80  # 80% max margin usage
MAX_LEVERAGE = 400.0           # 1:400 leverage
EPISODE_LENGTH = 100           # Fixed episode length
```

---

## Reward Scaling Rationale

### Why Clip to [-100, 100]?
1. **Value Function Stability:** Extreme rewards break value function learning (causes negative explained variance)
2. **Gradient Stability:** Prevents gradient explosions in PPO
3. **Training Stability:** Keeps loss curves smooth and predictable

### Why Normalize PnL by Equity?
1. **Scale Independence:** Same % gain/loss gives same reward regardless of account size
2. **Fair Comparison:** Allows comparison across different equity levels
3. **Stability:** Prevents extreme rewards when equity is small

### Why Quadratic Drawdown Penalty?
1. **Progressive Discouragement:** Small increases = small penalty, large increases = large penalty
2. **Risk Aversion:** Strongly discourages actions that worsen drawdown
3. **Smooth Gradient:** Quadratic function provides smooth gradients for learning

---

## Example Reward Scenarios

### Scenario 1: Profitable Trade with TP Hit
- Net PnL: +$2.00 (2% of $100 equity)
- TP Multiplier: 2.0x ATR
- Drawdown: No change
- Risk: Within limits

**Calculation:**
- `pnl_reward = clip(0.02 * 10.0, -50, 50) = 0.2`
- `tp_bonus = 0.5 * (2.0 / 8.0) = 0.125`
- `dd_penalty = 0.0`
- `risk_penalty = 0.0`
- **Total Reward: 0.325**

### Scenario 2: Loss with Drawdown Increase
- Net PnL: -$5.00 (5% of $100 equity)
- Drawdown increases: 10% → 15%
- No TP hit
- Risk: Within limits

**Calculation:**
- `pnl_reward = clip(-0.05 * 10.0, -50, 50) = -0.5`
- `tp_bonus = 0.0`
- `dd_penalty = -(0.05²) * 50.0 = -0.125`
- `risk_penalty = 0.0`
- **Total Reward: -0.625**

### Scenario 3: Risk Violation
- Net PnL: +$1.00 (1% of $100 equity)
- Actual risk: 50% (intended: 20%, exceeds 2x threshold)
- Excess: 30% of account
- Drawdown: No change

**Calculation:**
- `pnl_reward = clip(0.01 * 10.0, -50, 50) = 0.1`
- `tp_bonus = 0.0`
- `dd_penalty = 0.0`
- `risk_penalty = -2.0 * 0.30 = -0.6` (clipped to -10.0 max)
- **Total Reward: -0.5**

---

## Training Implications

### What the Agent Learns:
1. **Maximize PnL:** Primary objective through pnl_reward
2. **Set Appropriate TP:** TP bonus encourages good TP placement
3. **Control Drawdown:** Drawdown penalty discourages risky behavior
4. **Respect Risk Limits:** Risk violation penalty prevents over-leveraging

### Potential Issues Addressed:
- ✅ **Reward Stability:** Clipping prevents extreme values
- ✅ **Value Function Learning:** Normalized rewards help value function converge
- ✅ **Gradient Stability:** Bounded rewards prevent gradient explosions
- ✅ **Risk Control:** Multiple penalties encourage conservative risk management


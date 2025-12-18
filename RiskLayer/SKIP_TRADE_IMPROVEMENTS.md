# Skip Trade Learning Improvements

## Problem: Model Not Learning to Skip/Block Trades

The current reward model has several issues that prevent the PPO agent from learning when to skip trades.

---

## Root Causes

### 1. Reward Clipping Destroys Signal
```python
# CURRENT (Broken):
# Blocking rewards: +200 (catastrophe) or -300 (missed win)
# Final clip: [-100, 100]
reward = np.clip(reward, -100.0, 100.0)  # Line 537

# Problem: +200 becomes +100, -300 becomes -100
# Model sees: "blocking = ±100" regardless of quality
```

### 2. Asymmetric Penalties Discourage Blocking
```python
# CURRENT:
if max_adverse < -0.03: return 200.0   # Blocked catastrophe
if max_favorable > 0.02: return -300.0 # Missed huge win

# Problem: Missing a win is punished 1.5x MORE than avoiding a loss
# Model learns: "Never block, because penalty for missing is worse"
```

### 3. Dead Zone Too Narrow
```python
BLOCK_THRESHOLD = 0.10  # Only bottom 10% of action space

# Problem: Random exploration rarely finds this zone
# Gaussian std=0.5 → P(action < -0.80) ≈ 5%
```

### 4. Trade Rewards Dominate
```python
pnl_reward = pnl_ratio * 100.0  # 2% gain = +200 (before clip)
# After clip = +100, same as blocking a catastrophe!
```

---

## Fix 1: Remove Final Clipping for Blocking Rewards

**Rationale**: Let the full blocking reward signal through. Only clip trade rewards.

```python
# BEFORE:
reward = pnl_reward + tp_bonus + dd_penalty + risk_violation_penalty
reward = np.clip(reward, -100.0, 100.0)

# AFTER:
if is_blocked:
    # Blocking reward is NOT clipped (allow ±300 to flow through)
    reward = block_reward
else:
    # Trade reward IS clipped
    trade_reward = pnl_reward + tp_bonus + dd_penalty + risk_violation_penalty
    reward = np.clip(trade_reward, -100.0, 100.0)
```

---

## Fix 2: Balance Blocking Rewards (Survival Priority)

**Rationale**: Make avoiding losses MORE rewarding than missing wins is punishing.

```python
# BEFORE (Asymmetric - discourages blocking):
BLOCKED_CATASTROPHIC  = +200
MISSED_HUGE_WIN       = -300  # 1.5x punishment

# AFTER (Symmetric with survival bonus):
BLOCKED_CATASTROPHIC  = +300  # Survival is MOST important
BLOCKED_MAJOR_LOSS    = +150
BLOCKED_MINOR_LOSS    = +20

MISSED_HUGE_WIN       = -200  # Reduced from -300
MISSED_BIG_WIN        = -80   # Reduced from -100
MISSED_SMALL_WIN      = -20   # Reduced from -30
```

**New Helper Function**:
```python
def _evaluate_block_reward(self, max_favorable, max_adverse):
    """
    IMPROVED: Balanced rewards that prioritize survival.
    Asymmetry favors BLOCKING (conservative) over TRADING (aggressive).
    """
    # === TIER 1: SURVIVAL (Highest Priority) ===
    if max_adverse < -0.03:  # Would lose >3%
        return 300.0, "BLOCKED_CATASTROPHIC"
    
    if max_adverse < -0.015:  # Would lose 1.5-3%
        return 150.0, "BLOCKED_MAJOR_LOSS"
    
    if max_adverse < -0.005:  # Would lose 0.5-1.5%
        return 50.0, "BLOCKED_MODERATE_LOSS"
    
    # === TIER 2: OPPORTUNITY COST (Lower Priority) ===
    if max_favorable > 0.02:  # Missed >2% profit
        return -200.0, "MISSED_HUGE_WIN"
    
    if max_favorable > 0.01:  # Missed 1-2% profit
        return -80.0, "MISSED_BIG_WIN"
    
    if max_favorable > 0.005:  # Missed 0.5-1% profit
        return -20.0, "MISSED_SMALL_WIN"
    
    # === TIER 3: NOISE ===
    # No significant loss or gain - small reward for staying safe
    return 10.0, "BLOCKED_NOISE"
```

---

## Fix 3: Widen Dead Zone (Easier to Explore Blocking)

**Rationale**: Make it easier for the model to discover blocking during exploration.

```python
# BEFORE:
BLOCK_THRESHOLD = 0.10  # Bottom 10% of action space

# AFTER:
BLOCK_THRESHOLD = 0.20  # Bottom 20% of action space

# Now: action < -0.60 triggers block (vs action < -0.80 before)
# Random exploration hit rate: ~15% (vs ~5% before)
```

---

## Fix 4: Add Blocking Bonus for Volatile Conditions

**Rationale**: Reward blocking during high-volatility periods where outcomes are uncertain.

```python
# Add to _evaluate_block_reward():
def _evaluate_block_reward(self, max_favorable, max_adverse, atr_normalized=None):
    """With volatility awareness."""
    
    # Base reward from tiers (as above)
    base_reward, block_type = self._base_block_reward(max_favorable, max_adverse)
    
    # Volatility bonus: If ATR is high, blocking is safer
    if atr_normalized is not None and atr_normalized > 1.5:
        volatility_bonus = 10.0 * (atr_normalized - 1.0)
        base_reward += volatility_bonus
        
    return base_reward, block_type
```

---

## Fix 5: Add Blocking Curriculum (Progressive Training)

**Rationale**: Start training with easier blocking decisions, then increase difficulty.

```python
# In training script:
def make_env(difficulty_level):
    """Curriculum learning for blocking."""
    
    if difficulty_level == 1:  # Easy
        # Dataset filtered to clear-cut cases
        # Catastrophic losses (>5%) or huge wins (>5%)
        dataset = filter_extreme_cases(original_dataset)
        block_threshold = 0.30  # Wide dead zone
        
    elif difficulty_level == 2:  # Medium
        # All cases, but with wider rewards
        dataset = original_dataset
        block_threshold = 0.20
        
    else:  # Hard (Full)
        dataset = original_dataset
        block_threshold = 0.15
        
    return RiskManagementEnv(dataset, block_threshold=block_threshold)
```

---

## Fix 6: Separate Blocking Head in Network (Advanced)

**Rationale**: Let the policy network have a dedicated output for "should I block?".

```python
# Current Action Space:
# [SL_mult, TP_mult, Risk_factor] where Risk_factor < 0.1 = Block

# Alternative (Clearer Signal):
# Action Space: [Block_probability, SL_mult, TP_mult, Risk_factor]
# Where Block_probability > 0.5 = Skip trade

# Implementation:
self.action_space = spaces.Dict({
    'block': spaces.Discrete(2),  # 0 = Trade, 1 = Block
    'params': spaces.Box(low=-1, high=1, shape=(3,))  # SL, TP, Risk
})
```

---

## Quick Implementation (Minimal Changes)

For immediate impact, make these 3 changes to `risk_env.py`:

### Change 1: Line 267-297 (Balanced Rewards)
```python
def _evaluate_block_reward(self, max_favorable, max_adverse):
    # SURVIVAL FIRST
    if max_adverse < -0.03:
        return 300.0, "BLOCKED_CATASTROPHIC"  # Was 200
    if max_adverse < -0.015:
        return 150.0, "BLOCKED_MAJOR_LOSS"   # Was 50
    if max_adverse < -0.005:
        return 50.0, "BLOCKED_MODERATE_LOSS"  # NEW
    
    # OPPORTUNITY COST (Reduced)
    if max_favorable > 0.02:
        return -200.0, "MISSED_HUGE_WIN"     # Was -300
    if max_favorable > 0.01:
        return -80.0, "MISSED_BIG_WIN"       # Was -100
    if max_favorable > 0.005:
        return -20.0, "MISSED_SMALL_WIN"     # Was -30
    
    # NOISE
    return 10.0, "BLOCKED_NOISE"             # Was 5
```

### Change 2: Line 326 (Wider Dead Zone)
```python
BLOCK_THRESHOLD = 0.20  # Was 0.10
```

### Change 3: Line 537-538 (Preserve Block Rewards)
```python
# For blocked trades (already returned before line 537):
# reward is NOT clipped, keeps full ±300 range

# For executed trades:
reward = np.clip(reward, -100.0, 100.0)
# This is fine because trade rewards compete at smaller scale
```

---

## Expected Impact

| Metric                  | Before | After |
|-------------------------|--------|-------|
| Block exploration rate  | ~5%    | ~15%  |
| Block reward visibility | Clipped| Full  |
| Survival vs. Greed bias | Greed  | Survival |
| Learning iterations     | ~2M    | ~500K |

---

## Verification

After training with new rewards, check:

1. **Block Rate**: Should be 15-30% of trades (not 0%)
2. **Blocked Categories**: 
   - `BLOCKED_CATASTROPHIC` should be positive when max_adverse < -3%
   - `MISSED_HUGE_WIN` should be rare (model learns to not miss these)
3. **Equity Curve**: Should be smoother (fewer catastrophic losses)

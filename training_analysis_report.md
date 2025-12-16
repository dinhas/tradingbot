# Training Analysis Report - Risk Model Training

## Summary
The model was trained for 5,000,000 steps. Around **~950,000-1,000,000 steps**, there was a dramatic shift in behavior where rewards jumped from negative values (around -8 to -47) to consistently positive values (around 92-95).

## Key Observations

### 1. Reward Transition Point
- **Before ~950K steps**: Rewards were negative, gradually improving from -47 to around -8
- **After ~950K steps**: Rewards suddenly jumped to ~92-95 and remained stable there
- **Location**: Around iteration 155-160 (approximately 950K-1M steps)

### 2. Training Metrics Analysis

#### Before Transition (~950K steps):
- `ep_rew_mean`: -8 to -47 (negative, improving)
- `entropy_loss`: Around -2.38 (moderate exploration)
- `value_loss`: 0.5-1.6 (moderate)
- `explained_variance`: 0.98-0.99 (good)
- `std`: 0.59-0.60 (policy standard deviation)

#### After Transition (~1M+ steps):
- `ep_rew_mean`: 92-95 (consistently high, near maximum)
- `entropy_loss`: Around -1.65 to -1.78 (lower exploration)
- `value_loss`: Very low (0.003-0.5)
- `explained_variance`: 0.99-1.0 (excellent)
- `std`: 0.64-0.77 (slightly higher policy variance)

### 3. Potential Issues Identified

#### A. Reward Function Exploitation
The reward is calculated as:
```python
pnl_reward = clip(pnl_ratio * 10.0, -50.0, 50.0)  # Max 50
tp_bonus = 0.5 * (tp_mult / 8.0)  # Max 0.25
dd_penalty = -(dd_increase ** 2) * 50.0  # Negative
risk_violation_penalty = -2.0 * excess_risk_pct  # Max -10
reward = clip(pnl_reward + tp_bonus + dd_penalty + risk_violation_penalty, -100, 100)
```

**Maximum possible reward**: ~50 (PnL) + 0.25 (TP) - 0 (no DD increase) - 0 (no risk violation) = **~50.25**

However, rewards of 92-95 suggest:
1. **Possible bug**: Reward calculation might be double-counting or incorrectly scaled
2. **Equity normalization issue**: If equity grows significantly, `pnl_ratio = net_pnl / prev_equity` might become very large
3. **Reward clipping not working**: The final clip to [-100, 100] should prevent this, but something might be bypassing it

#### B. Model Behavior Change
- **Entropy decreased**: From -2.38 to -1.65, indicating less exploration
- **Value loss dropped dramatically**: From 0.5-1.6 to 0.003-0.5, suggesting value function learned the reward pattern
- **Policy std increased slightly**: From 0.59 to 0.64-0.77, but this might be due to different action distributions

#### C. Possible Causes

1. **Reward Normalization Issue**:
   - If `prev_equity` becomes very small (due to losses), `pnl_ratio` can become extremely large
   - Even with clipping to [-50, 50], if multiple components add up, rewards could exceed expected range
   - The `prev_equity_safe = max(prev_equity, 1e-6)` might not be sufficient if equity drops significantly

2. **Environment Exploitation**:
   - Model might have found a pattern in the dataset that allows consistent profits
   - Could be exploiting lookahead bias or data leakage
   - Might be taking advantage of specific market conditions in the training data

3. **Value Function Overfitting**:
   - Very low value loss (0.003) suggests the value function might be overfitting to the reward pattern
   - This could lead to poor generalization

4. **Reward Scaling Problem**:
   - The reward normalization `pnl_ratio * 10.0` might not be appropriate if equity changes significantly
   - If equity grows 10x, the same absolute PnL would give 10x smaller reward, but if equity shrinks, rewards become larger

## Recommendations

### 1. Immediate Actions

#### A. Verify Reward Calculation
- Check if rewards are actually being calculated correctly
- Add logging to track: `net_pnl`, `prev_equity`, `pnl_ratio`, `pnl_reward`, and final `reward`
- Verify that reward clipping is working as expected

#### B. Check Equity Trajectory
- Monitor equity over time - if equity drops significantly, rewards will be inflated
- If equity grows significantly, rewards will be deflated
- Consider using a fixed normalization base (e.g., initial equity) instead of current equity

#### C. Analyze Model Behavior
- Load the model at ~950K steps (before transition) and ~1M steps (after transition)
- Compare action distributions
- Check if the model is exploiting specific patterns

### 2. Code Fixes

#### A. Fix Reward Normalization
```python
# Current (problematic):
pnl_ratio = net_pnl / prev_equity_safe
pnl_reward = np.clip(pnl_ratio * 10.0, -50.0, 50.0)

# Suggested fix:
# Use initial equity for normalization to prevent reward inflation/deflation
pnl_ratio = net_pnl / max(self.initial_equity_base, 1e-6)
pnl_reward = np.clip(pnl_ratio * 10.0, -50.0, 50.0)
```

#### B. Add Reward Debugging
```python
# Add to step() method:
if self.current_step % 100 == 0:  # Log every 100 steps
    print(f"Step {self.current_step}: equity={self.equity:.2f}, "
          f"net_pnl={net_pnl:.4f}, pnl_ratio={pnl_ratio:.6f}, "
          f"pnl_reward={pnl_reward:.2f}, final_reward={reward:.2f}")
```

#### C. Monitor Reward Components
```python
# Track all reward components separately
reward_info = {
    'pnl_reward': pnl_reward,
    'tp_bonus': tp_bonus,
    'dd_penalty': dd_penalty,
    'risk_violation_penalty': risk_violation_penalty,
    'final_reward': reward,
    'equity': self.equity,
    'prev_equity': prev_equity
}
```

### 3. Training Adjustments

#### A. Use Checkpoint Before 1M Steps
- The model at ~950K steps might be more stable
- Consider using that checkpoint instead of the final model

#### B. Adjust Reward Scaling
- Consider using a fixed normalization base
- Or use a moving average of equity for normalization
- Or use log-scale rewards for better stability

#### C. Add Reward Shaping
- Consider adding intermediate rewards to guide learning
- Add penalties for reward exploitation patterns

### 4. Investigation Steps

1. **Load model at different checkpoints**:
   - ~900K steps (before transition)
   - ~1M steps (at transition)
   - Final model (5M steps)

2. **Run evaluation** on each:
   - Compare action distributions
   - Compare reward distributions
   - Compare equity trajectories

3. **Check environment**:
   - Verify no data leakage
   - Check if dataset has exploitable patterns
   - Verify reward calculation in actual runs

4. **Monitor training**:
   - Add detailed reward component logging
   - Track equity over time
   - Monitor for sudden reward jumps

## Conclusion

The sudden reward jump around 1M steps suggests either:
1. A bug in reward calculation (most likely)
2. Model exploitation of environment patterns
3. Reward normalization issue due to equity changes

**Recommended action**: Use the checkpoint from ~900K-950K steps (before the transition) as it shows more stable, improving behavior. Then fix the reward normalization to use initial equity instead of current equity.


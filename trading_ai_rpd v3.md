# Multi-Asset AI Trading System - Requirements & Planning Document (RPD) v3.0

## Executive Summary

This document outlines a **production-ready** reinforcement learning trading system for 6 cryptocurrency and forex pairs with dynamic capital allocation. The design incorporates all critical fixes to prevent common RL failures: reward hacking, catastrophic forgetting, transaction costs, session awareness, and multi-timeframe context.

**Core Philosophy**: Robust, not clever. Every design decision optimized to prevent known failure modes.

---

## 1. System Overview

### 1.1 Core Objectives
- **Universal Model**: Single PPO agent trading 6 assets + cash simultaneously
- **Dynamic Capital Allocation**: Model learns optimal portfolio weights including cash position
- **Temporal Memory**: LSTM architecture captures sequential patterns
- **Multi-Timeframe Awareness**: Combines 15-min execution with 4H/Daily trend context
- **Transaction Cost Aware**: Realistic fees prevent noise trading
- **Session Intelligence**: Action masking + time features for session-aware trading

### 1.2 Target Assets (7 Total)

**Cryptocurrency (3 pairs)**:
- BTC/USD (Bitcoin)
- ETH/USD (Ethereum)  
- SOL/USD (Solana)

**Forex (3 pairs)**:
- EUR/USD (Euro/US Dollar)
- GBP/USD (British Pound/US Dollar)
- USD/JPY (US Dollar/Japanese Yen)

**Cash Position**:
- USD/USDC (Risk-off safe haven)

---

## 2. Architecture Design

### 2.1 Model Architecture

**Algorithm**: Recurrent Proximal Policy Optimization (RecurrentPPO)

**Neural Network**: LSTM-based Policy
```
Input Layer (85 features) 
    → LSTM(128 units, return_sequences=True)
    → LSTM(128 units)
    → Dense(256, ReLU) 
    → Dense(128, ReLU) 
    → Output Layer (Action Distribution)
```

**Why LSTM?**
- **Temporal Memory**: Remembers last 10-20 candles of market behavior
- **Pattern Recognition**: Detects setups forming over time (consolidation → breakout)
- **Regime Adaptation**: Learns to recognize when market character changes
- **Sequential Context**: Understands "3 red candles in a row" vs "3 random down candles"

**Trade-off**:
- Training time: 3× slower than MLP (~24 hours total vs 8 hours)
- RAM usage: +40% due to hidden states (~900MB vs 650MB)
- **Acceptable**: With paid Colab Pro, we have 25GB RAM + longer sessions

### 2.2 Action Space Design

**Continuous Action Space**: `Box(low=0, high=1, shape=(7,))` 🔴 **FIXED**

**Action Vector** = `[w_btc, w_eth, w_sol, w_eur, w_gbp, w_jpy, w_cash]`

- Each value represents target portfolio weight (0.0 to 1.0)
- Softmax normalization ensures weights sum to 1.0
- **Cash is an explicit action** (can sit out bad markets)

**Example Action**:
```python
action = [0.15, 0.10, 0.05, 0.10, 0.10, 0.05, 0.45]
# Interpretation:
# - 15% BTC, 10% ETH, 5% SOL (40% crypto)
# - 10% EUR, 10% GBP, 5% JPY (25% forex)
# - 45% CASH (risk-off, waiting for opportunity)
```

**Action Masking** (Forbidden Actions):
```python
# If forex session is closed, mask those assets
action_mask = [
    True,   # BTC (always available)
    True,   # ETH (always available)
    True,   # SOL (always available)
    is_forex_session_open('EUR'),  # EUR (session-dependent)
    is_forex_session_open('GBP'),  # GBP (session-dependent)
    is_forex_session_open('JPY'),  # JPY (session-dependent)
    True    # Cash (always available)
]

# MaskablePPO forces masked actions to probability = 0
```

**Execution Logic with Transaction Costs**: 🔴 **FIXED**
```python
def execute_rebalance(action):
    # Normalize to portfolio weights
    weights = softmax(action)
    
    # Calculate target positions
    target_positions = portfolio_value * weights
    
    # Execute trades & deduct fees
    total_fees = 0
    for asset, target in target_positions.items():
        current = current_positions[asset]
        trade_amount = abs(target - current)
        
        if trade_amount > min_trade_size:
            # Calculate fee
            if asset in ['BTC', 'ETH', 'SOL']:
                fee = trade_amount * 0.001  # 0.1% crypto
            elif asset in ['EUR', 'GBP', 'JPY']:
                fee = trade_amount * 0.0005  # 0.05% forex spread
            else:  # Cash
                fee = 0
            
            total_fees += fee
            
            # Execute trade
            if target > current:
                buy(asset, target - current)
            else:
                sell(asset, current - target)
    
    # Deduct fees from portfolio BEFORE reward calculation
    portfolio_value -= total_fees
    
    return portfolio_value, total_fees
```

**Risk Controls** (Hard Constraints):
- Maximum single asset weight: **40%**
- Minimum position size: **$100** (avoid micro-positions)
- Minimum cash reserve: **5%** (always keep some liquidity)
- Maximum portfolio leverage: **1.0×** (no margin)

---

## 3. Data Strategy

### 3.1 Historical Data Requirements

**Training Window**: 5 years (2020-2024)
- Covers: COVID crash, bull run, bear market, recovery
- Sufficient market regime diversity
- Computational feasibility with Colab Pro

**Timeframe**: **15-minute candles** (Primary execution timeframe)
- ~175,000 candles per asset over 5 years
- Captures intraday volatility
- Balance between detail and computational efficiency

**Additional Timeframes** (Calculated from 15-min data):
- **4-Hour candles**: Mid-term trend context
- **Daily candles**: Long-term trend filter

**Market Regimes Included**:
1. **Crash**: March 2020 COVID panic
2. **Bull Market**: Late 2020 - Nov 2021
3. **Bear Market**: 2022 (Luna, FTX collapse)
4. **Recovery/Consolidation**: 2023-2024

### 3.2 Data Split Strategy

**Walk-Forward Validation** (Prevent Look-Ahead Bias):

```
Split 1: Train 2020-2021 (2 years) → Test 2022 (1 year)
Split 2: Train 2020-2022 (3 years) → Test 2023 (1 year)
Split 3: Train 2020-2023 (4 years) → Test 2024 (1 year)
```

**Final Model**:
- Train: 2020-2023 (4 years)
- Validation: 2024 Q1-Q3 (9 months)
- Out-of-Sample Test: 2024 Q4 (3 months) + Live Paper Trading

**Critical**: No future data leakage. When calculating Daily EMA on row N, use only data from rows 0 to N-1.

---

## 4. Feature Engineering

### 4.1 Design Principles
- **Stationarity**: All features must be stationary (no raw prices)
- **Multi-Timeframe**: Combine 15-min signals with 4H/Daily context
- **Normalization**: Asset-specific z-score scaling
- **Self-Awareness**: Include portfolio state

### 4.2 Core Market Features (Per Asset)

**15-Minute Timeframe** (8 features × 6 assets = 48 features):

1. **Log Returns** = log(close/close_prev)
2. **Distance from EMA(50)** = (close - EMA50) / EMA50
3. **ATR(14) / Close** - Normalized volatility
4. **Bollinger Band Width** = (upper - lower) / middle
5. **RSI(14) / 100** - Normalized momentum
6. **MACD Histogram / Close** - Normalized trend momentum
7. **Volume Ratio** = volume / SMA(volume, 20)
8. **ADX(14) / 100** - Trend strength

**4-Hour Timeframe** (3 features × 6 assets = 18 features): 🔴 **NEW**

9. **4H_RSI / 100** - Mid-term momentum
10. **Distance from 4H_EMA(50)** = (close - 4H_EMA50) / 4H_EMA50
11. **4H_ATR / Close** - Mid-term volatility

**Daily Timeframe** (2 features × 6 assets = 12 features): 🔴 **NEW**

12. **Distance from Daily_EMA(200)** = (close - Daily_EMA200) / Daily_EMA200
    - **Critical**: Long-term trend filter (is macro trend up or down?)
13. **Daily_RSI / 100** - Long-term momentum

**Total Market Features**: 48 + 18 + 12 = **78 features**

### 4.3 Temporal Features (3 features): 🔴 **FIXED**

14. **sin_hour** = sin(hour × 2π / 24) - Cyclical encoding of hour
15. **cos_hour** = cos(hour × 2π / 24) - Completes cycle representation
16. **day_of_week** = 0-6 (Monday=0, Sunday=6)

**Why Cyclical Encoding?**
- Hour 23 and Hour 0 are only 1 hour apart, not 23 hours
- sin/cos creates a continuous circle (no arbitrary boundaries)

### 4.4 Session Status Features (6 features): 🔴 **NEW**

17. **is_btc_tradeable** = 1 (always, except low-liquidity penalty window)
18. **is_eth_tradeable** = 1 (always)
19. **is_sol_tradeable** = 1 (always)
20. **is_eur_tradeable** = 1 if London/NY session open, else 0
21. **is_gbp_tradeable** = 1 if London/NY session open, else 0
22. **is_jpy_tradeable** = 1 if Tokyo/London session open, else 0

**Purpose**: Model learns to correlate time features with session status, understands when rejections happen.

### 4.5 Portfolio State Features (8 features): 🔴 **NEW**

23. **current_weight_btc** = current_btc_value / portfolio_value
24. **current_weight_eth** = ...
25. **current_weight_sol** = ...
26. **current_weight_eur** = ...
27. **current_weight_gbp** = ...
28. **current_weight_jpy** = ...
29. **current_weight_cash** = cash / portfolio_value
30. **unrealized_pnl_pct** = (current_portfolio_value - peak_value) / peak_value

**Why Include This?**
- Model needs "self-awareness" to make context-dependent decisions
- A weak signal means "ignore" when 80% cash, but "sell" when 100% invested
- Allows learning "profit protection" vs "loss cutting" strategies

**What We DON'T Include** (Prevent Overfitting):
- ❌ Entry price (model shouldn't anchor to purchase price)
- ❌ Holding duration (model shouldn't count days)
- ❌ Individual trade PnL (too noisy)

### 4.6 Cross-Asset Features (2 features): 

31. **crypto_correlation** = rolling_corr(BTC_returns, ETH_returns, 20)
32. **crypto_forex_divergence** = avg(crypto_momentum) - avg(forex_momentum)

**Total Features**: 78 + 3 + 6 + 8 + 2 = **97 features**

### 4.7 Normalization Strategy (Critical!)

**Problem**: BTC moves 5%/day, EUR/USD moves 0.5%/day. Without normalization, model ignores forex.

**Solution**: Asset-Specific Z-Score Normalization

```python
# CRITICAL: Calculate statistics on TRAINING data only (prevent leakage)
for asset in ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']:
    # Calculate per-feature statistics
    mean_dict[asset] = train_data[asset].mean(axis=0)
    std_dict[asset] = train_data[asset].std(axis=0)
    
    # Apply normalization
    normalized[asset] = (data[asset] - mean_dict[asset]) / std_dict[asset]

# Result: A 5% BTC move = ~1.0 std dev
#         A 0.5% EUR move = ~1.0 std dev
# Now they have equal "importance" to the neural network
```

**Reward Normalization** (Also Critical):
```python
# Calculate baseline volatility per asset (training data)
volatility_baseline = {
    'BTC': train_returns['BTC'].std(),  # ~0.03 (3% per 15-min)
    'EUR': train_returns['EUR'].std(),  # ~0.003 (0.3% per 15-min)
    # ...
}

# Normalize rewards
def calculate_reward(portfolio_return):
    # Raw return
    raw_reward = portfolio_return
    
    # Normalize by portfolio volatility
    portfolio_vol = weighted_avg([volatility_baseline[asset] 
                                   for asset in holdings])
    normalized_reward = raw_reward / portfolio_vol
    
    return normalized_reward
```

---

## 5. Reward Function Design (Simplified!)

### 5.1 Core Philosophy

**DO**: Reward outcomes (profit, risk-adjusted returns)  
**DON'T**: Reward behavior (setting stop-loss, trading in sessions, low risk)

### 5.2 Reward Formula

```python
# Step 1: Execute trades & deduct transaction costs
new_portfolio_value, fees_paid = execute_rebalance(action)

# Step 2: Calculate portfolio return
portfolio_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value

# Step 3: Calculate Sharpe component
sharpe_component = portfolio_return / portfolio_volatility

# Step 4: Combine (90% return, 10% Sharpe)
raw_reward = 0.9 * portfolio_return + 0.1 * sharpe_component

# Step 5: Normalize by volatility baseline (handle crypto vs forex scale)
portfolio_vol_baseline = sum([
    weights[asset] * volatility_baseline[asset] 
    for asset in holdings
])
normalized_reward = raw_reward / portfolio_vol_baseline

# Step 6: Scale to reasonable range for neural network
final_reward = normalized_reward * 100  # Scale to ±1.0 range typically
```

**That's it.** No bonuses, no penalties (except terminal conditions).

### 5.3 Terminal Penalties (Episode Ending)

**Bankruptcy**: Episode ends if `portfolio_value < 70% of initial_balance`
- Penalty: `-50.0` (large negative reward)
- Immediate episode termination

**Maximum Drawdown Breached**: Episode ends if `current_drawdown > 30%`
- Penalty: `-30.0`
- Immediate episode termination

### 5.4 Transaction Cost Impact

**Example Scenario**:
```
Model wants to rebalance:
- Sell $1000 BTC → Fee: $1 (0.1%)
- Buy $800 EUR → Fee: $0.40 (0.05%)
- Buy $200 Cash → Fee: $0

Total fees: $1.40
New portfolio value: $10,000 - $1.40 = $9,998.60

If the market doesn't move:
- Portfolio return = -$1.40 / $10,000 = -0.014%
- Reward = negative

Model learns: "Only rebalance if I expect to make > 0.014% profit"
```

This naturally prevents noise trading without explicit penalties.

---

## 6. Hard Constraints (Environment Enforcement)

### 6.1 Session Trading Constraints

**Crypto Assets** (BTC, ETH, SOL):
- Can trade 24/7
- **Low Liquidity Window**: Saturday 00:00-04:00 UTC
  - Not blocked, but action masking sets probability lower

**Forex Assets** (EUR, GBP, JPY):
- **Allowed Sessions**:
  - Tokyo: 00:00-09:00 GMT (Mon-Fri)
  - London: 08:00-16:00 GMT (Mon-Fri)
  - New York: 13:00-21:00 GMT (Mon-Fri)
- **Outside Sessions**: Masked by action masking (MaskablePPO)
- Weekend (Sat-Sun): Forex masked

**Implementation**:
```python
def get_action_mask(current_time):
    mask = np.ones(7, dtype=bool)
    
    # Crypto always available
    mask[0:3] = True  # BTC, ETH, SOL
    
    # Forex session-dependent
    mask[3] = is_forex_session_open('EUR', current_time)
    mask[4] = is_forex_session_open('GBP', current_time)
    mask[5] = is_forex_session_open('JPY', current_time)
    
    # Cash always available
    mask[6] = True
    
    return mask

def step(action):
    # Get current mask
    mask = get_action_mask(self.current_time)
    
    # MaskablePPO automatically respects mask
    # (forces probability of masked actions to 0)
    
    # Execute valid actions
    execute_rebalance(action, mask)
```

### 6.2 Risk Management Constraints

**Position Sizing**:
- Maximum single asset weight: **40%** (prevent concentration)
- Minimum position size: **$100** (avoid micro-positions)
- Minimum cash reserve: **5%** (maintain liquidity)

**Portfolio Risk Limit**:
- If `sum(risky_asset_weights) > 95%`: Force at least 5% cash
- Agent can go 100% cash, but cannot go 100% invested (must keep buffer)

**Stop-Loss & Take-Profit** (Automatic):
```python
# Monitor each position every step
for asset, position in open_positions.items():
    current_price = get_current_price(asset)
    entry_price = position.entry_price
    atr = get_atr(asset, period=14)
    
    # Calculate stop-loss (2× ATR below entry)
    stop_loss = entry_price - (2.0 * atr)
    
    # Calculate take-profit (3× ATR above entry, 1:1.5 R:R)
    take_profit = entry_price + (3.0 * atr)
    
    # Check if hit
    if current_price <= stop_loss:
        force_close_position(asset, reason='stop_loss')
    elif current_price >= take_profit:
        force_close_position(asset, reason='take_profit')
```

**Drawdown Protection**:
- Track peak portfolio value every step
- `current_drawdown = (peak - current) / peak`
- If `drawdown > 30%`: Force liquidate all + end episode

---

## 7. Training Strategy

### 7.1 Multi-Asset Simultaneous Training

**Critical**: All 6 assets + cash must be trained **together from day 1**.

**Vectorized Environment Setup**:
```python
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO

# Create 6 parallel environments (one per asset pair)
def make_env(asset, data):
    def _init():
        return TradingEnv(asset=asset, data=data)
    return _init

# Parallel environments for faster training
envs = [
    make_env('BTC', btc_data),
    make_env('ETH', eth_data),
    make_env('SOL', sol_data),
    make_env('EUR', eur_data),
    make_env('GBP', gbp_data),
    make_env('JPY', jpy_data)
]

# Use multiprocessing for true parallelism
vec_env = SubprocVecEnv(envs)

# RecurrentPPO with LSTM policy
model = RecurrentPPO(
    policy="MlpLstmPolicy",
    env=vec_env,
    verbose=1,
    tensorboard_log="./logs/"
)
```

**How It Works**:
1. Each training batch contains experiences from **all 6 assets**
2. LSTM learns universal sequential patterns across assets
3. No catastrophic forgetting (all assets in every update)

### 7.2 Training Phases

**Phase 1: Proof of Concept** (500K steps, ~6 hours)
- Train on 2 years (2020-2021)
- Test on 2022
- Goal: Achieve **any** positive return on test set

**Phase 2: Extended Training** (1M steps, ~12 hours)
- Train on 3 years (2020-2022)
- Test on 2023
- Goal: Sharpe ratio > 0.5 on test set

**Phase 3: Final Model** (2M steps, ~24 hours)
- Train on 4 years (2020-2023)
- Test on 2024 Q1-Q3
- Goal: Sharpe ratio > 1.0, Max Drawdown < 25%

**Phase 4: Walk-Forward Validation**
- Retrain model on each period
- Aggregate results across all test periods
- Goal: Consistent profitability across regimes

**Total Training Time**: ~42 hours (acceptable with Colab Pro)

### 7.3 Hyperparameters

```python
from sb3_contrib import RecurrentPPO, MaskablePPO

# Use RecurrentPPO with action masking
model = RecurrentPPO(
    policy="MlpLstmPolicy",
    env=vec_env,
    learning_rate=3e-4,        # Standard PPO
    n_steps=2048,              # Rollout buffer
    batch_size=64,             # Mini-batch size
    n_epochs=10,               # Training epochs per rollout
    gamma=0.99,                # Discount factor
    gae_lambda=0.95,           # GAE
    clip_range=0.2,            # PPO clipping
    ent_coef=0.01,             # Entropy (decay over time)
    vf_coef=0.5,               # Value function weight
    max_grad_norm=0.5,         # Gradient clipping
    policy_kwargs=dict(
        lstm_hidden_size=128,   # LSTM hidden units
        n_lstm_layers=2,        # Stack 2 LSTM layers
        enable_critic_lstm=True # Critic also uses LSTM
    ),
    verbose=1,
    tensorboard_log="./logs/",
    device="cuda"
)
```

**Entropy Decay Schedule**:
```python
# Encourage exploration early, exploitation later
# Step 0-250K: ent_coef = 0.02
# Step 250K-1M: ent_coef = 0.01
# Step 1M+: ent_coef = 0.005
```

### 7.4 Computational Requirements (Colab Pro)

**Hardware**: Google Colab Pro
- GPU: A100 (40GB) or V100 (16GB)
- RAM: 25GB available
- Session: Up to 24 hours continuous

**Memory Usage**:
- Training data (5 years, 15-min): ~600MB
- PPO buffer: ~300MB (includes LSTM states)
- Model parameters: ~80MB (LSTM is larger than MLP)
- LSTM hidden states: ~150MB
- **Total**: ~1.13GB ✅ (Well under 25GB limit)

**Training Speed**:
- LSTM is ~3× slower than MLP
- Phase 3 (2M steps): ~24 hours with A100
- Fits within Colab Pro session limits

**Checkpointing**:
```python
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,  # Save every 50K steps
    save_path='./models/',
    name_prefix='recurrent_ppo_trading'
)
```

---

## 8. Validation & Testing Protocol

### 8.1 During Training Validation

**EvalCallback** (Every 10K Steps):
```python
from stable_baselines3.common.callbacks import EvalCallback

eval_env = TradingEnv(asset='MULTI', data=validation_data)

eval_callback = EvalCallback(
    eval_env=eval_env,
    best_model_save_path='./models/best/',
    log_path='./logs/eval/',
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True  # No exploration during eval
)
```

**Metrics Tracked**:
- Average Episode Return
- Sharpe Ratio (rolling 20 episodes)
- Maximum Drawdown
- Win Rate (% profitable episodes)
- Average Cash Allocation (is model using cash position?)

### 8.2 Walk-Forward Backtesting

**Test 1**: Train 2020-2021 → Test 2022  
**Test 2**: Train 2020-2022 → Test 2023  
**Test 3**: Train 2020-2023 → Test 2024 Q1-Q3

**Aggregate Metrics**:
```
Average Return across all tests: ___%
Average Sharpe across all tests: ___
Worst Drawdown across all tests: ___%
Consistency: ___% of tests profitable
Cash Usage: Average % time in cash
```

**Pass Criteria**:
- ✅ Positive returns in ≥2 out of 3 tests
- ✅ Average Sharpe > 0.5
- ✅ Max drawdown < 30% in all tests
- ✅ Model uses cash position (not always 100% invested)

### 8.3 Stress Testing

**Specific Crisis Periods**:
1. **March 2020**: COVID crash (-50% BTC in 2 days)
2. **May 2021**: China mining ban (-40% BTC)
3. **Nov 2022**: FTX collapse (-20% BTC)

**Goal**: Model must survive with <30% portfolio drawdown in each crisis.

**Critical Test**: Did the model go to cash before/during the crash? Or did it ride it down?

### 8.4 Paper Trading (Live Simulation)

**Phase 1: Silent Mode** (1 month)
- Run model on live data stream (Alpaca Paper API)
- Log intended trades but don't execute
- Compare: What model wanted vs what would've happened

**Phase 2: Live Paper Trading** (2 months)
- Execute trades on paper account ($10,000 simulated)
- Monitor:
  - Execution slippage (intended vs actual price)
  - Latency (signal to execution time)
  - API errors/disconnections
  - Transaction costs (actual fees paid)
- Goal: Performance matches backtest within ±15%

**Success Criteria**:
- ✅ Positive returns over 3 months
- ✅ Sharpe > 0.5
- ✅ Drawdown < 25%
- ✅ No critical bugs
- ✅ Transaction costs < 1% of returns (not eating all profit)

---

## 9. Performance Metrics & KPIs

### 9.1 Primary Metrics

**Profitability**:
- **Total Return %**: (Final - Initial) / Initial
  - Minimum Acceptable: +10% per year
  - Target: +20% per year
- **Sharpe Ratio**: (Avg Return - Risk Free) / Std Dev
  - Minimum: 0.5
  - Target: 1.0+
- **Sortino Ratio**: (Avg Return) / Downside Deviation
  - Target: 1.5+

**Risk**:
- **Maximum Drawdown**: Largest peak-to-trough decline
  - Hard Limit: 30%
  - Target: <20%
- **Calmar Ratio**: Annual Return / Max Drawdown
  - Target: >1.0
- **Time in Drawdown**: % of time below previous peak
  - Target: <50%

**Trading Behavior**:
- **Win Rate**: Profitable Trades / Total Trades
  - Target: >40% (with good R:R)
- **Profit Factor**: Gross Profit / Gross Loss
  - Target: >1.3
- **Average R:R**: Take Profit / Stop Loss
  - Target: >1.5
- **Cash Usage**: % of time with >20% cash allocation
  - Target: >30% (model uses risk-off capability)
- **Transaction Cost Ratio**: Total Fees / Gross Returns
  - Target: <2% (fees not eating profit)

### 9.2 Monitoring Dashboards

**TensorBoard** (Real-Time During Training):
- Episode reward (moving average)
- Sharpe ratio (rolling)
- Drawdown progression
- Action distribution (portfolio weights over time)
- Cash allocation % (is model sitting out bad markets?)
- LSTM hidden state activations (what patterns is it detecting?)

**Backtest Report** (Post-Training):
- Equity curve
- Drawdown chart
- Win/Loss distribution
- Monthly returns heatmap
- Asset allocation over time (stacked area chart)
- Cash usage timeline (when did model go defensive?)

---

## 10. Implementation Roadmap

### Week 1-2: Foundation & Data
- [ ] Set up Google Colab Pro (A100 GPU)
- [ ] Install: `sb3-contrib`, `stable-baselines3`, `gymnasium`, `pandas-ta`
- [ ] Fetch 5 years of 15-min data for all 6 assets (2020-2024)
- [ ] Calculate 4H and Daily indicators from 15-min data
- [ ] Implement feature engineering pipeline (97 features)
- [ ] Verify: Asset-specific z-score normalization
- [ ] Implement volatility-based reward normalization

### Week 3: Environment Development
- [ ] Build `TradingEnv(gymnasium.Env)` with 7-asset action space
- [ ] Implement continuous action space (portfolio weights + cash)
- [ ] Add action masking logic (session awareness)
- [ ] Implement transaction cost deduction (0.1% crypto, 0.05% forex)
- [ ] Add hard constraints (max position 40%, min cash 5%)
- [ ] Implement automatic stop-loss & take-profit (2× and 3× ATR)
- [ ] Test environment (random actions, verify fees and constraints)

### Week 4: Reward & LSTM Setup
- [ ] Implement simplified reward (90% return, 10% Sharpe)
- [ ] Add volatility normalization to reward calculation
- [ ] Add terminal penalties (bankruptcy -50, drawdown -30)
- [ ] Configure RecurrentPPO with LSTM policy
- [ ] Set up LSTM hidden size (128) and layers (2)
- [ ] Test LSTM forward pass (verify shapes and memory)

### Week 5-6: Training Phase 1 & 2
- [ ] Create train/test split (walk-forward)
- [ ] Set up EvalCallback for validation
- [ ] Train Phase 1: 500K steps on 2020-2021 (~6 hours)
- [ ] Backtest on 2022 (evaluate cash usage, transaction costs)
- [ ] Analyze: Is model using cash position during downturns?
- [ ] Train Phase 2: 1M steps on 2020-2022 (~12 hours)
- [ ] Backtest on 2023 (evaluate)
- [ ] Check TensorBoard: Are LSTM patterns emerging?

### Week 7-8: Training Phase 3 & Walk-Forward
- [ ] Train Phase 3: 2M steps on 2020-2023 (~24 hours)
- [ ] Backtest on 2024 Q1-Q3
- [ ] Run all 3 walk-forward tests
- [ ] Aggregate results, calculate average Sharpe across periods
- [ ] Analyze LSTM attention: What patterns does it remember?

### Week 9-10: Analysis & Optimization
- [ ] Feature importance analysis (which timeframes matter most?)
- [ ] Analyze cash allocation patterns (when does model go defensive?)
- [ ] Check transaction cost impact (are fees eating profit?)
- [ ] Hyperparameter tuning (learning rate, LSTM hidden size)
- [ ] Stress test on crisis periods (March 2020, May 2021, Nov 2022)
- [ ] Final model selection (best checkpoint from walk-forward)

### Week 11-14: Paper Trading
- [ ] Deploy silent paper trading (1 month, log-only mode)
- [ ] Deploy live paper trading on Alpaca (1 month, real execution)
- [ ] Monitor daily:
  - Portfolio value progression
  - Cash allocation % (is it dynamic?)
  - Transaction costs (actual fees paid)
  - Execution slippage
- [ ] Compare backtest vs live results (±15% acceptable)
- [ ] Debug any discrepancies (look-ahead bias? Fee calculation wrong?)

### Week 15-16: Production Preparation
- [ ] Build real-time monitoring dashboard (Streamlit/Plotly)
  - Live equity curve
  - Current portfolio allocation (pie chart)
  - Cash % over time (line chart)
  - Drawdown from peak
- [ ] Implement alert system:
  - Email if drawdown > 15%
  - Email if cash allocation < 5% (model stuck fully invested?)
  - Email if transaction costs > 2% of returns
- [ ] Document full model architecture and training process
- [ ] Final safety checks:
  - Verify stop-losses are working
  - Verify action masking prevents forex trading on weekends
  - Verify transaction costs are deducted correctly
- [ ] Create model deployment script

---

## 11. Risk Mitigation

### 11.1 Technical Risks

| Risk | Mitigation |
|------|-----------|
| LSTM overfits to training sequences | Walk-forward validation across 3 periods |
| Colab session timeout during training | Checkpoint every 50K steps, resume from latest |
| Out of memory (LSTM states) | Colab Pro with 25GB RAM, batch_size=64 |
| Catastrophic forgetting | Train all assets simultaneously (VecEnv) |
| Reward hacking | Simple profit-based reward, hard constraints |
| Model ignores cash position | Monitor cash usage %, require >20% in validation |
| LSTM gradient vanishing | Gradient clipping (max_norm=0.5), LSTM cell design handles this |

### 11.2 Market Risks

| Risk | Mitigation |
|------|-----------|
| Black swan event (>50% crash) | 30% max drawdown limit, cash position available |
| Flash crash (sudden spike/drop) | Automatic stop-losses at 2× ATR |
| Slippage on live execution | Paper trade 3 months, measure actual vs intended |
| Model fails on new regime | Retrain quarterly on new data, walk-forward validates regime change |
| Transaction costs eat all profit | Explicit cost deduction in training, monitor cost ratio <2% |
| Overtrading (high fees) | Transaction costs naturally penalize noise trading |

### 11.3 Validation Risks

| Risk | Mitigation |
|------|-----------|
| Backtest looks good, live fails | Walk-forward validation (3 independent tests) |
| Overfitting to 2020-2023 | Out-of-sample test on 2024 + stress tests |
| Data snooping bias | Strict chronological splits, no peeking at test data |
| Look-ahead bias in multi-timeframe | Calculate Daily/4H indicators only from past data |
| Model learns to exploit test set | Use completely unseen 2024 Q4 data for final validation |

---

## 12. Success Criteria

### 12.1 Training Success (Proof of Concept)
- ✅ Model converges (reward increases over time)
- ✅ Positive returns on ≥2/3 walk-forward tests
- ✅ Sharpe ratio > 0.5 on validation data
- ✅ Model actively uses cash position (>20% average allocation)
- ✅ LSTM learns temporal patterns (observable in hidden state activations)

### 12.2 Backtest Success (Production-Ready)
- ✅ Total return > +15% per year
- ✅ Sharpe ratio > 1.0
- ✅ Max drawdown < 25%
- ✅ Survives all 3 stress test crashes (drawdown <30%)
- ✅ Transaction costs < 2% of gross returns
- ✅ Model goes to cash during crashes (observable in allocation timeline)
- ✅ Win rate > 40% with avg R:R > 1.5

### 12.3 Paper Trading Success (Live Validation)
- ✅ Positive returns over 3 months
- ✅ Performance within ±15% of backtest
- ✅ No execution errors or bugs
- ✅ Actual transaction costs match simulated costs
- ✅ Slippage < 0.05% on average
- ✅ Cash allocation behaves as expected (dynamic, not stuck at 0% or 100%)

### 12.4 Live Trading Readiness (Future Deployment)
- ✅ 6+ months profitable paper trading
- ✅ Consistent Sharpe > 1.0 month-over-month
- ✅ Max drawdown never exceeded 20% in live paper trading
- ✅ Model demonstrably goes defensive (cash) before major drawdowns
- ✅ Transaction costs remain manageable (<2% of returns)

---

## 13. Tools & Technologies

**Core Framework**:
- Python 3.10+
- **sb3-contrib** (RecurrentPPO, MaskablePPO)
- **Stable-Baselines3** (base RL framework)
- **Gymnasium** (environment interface)
- **PyTorch** (LSTM backend)

**Data & Analysis**:
- **Pandas** (data manipulation)
- **Pandas-TA** (technical indicators - RSI, MACD, ATR, etc.)
- **NumPy** (numerical computing)
- **Matplotlib/Plotly** (visualization)

**Data Sources**:
- **Alpaca API** (primary - crypto & forex, paper trading)
- **Yahoo Finance** (backup historical data)
- **Binance API** (alternative crypto data source)

**Infrastructure**:
- **Google Colab Pro** (A100/V100 GPU, 25GB RAM)
- **Google Drive** (model checkpoints, training logs)
- **TensorBoard** (training monitoring, LSTM visualization)

**Monitoring & Alerts** (Production):
- **Streamlit** (real-time dashboard)
- **Plotly Dash** (alternative dashboard)
- **SMTP/SendGrid** (email alerts)

---

## 14. Key Design Decisions Summary

### ✅ Critical Fixes from v2.0

1. **7th Action (Cash)**: Action space now includes cash position - model can sit out bad markets ✅
2. **Transaction Costs**: Explicitly deducted before reward calculation - prevents noise trading ✅
3. **Time Features**: sin/cos hour + day_of_week - model understands session timing ✅
4. **Multi-Timeframe**: Daily EMA-200 + 4H RSI - gives model "peripheral vision" ✅
5. **Portfolio State**: Current weights + unrealized PnL - model has "self-awareness" ✅
6. **Action Masking**: MaskablePPO prevents illegal forex trades - faster convergence ✅
7. **LSTM Architecture**: RecurrentPPO with 2-layer LSTM - temporal pattern recognition ✅

### ✅ Why LSTM Over MLP

**Decision**: Use LSTM from the start (not "wait for MLP to fail")

**Justification**:
1. **Trading is Sequential**: Patterns form over time (consolidation → breakout)
2. **Proven in Literature**: LSTMs consistently outperform MLPs in time-series prediction
3. **Colab Pro Available**: 25GB RAM + A100 GPU handles LSTM computational cost
4. **3× Time is Acceptable**: 24 hours vs 8 hours for better performance is worth it
5. **Market Memory Matters**: "3 down days in a row" is different from "random down days"

**Trade-off Accepted**: Longer training time (42 hours total) for better temporal understanding

### ✅ Feature Count: 97 Total

- **Market Features**: 78 (15-min, 4H, Daily indicators across 6 assets)
- **Time Features**: 3 (sin/cos hour, day of week)
- **Session Status**: 6 (is_tradeable per asset)
- **Portfolio State**: 8 (weights + unrealized PnL)
- **Cross-Asset**: 2 (correlations, divergence)

**RAM Usage**: 97 features × 6 assets × 175K rows ≈ **900MB** (fits in Colab Pro)

---

## 15. Failure Modes & Debugging

### 15.1 Common RL Trading Failures (How We Prevent Them)

| Failure Mode | How We Prevent It |
|--------------|-------------------|
| **Reward Hacking** | Simple profit-based reward, no bonuses to game |
| **Overfitting to Bull Market** | Train on 5 years including 2022 bear market |
| **Ignoring Transaction Costs** | Explicitly deduct fees before reward calculation |
| **Always 100% Invested** | Cash is explicit action, model can go defensive |
| **Trading Noise** | Transaction costs penalize frequent rebalancing |
| **Session Violations** | Action masking prevents illegal trades |
| **Catastrophic Forgetting** | All assets trained simultaneously |
| **Forex Ignored (Volatility Mismatch)** | Asset-specific normalization + reward normalization |
| **No Temporal Memory** | LSTM captures sequential patterns |
| **Blind to Macro Trends** | Daily EMA-200 and 4H indicators included |

### 15.2 Debugging Checklist (If Performance is Poor)

**If model makes no trades**:
- [ ] Check action masking (are all actions masked?)
- [ ] Check transaction costs (are they too high?)
- [ ] Check reward scale (is it too small to be meaningful?)

**If model always stays in cash**:
- [ ] Check if cash action is being favored (action distribution)
- [ ] Check if rewards are positive for holding assets
- [ ] Reduce transaction costs temporarily to encourage exploration

**If model always stays 100% invested**:
- [ ] Check if cash action is masked incorrectly
- [ ] Check portfolio state features (is cash weight visible?)
- [ ] Increase drawdown penalty to encourage defensive positioning

**If model only trades crypto (ignores forex)**:
- [ ] Verify volatility normalization is working
- [ ] Check reward normalization (forex returns scaled up?)
- [ ] Check action masking (is forex always blocked?)

**If model performance diverges in paper trading**:
- [ ] Measure actual transaction costs vs simulated
- [ ] Check for look-ahead bias in feature calculation
- [ ] Verify execution timing (are we trading at open or close?)
- [ ] Check slippage impact (intended vs actual prices)

**If LSTM is not learning**:
- [ ] Verify gradient flow (check TensorBoard gradients)
- [ ] Reduce sequence length if gradient vanishing
- [ ] Check LSTM hidden state dimensions
- [ ] Verify that episodes are long enough for LSTM to matter

---

## 16. Post-Deployment (Future)

### 16.1 Continuous Improvement

**Monthly Reviews**:
- Analyze last 30 days of performance
- Compare to backtest expectations
- Check if market regime has changed
- Review transaction cost ratio

**Quarterly Retraining**:
- Add last 3 months of data to training set
- Retrain model (walk-forward style)
- Validate on next month
- Deploy if performance improves

**Annual Architecture Review**:
- Evaluate new RL algorithms (SAC, TD3, DreamerV3)
- Consider adding Transformer layers
- Review feature importance (drop weak features, add new ones)
- Benchmark against buy-and-hold baseline

### 16.2 Scaling Strategy (Future)

**Phase 1**: Current system (6 assets + cash, $10K capital)
**Phase 2**: Add more assets (10-15 pairs, $50K capital)
**Phase 3**: Multi-strategy ensemble (3-5 models with different hyperparameters)
**Phase 4**: Hierarchical RL (meta-agent allocates to sub-agents per asset)

---

## 17. Appendix: Technical Details

### 17.1 LSTM Architecture Specifics

**Policy Network**:
```
Input (97 features)
  ↓
LSTM Layer 1 (128 units, return_sequences=True)
  ↓
Dropout(0.2)
  ↓
LSTM Layer 2 (128 units, return_sequences=False)
  ↓
Dense(256, ReLU)
  ↓
Dense(128, ReLU)
  ↓
Actor Head: Dense(7, Softmax) → Portfolio weights
```

**Value Network** (Critic):
```
Input (97 features)
  ↓
LSTM Layer 1 (128 units, return_sequences=True)
  ↓
LSTM Layer 2 (128 units, return_sequences=False)
  ↓
Dense(256, ReLU)
  ↓
Dense(128, ReLU)
  ↓
Value Head: Dense(1, Linear) → State value
```

**Why 2 LSTM Layers?**
- Layer 1: Captures short-term patterns (last 5-10 candles)
- Layer 2: Captures longer-term patterns (last 20-30 candles)
- Stacking allows hierarchical temporal abstraction

### 17.2 Session Definitions (Precise)

**Tokyo Session**:
- Open: 00:00 GMT (09:00 JST)
- Close: 09:00 GMT (18:00 JST)
- Mon-Fri only

**London Session**:
- Open: 08:00 GMT
- Close: 16:00 GMT
- Mon-Fri only

**New York Session**:
- Open: 13:00 GMT (08:00 EST)
- Close: 21:00 GMT (16:00 EST)
- Mon-Fri only

**High Liquidity Periods** (Session Overlaps):
- London-NY Overlap: 13:00-16:00 GMT (highest volume)
- Tokyo-London Overlap: 08:00-09:00 GMT

**Low Liquidity Periods**:
- Crypto: Saturday 00:00-04:00 UTC (weekend, low volume)
- Forex: All weekend (Sat-Sun), closed

### 17.3 Transaction Cost Schedule

| Asset | Fee Type | Cost |
|-------|----------|------|
| BTC/USD | Taker Fee | 0.10% |
| ETH/USD | Taker Fee | 0.10% |
| SOL/USD | Taker Fee | 0.10% |
| EUR/USD | Spread | 0.05% |
| GBP/USD | Spread | 0.05% |
| USD/JPY | Spread | 0.05% |
| Cash | Transfer | 0.00% |

**Note**: These are conservative estimates. Actual costs may be lower with maker orders or premium accounts.

### 17.4 Data Pipeline Pseudocode

```python
# Step 1: Fetch raw data
raw_data = fetch_ohlcv(asset='BTC', timeframe='15m', start='2020-01-01', end='2024-12-31')

# Step 2: Calculate 15-min indicators
data['returns'] = np.log(data['close'] / data['close'].shift(1))
data['rsi_14'] = ta.rsi(data['close'], length=14) / 100
data['atr_14'] = ta.atr(data['high'], data['low'], data['close'], length=14) / data['close']
# ... (8 features total)

# Step 3: Resample to 4H and Daily
data_4h = data.resample('4H', closed='left', label='left').last()
data_daily = data.resample('1D', closed='left', label='left').last()

# Step 4: Calculate higher timeframe indicators
data_4h['rsi_14'] = ta.rsi(data_4h['close'], length=14) / 100
data_daily['ema_200'] = ta.ema(data_daily['close'], length=200)

# Step 5: Merge back to 15-min (forward-fill)
data['4h_rsi'] = data_4h['rsi_14'].reindex(data.index, method='ffill')
data['daily_ema_200'] = data_daily['ema_200'].reindex(data.index, method='ffill')

# Step 6: Calculate distances
data['dist_from_daily_ema'] = (data['close'] - data['daily_ema_200']) / data['daily_ema_200']

# Step 7: Normalize per asset (z-score)
train_mean = data.loc[train_idx].mean()
train_std = data.loc[train_idx].std()
data_normalized = (data - train_mean) / train_std

# Step 8: Add time features
data['sin_hour'] = np.sin(data.index.hour * 2 * np.pi / 24)
data['cos_hour'] = np.cos(data.index.hour * 2 * np.pi / 24)
data['day_of_week'] = data.index.dayofweek

# Step 9: Add session status
data['is_eur_tradeable'] = data.apply(lambda row: is_forex_session_open('EUR', row.name), axis=1)

# Step 10: Final feature vector (97 features per row)
feature_vector = data[feature_columns].values  # Shape: (N_rows, 97)
```

---

## 18. Conclusion

This RPD represents a **production-ready** AI trading system designed to avoid all known failure modes:

✅ **No reward hacking** (simple profit-based reward)  
✅ **No catastrophic forgetting** (simultaneous multi-asset training)  
✅ **No volatility mismatch** (asset-specific normalization)  
✅ **No transaction cost blindness** (explicit fee deduction)  
✅ **No forced investment** (cash is explicit action)  
✅ **No session violations** (action masking)  
✅ **No temporal blindness** (LSTM memory)  
✅ **No macro blindness** (multi-timeframe features)  

**Expected Outcome**: A trading agent that:
- Learns to allocate capital dynamically across 6 assets + cash
- Goes defensive (cash) during market downturns
- Uses temporal patterns (LSTM) to predict regime changes
- Respects transaction costs (doesn't overtrade)
- Trades only during appropriate sessions
- Achieves Sharpe > 1.0 with <25% maximum drawdown

**Total Development Time**: 16 weeks (~4 months)  
**Total Training Time**: 42 hours (Colab Pro)  
**Final Model Size**: ~80MB (deployable)

---

**Document Version**: 3.0 (LSTM + All Critical Fixes)  
**Last Updated**: November 23, 2025  
**Status**: Ready for Implementation ✅

**Next Step**: Begin Week 1 (Foundation & Data Pipeline)
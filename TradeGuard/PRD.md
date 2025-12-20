# TradeGuard: Meta-Labeling Trade Filter

## Product Requirements Document (PRD)
**Version:** 1.0  
**Date:** 2025-12-20  
**Author:** AI Assistant  

---

## 1. Executive Summary

### 1.1 Problem Statement
The Alpha and Risk models have been trained and backtested with satisfactory results (positive Sharpe Ratio, good Risk-Reward). However, there are still **significant losses** occurring in the trading system. Instead of modifying the core models, we will add a **secondary filtering layer** that predicts whether a given trade signal is likely to result in a Win or Loss.

### 1.2 Solution
Implement a **Meta-Labeling** model called **TradeGuard** that:
- Acts as a binary classifier (Win vs. Loss)
- Receives trade signals from the Alpha model
- Outputs a **confidence score** (0.0 to 1.0)
- Only allows trades with confidence above a threshold (e.g., >0.65) to execute

### 1.3 Expected Outcomes
- Reduced drawdowns by filtering out low-probability trades
- Improved Sharpe Ratio by reducing false signals
- Preserved capital during adverse market conditions (news, low liquidity)

---

## 2. Model Architecture

### 2.1 Model Selection: LightGBM

**Chosen Model:** LightGBM (Light Gradient Boosting Machine)

**Rationale (vs. XGBoost):**

| Criteria | LightGBM | XGBoost |
|----------|----------|---------|
| **Training Speed** | 3-5x faster | Baseline |
| **Memory Usage** | Lower (Histogram-based) | Higher |
| **Large Dataset (2M+ rows)** | Optimized (GOSS, EFB) | Can struggle |
| **Categorical Features** | Native support | Requires encoding |
| **Accuracy** | Comparable | Comparable |
| **Overfitting Risk** | Slightly higher (leaf-wise) | Lower (level-wise) |

**Key LightGBM Features:**
- **Leaf-wise Tree Growth:** Finds complex patterns in noisy financial data
- **Histogram Algorithm:** Reduces computation during split finding
- **GOSS (Gradient-based One-Side Sampling):** Focuses on high-error samples
- **EFB (Exclusive Feature Bundling):** Optimizes memory for sparse features

**Hyperparameter Starting Point:**
```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'is_unbalance': True,  # Handle Win/Loss imbalance
    'seed': 42
}
```

---

## 3. Dataset Specification

### 3.1 Data Source
- **Historical Data:** 2016-2024 (8 years)
- **Timeframe:** 5-minute candles
- **Assets:** EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD
- **Estimated Rows:** ~2,000,000 trade signals

### 3.2 Label Generation (Ground Truth)

Each trade signal from the Alpha model will be labeled as:

| Label | Value | Definition |
|-------|-------|------------|
| **WIN** | 1 | Trade hit Take-Profit before Stop-Loss |
| **LOSS** | 0 | Trade hit Stop-Loss first, or timed out at loss |

**Labeling Logic:**
```python
def label_trade(entry_price, direction, sl, tp, future_highs, future_lows):
    """
    Simulates trade outcome using lookahead data.
    Returns: 1 (Win) or 0 (Loss)
    """
    for i in range(len(future_highs)):
        if direction == 1:  # Long
            if future_lows[i] <= sl:
                return 0  # Loss
            if future_highs[i] >= tp:
                return 1  # Win
        else:  # Short
            if future_highs[i] >= sl:
                return 0  # Loss
            if future_lows[i] <= tp:
                return 1  # Win
    
    # Timed out - check final P&L
    final_price = future_closes[-1]
    pnl = (final_price - entry_price) * direction
    return 1 if pnl > 0 else 0
```

---

## 4. Feature Engineering (60 Features)

### 4.1 Feature Categories Overview

| Category | Count | Description |
|----------|-------|-------------|
| **Alpha Model Confidence** | 10 | Model's internal state at signal time |
| **Synthetic News Proxies** | 10 | Volatility/volume anomalies that detect news |
| **Market Regime** | 10 | Trending vs. Mean-reverting detection |
| **Session Edge** | 10 | Time-based liquidity patterns |
| **Execution Statistics** | 10 | Entry quality and risk metrics |
| **Price Action Context** | 10 | Candle structure and price levels |
| **TOTAL** | **60** | |

---

### 4.2 Feature Group A: Alpha Model Confidence (10 Features)

These features capture how "confident" the Alpha PPO model was when generating the signal.

| # | Feature Name | Description | Calculation |
|---|--------------|-------------|-------------|
| 1 | `alpha_action_raw` | Raw action output for the asset | `action[asset_idx]` |
| 2 | `alpha_action_abs` | Absolute strength of signal | `abs(action[asset_idx])` |
| 3 | `alpha_action_std` | Volatility of recent actions (last 5) | `std(actions[-5:])` |
| 4 | `alpha_signal_persistence` | Consecutive candles with same signal | Count of same direction |
| 5 | `alpha_signal_reversal` | Signal just flipped direction | Binary (0/1) |
| 6 | `alpha_portfolio_drawdown` | Current drawdown at signal time | `1 - (equity / peak_equity)` |
| 7 | `alpha_open_positions` | Number of other open positions | Count |
| 8 | `alpha_margin_usage` | Current margin utilization | `total_exposure / equity` |
| 9 | `alpha_recent_win_rate` | Win rate of last 10 trades | Rolling calculation |
| 10 | `alpha_recent_pnl` | P&L of last 10 trades | Rolling sum |

**Note:** Full logit/entropy extraction requires modifying the SB3 predict call. For MVP, we use observable state features.

---

### 4.3 Feature Group B: Synthetic News Proxies (10 Features)

These features detect "news-like" market conditions using only OHLCV data.

| # | Feature Name | Description | Calculation |
|---|--------------|-------------|-------------|
| 11 | `volume_ratio` | Current volume vs. 50-period average | `volume / volume.rolling(50).mean()` |
| 12 | `volume_zscore` | Standardized volume anomaly | `(volume - mean) / std` |
| 13 | `range_ratio` | Current candle range vs. average | `(high - low) / atr_14` |
| 14 | `range_compression` | Is range in bottom 10%? | Binary: `range < percentile_10` |
| 15 | `body_to_range` | Candle body vs. total range | `abs(close - open) / (high - low)` |
| 16 | `wick_ratio` | Total wick length vs. body | `(upper_wick + lower_wick) / body` |
| 17 | `gap_size` | Gap from previous close | `abs(open - prev_close) / atr` |
| 18 | `tick_surge` | Volume acceleration | `volume.diff() / volume.rolling(10).mean()` |
| 19 | `volatility_regime` | ATR percentile (0-100) | `atr.rolling(200).rank(pct=True)` |
| 20 | `quiet_market` | Extremely low volatility flag | Binary: `atr < atr.rolling(100).quantile(0.05)` |

---

### 4.4 Feature Group C: Market Regime (10 Features)

These features help the model understand the current market "mode."

| # | Feature Name | Description | Calculation |
|---|--------------|-------------|-------------|
| 21 | `hurst_exponent` | Trending (>0.5) vs. Mean-reverting (<0.5) | Rolling Hurst calculation |
| 22 | `adx_value` | Trend strength (0-100) | ADX indicator |
| 23 | `adx_slope` | Is trend strengthening or weakening? | `adx.diff(5)` |
| 24 | `price_vs_sma200` | Distance from 200-SMA in % | `(close - sma_200) / sma_200` |
| 25 | `sma_slope_200` | Long-term trend direction | `sma_200.diff(20) / sma_200` |
| 26 | `correlation_eurusd` | Current correlation with EURUSD | Rolling 50-period correlation |
| 27 | `correlation_xauusd` | Current correlation with Gold | Rolling 50-period correlation |
| 28 | `basket_direction` | Are most assets going same way? | Mean of all asset returns |
| 29 | `dispersion` | How spread out are asset returns? | Std of all asset returns |
| 30 | `risk_on_score` | Risk-on/Risk-off proxy | `(gbpusd_ret + xauusd_ret) / 2` |

---

### 4.5 Feature Group D: Session Edge (10 Features)

These features capture time-based liquidity and behavior patterns.

| # | Feature Name | Description | Calculation |
|---|--------------|-------------|-------------|
| 31 | `hour_sin` | Cyclical hour encoding | `sin(2π * hour / 24)` |
| 32 | `hour_cos` | Cyclical hour encoding | `cos(2π * hour / 24)` |
| 33 | `day_sin` | Cyclical day encoding | `sin(2π * dayofweek / 7)` |
| 34 | `day_cos` | Cyclical day encoding | `cos(2π * dayofweek / 7)` |
| 35 | `session_asian` | Is Asian session? | Binary (00:00-09:00 UTC) |
| 36 | `session_london` | Is London session? | Binary (08:00-17:00 UTC) |
| 37 | `session_ny` | Is New York session? | Binary (13:00-22:00 UTC) |
| 38 | `session_overlap` | Is London/NY overlap? | Binary (13:00-17:00 UTC) |
| 39 | `minutes_to_session_change` | Time until next session boundary | Calculated from hour |
| 40 | `friday_risk` | Distance to weekend (Friday close) | Higher on Friday afternoon |

---

### 4.6 Feature Group E: Execution Statistics (10 Features)

These features assess the quality and risk of the specific trade entry.

| # | Feature Name | Description | Calculation |
|---|--------------|-------------|-------------|
| 41 | `entry_atr_distance` | How far has price moved already? | `abs(close - open_12bars) / atr` |
| 42 | `sl_distance_atr` | Stop-loss distance in ATR units | `sl_pips / atr` |
| 43 | `tp_distance_atr` | Take-profit distance in ATR units | `tp_pips / atr` |
| 44 | `risk_reward_ratio` | Reward vs. Risk | `tp_distance / sl_distance` |
| 45 | `position_size_pct` | Size relative to equity | `position_value / equity` |
| 46 | `current_drawdown` | Portfolio drawdown at entry | `1 - (equity / peak_equity)` |
| 47 | `spread_estimate` | Estimated spread from OHLC | `2 * (high - low) / (high + low)` |
| 48 | `momentum_at_entry` | RSI at entry time | RSI-14 value |
| 49 | `bb_position` | Bollinger Band position | `(close - lower) / (upper - lower)` |
| 50 | `macd_histogram` | MACD momentum | MACD histogram value |

---

### 4.7 Feature Group F: Price Action Context (10 Features)

These features provide structural context around the entry point.

| # | Feature Name | Description | Calculation |
|---|--------------|-------------|-------------|
| 51 | `candle_direction` | Is current candle bullish? | Binary: `close > open` |
| 52 | `candle_body_size` | Body size relative to ATR | `abs(close - open) / atr` |
| 53 | `upper_wick_size` | Upper wick relative to body | `(high - max(open, close)) / body` |
| 54 | `lower_wick_size` | Lower wick relative to body | `(min(open, close) - low) / body` |
| 55 | `consecutive_direction` | Consecutive same-direction candles | Count |
| 56 | `distance_from_high_20` | Distance from 20-period high | `(high_20 - close) / atr` |
| 57 | `distance_from_low_20` | Distance from 20-period low | `(close - low_20) / atr` |
| 58 | `ema_alignment` | EMA9 vs EMA21 alignment | `(ema_9 - ema_21) / ema_21` |
| 59 | `price_velocity` | Rate of price change | `close.diff(5) / atr` |
| 60 | `volume_price_trend` | Volume-weighted price trend | VPT indicator |

---

## 5. Training Pipeline

### 5.1 Data Generation Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATASET GENERATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load Historical OHLCV Data (2016-2024)                      │
│                         ↓                                       │
│  2. Load Trained Alpha Model (PPO)                              │
│                         ↓                                       │
│  3. Run Alpha Model Through All Data                            │
│     - For each timestep, get action output                      │
│     - If action triggers signal (>0.33 or <-0.33):              │
│       → Record entry point                                      │
│       → Calculate all 60 features                               │
│       → Simulate trade outcome (label)                          │
│                         ↓                                       │
│  4. Save Dataset as Parquet                                     │
│     - Features: 60 columns                                      │
│     - Label: 1 column (win/loss)                                │
│     - Metadata: timestamp, asset, direction                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Training Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Load Generated Dataset                                      │
│                         ↓                                       │
│  2. Train/Validation Split (80/20, time-based)                  │
│     - Train: 2016-2022                                          │
│     - Validation: 2023-2024                                     │
│                         ↓                                       │
│  3. Handle Class Imbalance                                      │
│     - Use `is_unbalance=True` or SMOTE                          │
│                         ↓                                       │
│  4. Train LightGBM Model                                        │
│     - Early stopping on validation AUC                          │
│     - Cross-validation for hyperparameter tuning                │
│                         ↓                                       │
│  5. Evaluate on Validation Set                                  │
│     - Metrics: AUC, Precision, Recall, F1                       │
│     - Calibration curve analysis                                │
│                         ↓                                       │
│  6. Determine Optimal Threshold                                 │
│     - Trade-off: Precision vs. Trade Frequency                  │
│     - Target: >60% precision with acceptable recall             │
│                         ↓                                       │
│  7. Save Model and Threshold                                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Integration Architecture

### 6.1 Live Trading Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     LIVE TRADING FLOW                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Market Data (5m Candle)                                        │
│           ↓                                                     │
│  ┌───────────────────┐                                          │
│  │   ALPHA MODEL     │  → Action: BUY EURUSD                    │
│  │   (PPO Agent)     │                                          │
│  └───────────────────┘                                          │
│           ↓                                                     │
│  ┌───────────────────┐                                          │
│  │   RISK LAYER      │  → Position Size: 0.5 lots               │
│  │   (PPO Agent)     │  → SL: 1.5 ATR, TP: 2.5 ATR              │
│  └───────────────────┘                                          │
│           ↓                                                     │
│  ┌───────────────────┐                                          │
│  │   TRADE GUARD     │  → Confidence: 0.72                      │
│  │   (LightGBM)      │  → Decision: EXECUTE ✓                   │
│  └───────────────────┘                                          │
│           ↓                                                     │
│  ┌───────────────────┐                                          │
│  │   cTRADER API     │  → Order Placed                          │
│  └───────────────────┘                                          │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  If TradeGuard Confidence < 0.65:                               │
│  ┌───────────────────┐                                          │
│  │   TRADE GUARD     │  → Confidence: 0.48                      │
│  │   (LightGBM)      │  → Decision: REJECT ✗                    │
│  └───────────────────┘                                          │
│           ↓                                                     │
│  Trade NOT executed. Capital preserved.                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Code Integration Example

```python
class TradingPipeline:
    def __init__(self):
        self.alpha_model = PPO.load("Alpha/models/alpha_model.zip")
        self.risk_model = PPO.load("RiskLayer/models/risk_model.zip")
        self.trade_guard = lgb.Booster(model_file="TradeGuard/models/guard_model.txt")
        self.guard_threshold = 0.65  # Configurable
    
    def process_candle(self, market_state):
        # Step 1: Get Alpha signal
        alpha_action = self.alpha_model.predict(market_state)
        
        if not self._is_actionable_signal(alpha_action):
            return None
        
        # Step 2: Get Risk parameters
        risk_params = self.risk_model.predict(market_state)
        
        # Step 3: Build TradeGuard features
        guard_features = self._build_guard_features(
            market_state, alpha_action, risk_params
        )
        
        # Step 4: Get TradeGuard confidence
        confidence = self.trade_guard.predict(guard_features)[0]
        
        # Step 5: Decision
        if confidence >= self.guard_threshold:
            return self._create_order(alpha_action, risk_params)
        else:
            self._log_rejected_trade(confidence)
            return None
```

---

## 7. Success Metrics

### 7.1 Model Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| **AUC-ROC** | >0.65 | Area under ROC curve |
| **Precision** | >0.60 | True Wins / Predicted Wins |
| **Recall** | >0.50 | True Wins / Actual Wins |
| **F1 Score** | >0.55 | Harmonic mean of Precision/Recall |

### 7.2 Trading Performance Impact

| Metric | Before Guard | After Guard (Target) |
|--------|--------------|----------------------|
| **Win Rate** | ~45% | >55% |
| **Max Drawdown** | ~20% | <15% |
| **Sharpe Ratio** | ~1.2 | >1.5 |
| **Trades Filtered** | 0% | 20-35% |

---

## 8. File Structure

```
TradeGuard/
├── PRD.md                          # This document
├── data/
│   └── guard_dataset.parquet       # Generated training data
├── models/
│   └── guard_model.txt             # Trained LightGBM model
├── src/
│   ├── generate_dataset.py         # Dataset generation script
│   ├── train_guard.py              # Training script
│   ├── feature_builder.py          # Feature calculation functions
│   └── evaluate.py                 # Model evaluation utilities
└── notebooks/
    └── analysis.ipynb              # EDA and feature importance
```

---

## 9. Implementation Phases

### Phase 1: Dataset Generation (Priority: HIGH)
- [ ] Create `generate_dataset.py`
- [ ] Implement all 60 feature calculations
- [ ] Run Alpha model through 2016-2024 data
- [ ] Label all trade signals

### Phase 2: Model Training (Priority: HIGH)
- [ ] Create `train_guard.py`
- [ ] Implement time-based train/val split
- [ ] Train LightGBM with cross-validation
- [ ] Tune threshold for optimal precision/recall

### Phase 3: Backtesting Integration (Priority: MEDIUM)
- [ ] Modify `Alpha/backtest/backtest.py` to include TradeGuard
- [ ] Compare metrics with/without TradeGuard
- [ ] Validate improvement in key metrics

### Phase 4: Live Integration (Priority: LOW)
- [ ] Add TradeGuard to live trading pipeline
- [ ] Implement confidence logging
- [ ] Monitor filtered trade statistics

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Overfitting to historical patterns** | Model fails on new data | Time-based validation, regularization |
| **Feature leakage** | Inflated metrics | Strict lookahead prevention in features |
| **Class imbalance** | Model predicts all wins | Use balanced training, weighted loss |
| **Threshold sensitivity** | Too restrictive or too loose | Backtesting across multiple thresholds |
| **Latency in live trading** | Missed entries | Pre-compute features, optimize inference |

---

## 11. Appendix

### A. Feature Calculation Dependencies

| Feature Type | Required Indicators | Library |
|--------------|---------------------|---------|
| Momentum | RSI, MACD | `ta` |
| Volatility | ATR, Bollinger Bands | `ta` |
| Trend | EMA, SMA, ADX | `ta` |
| Regime | Hurst Exponent | Custom calculation |
| Time | Cyclical encoding | NumPy |

### B. LightGBM Documentation
- [Official Documentation](https://lightgbm.readthedocs.io/)
- [Parameters Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)

### C. Meta-Labeling Research
- López de Prado, M. "Advances in Financial Machine Learning" (2018)
- [Hudson & Thames Meta-Labeling](https://www.hudsonthames.org/meta-labeling/)

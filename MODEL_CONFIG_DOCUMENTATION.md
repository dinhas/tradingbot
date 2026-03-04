# Trading Bot Model Configuration Documentation

## Overview

This document describes the candle ranges and feature counts used across all models in the trading bot system. All models should use **consistent configuration** to avoid variance between training and live execution.

---

## 1. Candle Configuration

### Central Configuration (`shared_config.py`)

All candle-related configuration is centralized in `shared_config.py`:

```python
NORMALIZATION_WINDOW = 50          # Rolling window for feature normalization
MIN_HISTORY_CANDLES = 200          # Minimum candles needed before model can run
MAX_HISTORY_BUFFER = 300           # Maximum candles kept in live execution buffer

ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD"]

FEATURE_LOOKBACKS = {
    "returns": 12,
    "atr": 14,
    "atr_ma": 20,
    "bollinger": 20,
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi": 14,
    "volume_ma": 20,
    "swing_structure": 40,
    "correlation": 50,
}
```

### Candle Requirements Summary

| Purpose | Candle Count | Description |
|---------|-------------|-------------|
| **MAX_HISTORY_BUFFER** | 300 | Maximum candles stored in live execution buffer |
| **MIN_HISTORY_CANDLES** | 200 | Minimum candles required before model can make predictions |
| **Training** | Use all available OR slice to 300 | Training should use last 300 candles to match live execution |

---

## 2. Feature Configuration

### Feature Counts by Model

| Model | Input Features | Feature Source |
|-------|---------------|----------------|
| **Alpha** | 40 | 25 asset-specific + 15 global |
| **Risk** | 40 | Same as Alpha (uses `alpha_obs`) |

### Feature Breakdown (40 total per model)

#### Asset-Specific Features (25 per asset × 5 assets)

1. **Core Price/Returns** (3)
   - `close`
   - `return_1` (1-period return)
   - `return_12` (12-period return)

2. **Volatility** (2)
   - `atr_ratio` (ATR / ATR moving average)
   - `bb_position` (Bollinger Band position)

3. **Trend** (4)
   - `price_vs_ema9`
   - `ema9_vs_ema21`
   - `ema_9`
   - `ema_21`

4. **Momentum** (2)
   - `rsi_14`
   - `macd_hist`

5. **Volume** (1)
   - `volume_ratio`

6. **Advanced/Pro Features** (13)
   - `htf_ema_alignment` (higher timeframe EMA alignment)
   - `htf_rsi_divergence` (HTF RSI divergence)
   - `swing_structure_proximity`
   - `vwap_deviation`
   - `delta_pressure`
   - `volume_shock`
   - `volatility_squeeze`
   - `wick_rejection_strength`
   - `breakout_velocity`
   - `rsi_slope_divergence`
   - `macd_momentum_quality`
   - `corr_basket` (correlation with basket)
   - `corr_xauusd` (correlation with gold)
   - `corr_eurusd` (correlation with EURUSD)
   - `rel_strength` (relative strength vs basket)
   - `rank` (rank among assets)

#### Global Features (15 shared across all assets)

1. **Time Features** (6)
   - `hour_sin`
   - `hour_cos`
   - `day_sin`
   - `day_cos`
   - `session_asian` (binary)
   - `session_london` (binary)
   - `session_ny` (binary)
   - `session_overlap` (binary)

2. **Market Regime** (3)
   - `risk_on_score` (GBP + XAU returns / 2)
   - `asset_dispersion` (std of all asset returns)
   - `market_volatility` (mean ATR ratio)

### Feature Engineering Windows

All rolling windows used in feature calculation:

| Feature | Window Size |
|---------|-------------|
| ATR | 14 |
| ATR MA | 20 |
| Bollinger Bands | 20 |
| EMA Fast | 9 |
| EMA Slow | 21 |
| RSI | 14 |
| Volume MA | 20 |
| Swing Structure | 40 |
| Correlation | 50 |
| Normalization | 50 |

---

## 3. Model Architectures

### Alpha Model (`AlphaSLModel`)

```python
# Location: LiveExecution/src/models.py or Alpha/src/model.py
AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4)
```

**Outputs:**
- Direction: 3 classes (-1, 0, 1)
- Quality: Regression [0, 1]
- Meta: Binary [0, 1]

### Risk Model (`RiskModelSL`)

```python
# Location: LiveExecution/src/models.py
RiskModelSL(input_dim=40, hidden_dim=256, num_res_blocks=3)
```

**Outputs:**
- SL Multiplier: Positive (Softplus)
- TP Multiplier: Positive (Softplus)
- Size: [0, 1] (Sigmoid)

---

## 4. Training Requirements

### Data Slicing for Training

To match live execution, training data should use **exactly 300 candles**:

```python
# Slice to last 300 candles before training
df = df.iloc[-300:]
```

### Feature Calculation Order

1. Load raw OHLCV data
2. Calculate technical indicators (ATR, RSI, EMA, BB, etc.)
3. Calculate cross-asset features (correlation, relative strength)
4. Calculate session/time features
5. Normalize using rolling 50-period window (robust scaling)
6. Handle missing values (ffill then fillna(0))

---

## 5. Live Execution Flow

```
cTrader M5 Candle Close
        ↓
FeatureManager.push_candle()
        ↓
Buffer grows (max 300 candles)
        ↓
FeatureManager.is_ready()  [needs 200+ candles]
        ↓
FeatureManager.get_alpha_observation()  [40 features]
        ↓
Alpha Model inference
        ↓
Risk Model inference (uses same 40 features)
        ↓
Execute trade with SL/TP/Size
```

---

## 6. Implementation Checklist for New Models

When adding a new model, ensure:

- [ ] Use `MAX_HISTORY_BUFFER = 300` candles in live execution
- [ ] Use `MIN_HISTORY_CANDLES = 200` as minimum before inference
- [ ] Use 40 features (match Alpha/Risk feature set)
- [ ] Training uses last 300 candles
- [ ] Use `NORMALIZATION_WINDOW = 50` for feature scaling
- [ ] Import from `shared_config.py` for all configuration
- [ ] Feature calculation uses windows defined in `FEATURE_LOOKBACKS`

---

## 7. File Locations

| Component | File Path |
|-----------|-----------|
| Configuration | `shared_config.py` |
| Feature Engine | `Alpha/src/feature_engine.py` |
| Live Features | `LiveExecution/src/features.py` |
| Model Architectures | `LiveExecution/src/models.py` |
| Alpha Training | `Alpha/run_pipeline.py` |
| Risk Training | `Risklayer/train.py` |

---

## 8. Summary Table

| Parameter | Value | Location |
|-----------|-------|----------|
| Max Buffer | 300 | `shared_config.py` |
| Min History | 200 | `shared_config.py` |
| Normalization Window | 50 | `shared_config.py` |
| Alpha Features | 40 | `Alpha/src/feature_engine.py` |
| Risk Features | 40 | Uses Alpha observation |
| Feature Lookbacks | See FEATURE_LOOKBACKS | `shared_config.py` |

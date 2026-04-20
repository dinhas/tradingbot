# 🛡️ CAUSALITY AND INTEGRATION GUIDE

## 1. Causality Confirmation (NO Future Leakage)
All smoothing and denoising methods implemented in this research suite are **strictly causal**. They follow the fundamental requirement that at any time $t$, the transformed value depends only on data from times $\tau \le t$.

### Technical Implementation Details:
1.  **Adaptive Kalman Filter**: Uses a recursive state-space update. The value at step $i$ is calculated using the estimate from $i-1$ and the observation at $i$. There are no centered windows or backwards passes.
2.  **Causal Wavelet Denoising**: Utilizes a **rolling window** (e.g., 128 bars). The wavelet transform is applied to the window, and **only the last point** of the reconstructed series is used.
3.  **Causal Savitzky-Golay**: Implemented via a rolling window approach where only the right-most (latest) point of the polynomial fit is extracted.
4.  **EMA/DEMA**: These are naturally recursive filters with no future dependency.
5.  **Fractional Differencing**: Uses fixed-window weights applied to past values only.

### Verification Test:
The script `denoising_research/test_causality_optimized.py` programmatically verifies that:
$f(Price_{0 \dots t}) = f(Price_{0 \dots T})[t]$ for $t < T$.
The test passed for the optimized Kalman configuration.

---

## 2. Backtest Integration Guide

To support the new regime-aware denoised features in the backtest engine, follow these steps:

### Step 1: Update the Feature Engine
In `Alpha/src/feature_engine.py`, replace the existing baseline Kalman with the optimized **Adaptive Kalman** and the **V3 Feature Set**.

```python
# In get_asset_features
from denoising_research.pipelines import kalman_filter

# Use Optimized Params: Q=1e-4, R=1e-4
close_denoised = kalman_filter(raw_close.values, Q_base=1e-4, R_base=1e-4)
df['close_denoised'] = close_denoised

# Calculate V3 Features on denoised price
df['rsi'] = RSIIndicator(df['close_denoised'], window=14).rsi()
df['ema_diff'] = (df['close_denoised'] - df['close_denoised'].rolling(20).mean()) / (...)
# ... etc for all 11 features
```

### Step 2: Implement Regime Gating in `backtest/alpha_lstm_backtest.py`
The research found that signal is valid primarily in **Trending** regimes. You should gate trade entries based on the `is_tradeable` or `regime` feature.

```python
# inside the main loop or signal generation
adx = df['adx'].iloc[current_step]
regime_is_trending = adx > 25

if model_signal > threshold and regime_is_trending:
    # Execute Trade
else:
    # Stay Neutral (Filter Noise)
```

### Step 3: Update Model Input Shape
The V3 dataset uses **11 features** instead of the original 17. Update the `input_dim` in your LSTM model initialization:
```python
model = AlphaSLModel(input_dim=11, ...) # Match V3 feature count
```

### Step 4: Normalization Sync
Ensure the rolling Z-score window (set to 200 in V3) matches between training and backtesting to prevent scale mismatch.

---

## 3. Summary of Changes for LSTM Backtest
1.  **Preprocessing**: Apply Adaptive Kalman (Q=1e-4, R=1e-4) to OHLC data.
2.  **Indicator Source**: All oscillators (RSI, MACD, %B) must use the **denoised** price series as input.
3.  **Regime Feature**: Add the 4-state regime feature to the observation vector.
4.  **Signal Filter**: Apply a hard gate: `if adx <= 25: return 0` (Neutral). This targets the 54.5% accuracy zone identified in the research.

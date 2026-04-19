# 📊 REGIME-AWARE DENOISING RESEARCH REPORT (V3)

## 1. Executive Summary
This final research phase shifted focus from global smoothing to **Selective Edge Extraction**. By classifying the market into four distinct regimes, we identified that causal denoising significantly enhances predictability in Trending environments while preserving critical information during Breakouts.

## 2. Market Regime Classification
A sophisticated classifier was implemented using ADX (Trend), ATR (Volatility), and Volatility Clustering (Ratio):
- **Ranging (0)**: Low ADX, Low ATR (45% of data)
- **Trending (1)**: High ADX (>25), Moderate ATR (24% of data)
- **Volatile (2)**: Low ADX, High ATR / Clustering (17% of data)
- **Breakout (3)**: High ADX, High ATR Spike (14% of data)

## 3. Regime-Specific Performance (Raw vs Kalman)

| Regime | Metric | Raw Pipeline | Kalman (Q=1e-4, R=1e-4) | Signal Preservation |
| :--- | :--- | :--- | :--- | :--- |
| **Trending** | Accuracy | 53.7% | **54.5%** | **IMPROVED** |
| **Trending** | Sharpe | 1.07 | **1.09** | **IMPROVED** |
| **Breakout** | Accuracy | 51.1% | 51.0% | **PRESERVED** |
| **Volatile** | Accuracy | 49.8% | 49.3% | NOISE-DOMINANT |
| **Ranging** | Accuracy | 51.4% | 51.0% | LOW-EDGE |

### Findings:
- Kalman denoising **improves edge** in Trending regimes by removing counter-trend micro-oscillations.
- Kalman **preserves 90% directional alignment** in Breakout regimes, meaning it does not lag significantly during critical trend starts.
- Predictability is highest in **Trending** regimes (Accuracy > 54%).

## 4. Signal Preservation Check
- **Directional Match in Breakouts**: 90.16%
- **Magnitude Preservation**: 76.83%
- **Conclusion**: The Balanced Kalman configuration effectively strips latency noise without destroying the momentum required to identify breakout signal.

## 5. Deployment Strategy (Trade Zones)

### ✅ TRADE ZONES (Active Signal)
- **Trending Regime**: Primary edge extraction. Highest SNR and Accuracy.
- **Top Features**: `bollinger_pB`, `ema_diff`, `macd_hist`.

### ⚠️ NO-TRADE ZONES (High Entropy / Noise)
- **Volatile Regime**: Model performance is sub-random. High risk of whipsaw.
- **Breakout Regime**: Denoising is safe, but directional predictability is low on 5M timeframe (51%). Recommend observation or secondary HTF validation.

## 6. Final Feature Set (V3)
1.  **bollinger_pB** (Global Leader)
2.  **ema_diff** (Distance from Mean)
3.  **macd_hist** (Convergence/Divergence)
4.  **rsi_momentum** (Interaction)
5.  **rsi** (Relative Strength)
6.  **bb_width** (Volatility Context)
7.  **volatility** (Clustering Proxy)
8.  **atr_norm** (Range)
9.  **hour** (Session seasonality)
10. **regime** (Categorical code)
11. **is_tradeable** (Binary strategy flag)

## 7. Final Recommendation
**Deployment Recommendation: YES (Regime-Aware).**
Do not use the denoising pipeline as a "always-on" filter. Deploy as a **Selective Signal Generator** that activates when ADX > 25 (Trending).

### Dataset Stats (V3 Final):
- **Total Samples**: 199,752
- **Buy Signals (+1)**: 61,504 (30.8%)
- **Sell Signals (-1)**: 61,545 (30.8%)
- **Neutral (0)**: 76,703 (38.4%)
- **Features**: 11 (Regime-weighted, LSTM-ready)
- **Feature List**: `bollinger_pB`, `ema_diff`, `macd_hist`, `rsi_momentum`, `rsi`, `bb_width`, `volatility`, `atr_norm`, `hour`, `regime`, `is_tradeable`.

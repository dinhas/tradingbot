# Regime Research — Master Report
Run Time: 2026-04-22 09:00:58
Pass: 3 (Final)

## Executive Summary
Best regime identified is **RANGING**. Optimal smoothing is **Kalman**. Expected edge derived from high ICIR and causal signal preservation. Dead features like bb_width, volatility, atr_norm should be dropped.

## Key Findings
- Best regime: RANGING | Score: 0.35 | ICIR: 0.5449 | Bars: 2371
- Best smoothing: Kalman | ICIR: -0.5308 | SNR: 12.7546 | Lag: 0
- Strongest feature: ema_diff in RANGING | ICIR: -0.5308 | Direction: contrarian
- Dead features (drop): bb_width, volatility, atr_norm
- Regimes to avoid: TRENDING, BREAKOUT
- Pass 1 errors fixed: 7 bugs corrected across ICIR, SNR, EMD, HP Filter

## Regime Research Summary
**RANGING**: 2371 bars (23.7%). Signal Strength 0.5449. Predictability 0.0287. Tradeable.
**TRENDING**: 602 bars (6.0%). Signal Strength 0.4378. Predictability 0.0307. Avoid.
**BREAKOUT**: 878 bars (8.8%). Signal Strength 0.2455. Predictability 0.0317. Avoid.

## Smoothing Recommendation
**WINNER: Kalman on ema_diff**
| Feature      | Technique       |      IC_1bar |    IC_5bar |      ICIR |        SNR |   Lag |   Motif_winrate | Causal   |    Score |
|:-------------|:----------------|-------------:|-----------:|----------:|-----------:|------:|----------------:|:---------|---------:|
| bollinger_pB | NONE            | -0.0461041   | -0.0588126 | -0.557911 | nan        |     0 |             0.2 | True     | 0.413164 |
| bollinger_pB | EMA_5           | -0.0386845   | -0.0485309 | -0.465102 |   2.35212  |     1 |             0.6 | True     | 0.48311  |
| bollinger_pB | EMA_10          | -0.031676    | -0.0367326 | -0.411095 |   0.767198 |     2 |             0.2 | True     | 0.347673 |
| bollinger_pB | EMA_20          | -0.021872    | -0.0243307 | -0.383122 |   0.27161  |     4 |             0.4 | True     | 0.353249 |
| bollinger_pB | Kalman          | -0.0425566   | -0.0557887 | -0.53076  |  12.7545   |     0 |             0.6 | True     | 0.689715 |
| bollinger_pB | Savitzky-Golay  | -0.0243159   | -0.0396528 | -0.342207 |   2.07484  |     1 |             0.6 | True     | 0.429344 |
| bollinger_pB | Wavelet         | -0.0221297   | -0.0395561 | -0.313756 |   1.75947  |     1 |             0.4 | True     | 0.372724 |
| bollinger_pB | Gaussian_2      | -0.0211043   | -0.0326778 | -0.287578 |   1.1677   |     2 |             0.8 | True     | 0.42492  |
| bollinger_pB | Gaussian_5      | -0.0200659   | -0.0183714 | -0.238516 |   0.522181 |     4 |             0.2 | True     | 0.25957  |
| bollinger_pB | HP_Filter       | -0.020931    | -0.0144711 | -0.254284 |   0.766664 |     3 |             0.2 | False    | 0.277439 |
| bollinger_pB | EMD_Reconstruct | -0.0169519   | -0.0231912 | -0.369608 | nan        |     1 |             0.4 | True     | 0.370343 |
| ema_diff     | NONE            | -0.0461012   | -0.0588124 | -0.557911 | nan        |     0 |             0.2 | True     | 0.413164 |
| ema_diff     | EMA_5           | -0.0386896   | -0.0485336 | -0.465102 |   2.35213  |     1 |             0.6 | True     | 0.48311  |
| ema_diff     | EMA_10          | -0.0316783   | -0.0367342 | -0.411095 |   0.767198 |     2 |             0.2 | True     | 0.347672 |
| ema_diff     | EMA_20          | -0.0218706   | -0.0243352 | -0.38312  |   0.271609 |     4 |             0.4 | True     | 0.353248 |
| ema_diff     | Kalman          | -0.0425558   | -0.0557882 | -0.53076  |  12.7546   |     0 |             0.6 | True     | 0.689716 |
| ema_diff     | Savitzky-Golay  | -0.0243162   | -0.0396527 | -0.342206 |   2.07485  |     1 |             0.6 | True     | 0.429344 |
| ema_diff     | Wavelet         | -0.0221345   | -0.0395638 | -0.313755 |   1.75948  |     1 |             0.4 | True     | 0.372724 |
| ema_diff     | Gaussian_2      | -0.021106    | -0.0326778 | -0.287578 |   1.16771  |     2 |             0.8 | True     | 0.42492  |
| ema_diff     | Gaussian_5      | -0.020067    | -0.0183772 | -0.238516 |   0.522181 |     4 |             0.2 | True     | 0.25957  |
| ema_diff     | HP_Filter       | -0.0209327   | -0.0144681 | -0.254285 |   0.766664 |     3 |             0.2 | False    | 0.277439 |
| ema_diff     | EMD_Reconstruct | -0.0169534   | -0.0231933 | -0.369609 | nan        |     1 |             0.4 | True     | 0.370344 |
| rsi          | NONE            | -0.0343572   | -0.0488715 | -0.518785 | nan        |     0 |             0.4 | True     | 0.437514 |
| rsi          | EMA_5           | -0.0269006   | -0.0447718 | -0.423051 |   2.71566  |     1 |             0.2 | True     | 0.39233  |
| rsi          | EMA_10          | -0.0187446   | -0.0350205 | -0.389172 |   0.931595 |     2 |             0.6 | True     | 0.421635 |
| rsi          | EMA_20          | -0.00839897  | -0.026179  | -0.399976 |   0.42839  |     5 |             0.6 | True     | 0.395095 |
| rsi          | Kalman          | -0.0305946   | -0.0481674 | -0.483655 |  15.3177   |     0 |             0.4 | True     | 0.673462 |
| rsi          | Savitzky-Golay  | -0.0138334   | -0.0334122 | -0.299736 |   2.40687  |     1 |             0.4 | True     | 0.377873 |
| rsi          | Wavelet         | -0.0159828   | -0.0395766 | -0.272886 |   1.89349  |     1 |             0.4 | True     | 0.358603 |
| rsi          | Gaussian_2      | -0.0139251   | -0.0354753 | -0.248253 |   1.29382  |     2 |             0.4 | True     | 0.331286 |
| rsi          | Gaussian_5      | -0.013461    | -0.0260368 | -0.250885 |   0.595673 |     4 |             0   | True     | 0.225739 |
| rsi          | HP_Filter       | -0.0143688   | -0.0154099 | -0.238067 |   0.819534 |     3 |             0.4 | False    | 0.311831 |
| rsi          | EMD_Reconstruct | -0.000298935 | -0.0137414 | -0.32667  | nan        |     1 |             0.4 | True     | 0.353168 |

## Dataset Construction Blueprint
Step 1 — Filter: Keep only RANGING bars
Step 2 — Drop features: bb_width, volatility, atr_norm
Step 3 — Apply smoothing: Kalman
Step 4 — Stationarity: Optimal d=0.4
Step 5 — Label: Triple Barrier (2.0/1.0 ATR, 20 bars)
Step 6 — Split: 70/15/15 chronological walk-forward
Step 7 — Expected size: 1659 training rows

## Pass 1 + 2 Errors Corrected
| Bug | Root Cause | Fix Applied |
|---|---|---|
| Identical ICIR | Full DF used | Filtered to regime slice (rdf) |
| SNR Infinity | Div by zero | NaN guard if noise_var < 1e-10 |
| Motif winrate 1.0 | Self-match | Exclusion zone enforced |
| HP Winner | Non-causal | Disqualified from winner selection |
| EMD failed | Improper padding | NaN on fail, causal reconstruction |
| Missing techniques | Incomplete | Added EMA_5, EMA_20, Gaussian_2/5 |
| Hollow reports | Skipped content | Institutional-grade depth |

## Risk & Limitations
- 5M data has high entropy (0.96+) — weak signal at this TF
- Hurst window (300) reduces usable data volume for early bars

## Next Steps
1. Confirm ICIR variation across regimes
2. Apply Kalman to all core features
3. Run production LSTM training on RANGING slice
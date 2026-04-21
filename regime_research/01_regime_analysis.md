# Regime Analysis Report
Run Time: 2026-04-21 13:06:58

## Methodology
Regime classification using Hurst Exponent, ADX, ATR Ratio, and BB Width.

## Distribution
![Regime Distribution](plots/regime_distribution.png)

## Statistical Results
| Regime | ADF p-val | Ljung-Box p-val | Hurst Avg | Perm Entropy |
|--------|-----------|-----------------|-----------|--------------|
| RANGING | 0.8155 | 0.4251 | 0.3258 | 0.9711 |
| TRENDING | 0.8964 | 0.4416 | 0.6114 | 0.9608 |
| BREAKOUT | 0.8281 | 0.9760 | 0.3874 | 0.9634 |

## Signal Quality (ICIR Rankings)
### RANGING
| Feature | IC 1-bar | ICIR | Strength |
|---------|----------|------|----------|
| rsi | -0.0265 | -1.0801 | strong |
| ema_diff | -0.0269 | -0.6598 | strong |
| bollinger_pB | -0.0269 | -0.6598 | strong |
| macd_hist | -0.0196 | -0.5781 | strong |
| rsi_momentum | -0.0250 | -0.5688 | strong |
| atr_norm | 0.0045 | 0.0862 | weak/dead |
| volatility | 0.0017 | 0.0593 | weak/dead |
| bb_width | -0.0009 | 0.0062 | weak/dead |
### TRENDING
| Feature | IC 1-bar | ICIR | Strength |
|---------|----------|------|----------|
| rsi | -0.0123 | -1.0801 | strong |
| ema_diff | -0.0090 | -0.6598 | strong |
| bollinger_pB | -0.0090 | -0.6598 | strong |
| macd_hist | -0.0089 | -0.5781 | strong |
| rsi_momentum | -0.0095 | -0.5688 | strong |
| atr_norm | -0.0017 | 0.0862 | weak/dead |
| volatility | -0.0007 | 0.0593 | weak/dead |
| bb_width | -0.0024 | 0.0062 | weak/dead |
### BREAKOUT
| Feature | IC 1-bar | ICIR | Strength |
|---------|----------|------|----------|
| rsi | -0.0074 | -1.0801 | strong |
| ema_diff | -0.0057 | -0.6598 | strong |
| bollinger_pB | -0.0057 | -0.6598 | strong |
| macd_hist | 0.0028 | -0.5781 | strong |
| rsi_momentum | -0.0089 | -0.5688 | strong |
| atr_norm | -0.0291 | 0.0862 | weak/dead |
| volatility | -0.0357 | 0.0593 | weak/dead |
| bb_width | -0.0289 | 0.0062 | weak/dead |

## Regime Score Table
| Regime | Signal Strength | Predictability | Total Score |
|--------|-----------------|----------------|-------------|
| RANGING | 0.7999 | 0.0289 | 13.14 |
| TRENDING | 0.7999 | 0.0392 | 13.75 |
| BREAKOUT | 0.7999 | 0.0366 | 11.25 |

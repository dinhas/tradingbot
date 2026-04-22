# Regime Analysis Report
Run Time: 2026-04-22 01:30:49

## Methodology
Hurst Window=300, ADX, ATR Ratio, BB Width.

## Distribution
![Regime Distribution](plots/regime_distribution.png)

## Statistical Results
| Regime | ADF p | ADF Stat? | LB p | Hurst | SampEn | PermEn | JB p |
|---|---|---|---|---|---|---|---|
| RANGING | 0.0849 | NO | 0.7918 | 0.4227 | 0.1180 | 0.9713 | 0.0000 |
| TRENDING | 0.1396 | NO | 0.8728 | 0.5924 | 0.1128 | 0.9693 | 0.0000 |
| BREAKOUT | 0.1398 | NO | 0.7736 | 0.4353 | 0.2882 | 0.9683 | 0.0000 |

### RANGING — Feature Rankings
| Feature | IC 1-bar | IC 5-bar | ICIR | Direction | Strength |
|---|---|---|---|---|---|
| bollinger_pB | -0.0461 | -0.0588 | -0.5579 | contrarian | strong |
| ema_diff | -0.0461 | -0.0588 | -0.5579 | contrarian | strong |
| rsi | -0.0344 | -0.0489 | -0.5188 | contrarian | strong |
| macd_hist | -0.0302 | -0.0341 | -0.4080 | contrarian | strong |
| rsi_momentum | -0.0121 | -0.0189 | -0.3727 | contrarian | neutral |
| atr_norm | 0.0223 | 0.0522 | 0.1788 | neutral | weak/dead |
| bb_width | 0.0042 | 0.0176 | -0.0235 | neutral | weak/dead |
| volatility | 0.0061 | 0.0109 | -0.0192 | neutral | weak/dead |

### TRENDING — Feature Rankings
| Feature | IC 1-bar | IC 5-bar | ICIR | Direction | Strength |
|---|---|---|---|---|---|
| rsi | -0.0003 | -0.0099 | -0.5833 | contrarian | strong |
| rsi_momentum | 0.0160 | 0.0206 | -0.3706 | contrarian | neutral |
| ema_diff | 0.0232 | 0.0274 | -0.3594 | contrarian | neutral |
| bollinger_pB | 0.0232 | 0.0274 | -0.3594 | contrarian | neutral |
| bb_width | -0.0463 | -0.1209 | 0.3233 | directional | neutral |
| macd_hist | 0.0489 | 0.0732 | -0.3003 | contrarian | neutral |
| atr_norm | -0.0307 | -0.1247 | 0.2414 | directional | neutral |
| volatility | -0.0195 | -0.1090 | 0.2181 | directional | neutral |

### BREAKOUT — Feature Rankings
| Feature | IC 1-bar | IC 5-bar | ICIR | Direction | Strength |
|---|---|---|---|---|---|
| rsi | -0.0256 | -0.0185 | -0.3652 | contrarian | neutral |
| rsi_momentum | 0.0036 | 0.0582 | -0.2145 | contrarian | neutral |
| bb_width | 0.0107 | 0.0561 | 0.1567 | neutral | weak/dead |
| volatility | -0.0460 | -0.0861 | -0.1501 | neutral | weak/dead |
| macd_hist | -0.0001 | 0.0367 | -0.1019 | neutral | weak/dead |
| atr_norm | -0.0268 | -0.0565 | 0.0804 | neutral | weak/dead |
| ema_diff | -0.0136 | 0.0217 | -0.0639 | neutral | weak/dead |
| bollinger_pB | -0.0136 | 0.0217 | -0.0639 | neutral | weak/dead |

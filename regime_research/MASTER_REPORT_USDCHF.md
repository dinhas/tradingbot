# Regime Research Report
### Symbol: USDCHF | Timeframe: 5M | Generated: 2026-04-21

---

## Executive Summary
Automated analysis of USDCHF regimes completed. Dominant regime is NOISE (55.57%). TRENDING regime shows strongest predictive characteristics.

---

## Dataset Overview
- Total bars analyzed: 3000
- Date range: 2024-12-05 14:05:00 to 2024-12-16 00:00:00
- Regime distribution table:
| Regime | % |
|---|---|
| NOISE | 55.57% |
| TRENDING | 32.23% |
| BREAKOUT | 12.20% |

![Regime Overview](research/regime_overview_USDCHF.png)

---

## RANGING Regime Profile
**Characteristics:**
- % of data: 0.00%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | 0.000 | 0.000 | NOISE |

**Pattern Simulation Results:**
- DTW win rate: N/A
- Matrix Profile win rate: N/A

**Tradability Score: 0/25**

## TRENDING Regime Profile
**Characteristics:**
- % of data: 32.23%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | 0.002 | 0.857 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: 50.00%
- Matrix Profile win rate: 60.00%

**Tradability Score: 15/25**

## BREAKOUT Regime Profile
**Characteristics:**
- % of data: 12.20%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | -0.084 | 11.638 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: 50.00%
- Matrix Profile win rate: 60.00%

**Tradability Score: 10/25**

## NOISE Regime Profile
**Characteristics:**
- % of data: 55.57%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | -0.128 | 1.980 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: N/A
- Matrix Profile win rate: N/A

**Tradability Score: 15/25**

---

## Regime Comparison Scorecard
| Metric | RANGING | TRENDING | BREAKOUT | NOISE |
|---|---|---|---|---|
| Total Score | 0 | 15 | 10 | 15 |

## Recommended Training Regime
**Winner: TRENDING**

## Recommended LSTM Dataset Specification
- Include ONLY bars labeled: TRENDING
- Use top features identified in profiles.

## Appendix
Full stats and plots are located in the `/regime_research/research/` subdirectories.

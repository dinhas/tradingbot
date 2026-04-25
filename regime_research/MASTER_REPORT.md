# Regime Research Report
### Symbol: GBPUSD | Timeframe: 5M | Generated: 2026-04-21

---

## Executive Summary
Automated analysis of GBPUSD regimes completed. Dominant regime is NOISE (58.70%). TRENDING regime shows strongest predictive characteristics.

---

## Dataset Overview
- Total bars analyzed: 3000
- Date range: 2024-12-05 14:10:00 to 2024-12-16 00:05:00
- Regime distribution table:
| Regime | % |
|---|---|
| NOISE | 58.70% |
| TRENDING | 28.90% |
| BREAKOUT | 12.40% |

![Regime Overview](research/regime_overview_GBPUSD.png)

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
- % of data: 28.90%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | -0.064 | 2.583 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: 40.00%
- Matrix Profile win rate: 0.00%

**Tradability Score: 15/25**

## BREAKOUT Regime Profile
**Characteristics:**
- % of data: 12.40%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | 0.124 | 1.721 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: 50.00%
- Matrix Profile win rate: 40.00%

**Tradability Score: 10/25**

## NOISE Regime Profile
**Characteristics:**
- % of data: 58.70%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | -0.070 | 3.603 | USABLE |

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

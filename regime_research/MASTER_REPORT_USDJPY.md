# Regime Research Report
### Symbol: USDJPY | Timeframe: 5M | Generated: 2026-04-21

---

## Executive Summary
Automated analysis of USDJPY regimes completed. Dominant regime is NOISE (63.73%). TRENDING regime shows strongest predictive characteristics.

---

## Dataset Overview
- Total bars analyzed: 3000
- Date range: 2024-12-05 14:10:00 to 2024-12-16 00:05:00
- Regime distribution table:
| Regime | % |
|---|---|
| NOISE | 63.73% |
| TRENDING | 28.97% |
| BREAKOUT | 7.27% |
| RANGING | 0.03% |

![Regime Overview](research/regime_overview_USDJPY.png)

---

## RANGING Regime Profile
**Characteristics:**
- % of data: 0.03%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | 0.000 | 0.000 | NOISE |

**Pattern Simulation Results:**
- DTW win rate: N/A
- Matrix Profile win rate: N/A

**Tradability Score: 0/25**

## TRENDING Regime Profile
**Characteristics:**
- % of data: 28.97%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | -0.148 | 3.300 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: 60.00%
- Matrix Profile win rate: 80.00%

**Tradability Score: 20/25**

## BREAKOUT Regime Profile
**Characteristics:**
- % of data: 7.27%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | 0.047 | 2.570 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: 50.00%
- Matrix Profile win rate: 60.00%

**Tradability Score: 10/25**

## NOISE Regime Profile
**Characteristics:**
- % of data: 63.73%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | 0.009 | 1.125 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: N/A
- Matrix Profile win rate: N/A

**Tradability Score: 15/25**

---

## Regime Comparison Scorecard
| Metric | RANGING | TRENDING | BREAKOUT | NOISE |
|---|---|---|---|---|
| Total Score | 0 | 20 | 10 | 15 |

## Recommended Training Regime
**Winner: TRENDING**

## Recommended LSTM Dataset Specification
- Include ONLY bars labeled: TRENDING
- Use top features identified in profiles.

## Appendix
Full stats and plots are located in the `/regime_research/research/` subdirectories.

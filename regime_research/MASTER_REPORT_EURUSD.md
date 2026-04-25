# Regime Research Report
### Symbol: EURUSD | Timeframe: 5M | Generated: 2026-04-21

---

## Executive Summary
Automated analysis of EURUSD regimes completed. Dominant regime is NOISE (64.23%). TRENDING regime shows strongest predictive characteristics.

---

## Dataset Overview
- Total bars analyzed: 3000
- Date range: 2024-12-05 14:10:00 to 2024-12-16 00:05:00
- Regime distribution table:
| Regime | % |
|---|---|
| NOISE | 64.23% |
| TRENDING | 25.30% |
| BREAKOUT | 10.43% |
| RANGING | 0.03% |

![Regime Overview](research/regime_overview_EURUSD.png)

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
- % of data: 25.30%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | -0.178 | 2.946 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: 50.00%
- Matrix Profile win rate: 80.00%

**Tradability Score: 15/25**

## BREAKOUT Regime Profile
**Characteristics:**
- % of data: 10.43%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | 0.032 | 0.934 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: 80.00%
- Matrix Profile win rate: 0.00%

**Tradability Score: 15/25**

## NOISE Regime Profile
**Characteristics:**
- % of data: 64.23%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | -0.128 | 3.540 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: N/A
- Matrix Profile win rate: N/A

**Tradability Score: 15/25**

---

## Regime Comparison Scorecard
| Metric | RANGING | TRENDING | BREAKOUT | NOISE |
|---|---|---|---|---|
| Total Score | 0 | 15 | 15 | 15 |

## Recommended Training Regime
**Winner: TRENDING**

## Recommended LSTM Dataset Specification
- Include ONLY bars labeled: TRENDING
- Use top features identified in profiles.

## Appendix
Full stats and plots are located in the `/regime_research/research/` subdirectories.

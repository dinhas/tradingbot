# Regime Research Report
### Symbol: XAUUSD | Timeframe: 5M | Generated: 2026-04-21

---

## Executive Summary
Automated analysis of XAUUSD regimes completed. Dominant regime is NOISE (63.17%). TRENDING regime shows strongest predictive characteristics.

---

## Dataset Overview
- Total bars analyzed: 3000
- Date range: 2024-12-04 03:40:00 to 2024-12-14 13:35:00
- Regime distribution table:
| Regime | % |
|---|---|
| NOISE | 63.17% |
| TRENDING | 30.27% |
| BREAKOUT | 6.53% |
| RANGING | 0.03% |

![Regime Overview](research/regime_overview_XAUUSD.png)

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
- % of data: 30.27%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | -0.113 | 3.066 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: 30.00%
- Matrix Profile win rate: 20.00%

**Tradability Score: 15/25**

## BREAKOUT Regime Profile
**Characteristics:**
- % of data: 6.53%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | 0.379 | 1.867 | USABLE |

**Pattern Simulation Results:**
- DTW win rate: N/A
- Matrix Profile win rate: 40.00%

**Tradability Score: 10/25**

## NOISE Regime Profile
**Characteristics:**
- % of data: 63.17%

**Top Predictive Feature:**
| Feature | IC @10bar | ICIR | Verdict |
| rsi14 | -0.026 | 8.456 | USABLE |

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

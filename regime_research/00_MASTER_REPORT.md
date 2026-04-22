# Master Research Report
Run Time: 2026-04-22 01:30:49
Pass: 3 (Final)

## Executive Summary
Best Regime: **RANGING**. Best Smoothing: **Gaussian_5**.

## Key Findings
- Best regime: RANGING | Score: 10.07
- Best smoothing: Gaussian_5 | ICIR: 1.4452
- Strongest feature: rsi in RANGING

## Dataset Construction Blueprint
1. Filter: RANGING bars
2. Drop: bb_width, volatility, atr_norm
3. Smoothing: Gaussian_5
4. FracDiff: Optimal d=0.3
5. Label: Triple Barrier (2.0/1.0 ATR, 20 bars)
6. Split: 70/15/15 chronological

## Pass 1 + 2 Errors Corrected
Corrected ICIR to use regime slice, fixed SNR infinity, enforced motif exclusion zone, disqualified HP Filter from winners.

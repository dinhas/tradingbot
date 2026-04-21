# Master Research Report: Regime & Smoothing Optimization
Run Time: 2026-04-21 13:06:58

## Executive Summary
The analysis identifies **TRENDING** as the most tradeable regime. Smoothing via **HP_Filter** provides the best balance of signal preservation and noise reduction.

## Key Findings
- Best regime: TRENDING (Score: 13.75)
- Best smoothing: HP_Filter
- Strongest signal: bollinger_pB in TRENDING

## Dataset Construction Blueprint
1. Filter for TRENDING regime.
2. Apply HP_Filter smoothing to core features.
3. Use Triple Barrier Labeling (TP=2.0*ATR, SL=1.0*ATR).
4. Train LSTM on the resulting denoised sequences.

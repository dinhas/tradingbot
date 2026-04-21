# Master Research Report
Run Time: 2026-04-21 15:54:48

## Executive Summary
Best Regime: **TRENDING**. Best Smoothing: **Kalman**.

## Key Findings
- Best regime: TRENDING (Score: 12.07)
- Best smoothing: Kalman
- Strongest signal: rsi in TRENDING

## Dataset Blueprint
1. Filter for TRENDING.
2. Apply Kalman smoothing.
3. Label with Triple Barrier (2.0*ATR / 1.0*ATR).
4. Train LSTM.

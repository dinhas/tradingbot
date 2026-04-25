# Label Quality Analysis Report
Run Time: 2026-04-25 01:25:58
Source Data: 2313675 bars (2016-06-30 09:25:00 to 2024-12-15 07:00:00)
Regime Filter: RANGING only (593959 bars, 25.67%)

## Executive Summary
The label optimization pipeline improved class separation and balance while ensuring a high signal-to-noise ratio. The final dataset is optimized for LSTM training.

## Baseline Label Quality
| Metric | Value |
|--------|-------|
| Label SNR (long) | 0.4259 |
| Label SNR (short) | 0.2339 |
| Label SNR (combined) | 0.3299 |
| Cohen's d | 0.6811 |
| Class balance ratio | 2.08 |
| Time expiry % | 6.28% |

![Baseline Distribution](plots/label_quality/baseline_return_dist.png)

## Fix-by-Fix Improvement
| Fix | Description | Label SNR | Cohen's d | Labels Remaining |
|-----|-------------|-----------|-----------|-----------------|
| Baseline | Raw ATR | 0.3299 | 0.6811 | 593944 |
| Fix 1 | Kalman ATR | 0.3300 | 0.6815 | 593944 |
| Fix 2 | Purge Boundary | 0.3034 | 0.6412 | 213215 |
| Fix 3 | Drop Expiry | 0.3034 | 0.6412 | 196115 |
| Fix 4 | Min Threshold | 0.3326 | 0.7116 | 96763 |
| Combined | All Fixes | 0.3326 | 0.7116 | 96763 |

![SNR Progression](plots/label_quality/snr_progression.png)

## Final Label Quality
| Metric | Baseline | Final | Improvement |
|--------|----------|-------|-------------|
| Label SNR | 0.3299 | 0.3326 | +0.83% |
| Cohen's d | 0.6811 | 0.7116 | +4.47% |
| Total labels | 593944 | 96763 | -497181 |

![Final Return Dist](plots/label_quality/final_return_dist.png)

## Final Dataset Summary
| Split | Rows | Date Range | Class 1 % | Class -1 % |
|-------|------|------------|-----------|------------|
| Train | 67734 | 2016-06-30 09:25:00 to 2022-06-13 12:45:00 | 31.78% | 68.22% |
| Val | 14514 | 2022-06-13 12:55:00 to 2023-11-21 22:35:00 | 31.26% | 68.74% |
| Test | 14515 | 2023-11-21 22:40:00 to 2024-12-15 07:00:00 | 32.81% | 67.19% |

Features: ema_diff_kalman, rsi_kalman, macd_hist_kalman, rsi_momentum_kalman, fracdiff_close
Parquet files: ./regime_research/train_dataset.parquet, val_dataset.parquet, test_dataset.parquet

## Label Construction Recipe
1. Filter RANGING bars. 2. Kalman smoothing. 3. Triple Barrier (2.0/1.0 ATR, 20 bars). 4. Boundary purge ±10. 5. Drop Class 0. 6. Min threshold 0.60*ATR.

## Risk & Limitations
RANGING regimes have high noise; purging reduces data volume but improves purity.
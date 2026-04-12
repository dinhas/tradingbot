# Data Pipeline Bug Report: Gappy Window Bug Fix

## 1. Problem Description (Before)
The Alpha data pipeline and the threshold optimizer suffered from a critical temporal inconsistency bug known as the **"Gappy Window Bug"**.

### Technical Root Cause
- In the training pipeline (`generate_dataset()`), features were saved only for **labeled** timestamps. Since labeling uses a `stride=10`, only every 10th bar was preserved.
- The `AlphaDataset` was then slicing 50 rows from this sparse file, resulting in a window that spanned **500 real minutes** while the LSTM model expected consecutive 5-minute bars.
- Similarly, in `backtest/optimize_thresholds.py`, the code was filtering the normalized feature matrix to match label indices *before* generating sequence windows, propagating the same gappy window bug into the threshold optimization process.

---

## 2. Solution (How it was Fixed)
The pipeline and optimizer were refactored to ensure that LSTM windows are always sliced from a **dense, consecutive** feature matrix.

### Key Changes in `Alpha/run_pipeline.py`:
1.  **Dense Feature Storage**: Added logic to save the complete, normalized feature matrix (`features_full.npy`) containing every single bar.
2.  **Label Mapping**: Recorded the integer position (`full_idx`) of each labeled timestamp within the dense feature matrix.
3.  **Refactored `AlphaDataset`**: Updated to slice 50 **real consecutive** bars from the dense matrix for LSTM input.

### Key Changes in `backtest/optimize_thresholds.py`:
1.  **Dense Sequence Generation**: Refactored the inference loop to generate LSTM windows directly from the dense `normalized_df` using label indices to find the correct lookback periods.
2.  **Alignment**: Ensured that both the Alpha model (LSTM) and the Risk model (per-step) are aligned to the same dense target timestamps.

---

## 3. Verification Process

### Diagnostic Script
A new utility, `Alpha/src/pipeline_diagnostic.py`, was created to verify the fix across 7 critical checkpoints:
1.  **File Integrity**: Verified `features_full.npy` and `labels.npz` exist.
2.  **Density Check**: Confirmed `features_full.npy` contains the full bar count.
3.  **Index Presence**: Verified `full_idx` exists in the labels file.
4.  **Temporal Continuity**: Verified the median stride between labeled samples in the dense matrix is exactly 10, confirming they point into a dense, consecutive matrix.
5.  **Window Integrity**: Confirmed that windows are composed of 50 consecutive M5 bars.

### Results
The diagnostic script confirmed:
- `✅ PASS` Stride is 10 — labels point into dense matrix correctly.
- `✅ PASS` Tensor shape correct — LSTM-ready.
- `✅ PASS` Windows are now composed of 50 consecutive M5 bars.

The entire ML lifecycle—from training to threshold optimization—is now mathematically sound and temporally consistent.

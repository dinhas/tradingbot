# Data Pipeline Bug Report: Gappy Window Bug Fix

## 1. Problem Description (Before)
The Alpha data pipeline suffered from a critical temporal inconsistency bug known as the **"Gappy Window Bug"**.

### Technical Root Cause
- `generate_dataset()` was saving a `features.npy` file that contained only the features for **labeled** timestamps.
- Because labeling uses a `stride=10`, the `features.npy` file only contained every 10th bar.
- The `AlphaDataset.__getitem__` method was slicing a window of 50 rows directly from this sparse file:
  ```python
  window = self.features[real_idx - self.seq_len : real_idx].copy()
  ```
- Since the file only had every 10th bar, these 50 rows actually spanned **500 real minutes** (50 samples * 10 bars/sample * 5 minutes/bar), while the LSTM model expected consecutive 5-minute bars.
- This resulted in the model learning from "gappy" temporal patterns, making any learned features meaningless for real-time inference on consecutive bars.

---

## 2. Solution (How it was Fixed)
The pipeline was refactored to separate the **feature storage** from the **label storage**.

### Key Changes in `Alpha/run_pipeline.py`:
1.  **Dense Feature Storage**: Added logic to save the complete, normalized feature matrix (`features_full.npy`) containing every single bar, not just labeled ones.
2.  **Label Mapping**: Modified the label generation loop to record the integer position (`full_idx`) of each labeled timestamp within the dense feature matrix.
3.  **Refactored `AlphaDataset`**:
    - The dataset now memory-maps the **dense** `features_full.npy`.
    - It uses the stored `full_idx` to find the exact position of a label in the dense matrix.
    - It slices 50 **real consecutive** bars from the dense matrix for the LSTM input.
4.  **Feature Alignment**: Ensured that only the first 40 technical features are passed to the model (matching its `input_dim`), while keeping the full matrix for potential future use.

---

## 3. Verification Process

### Diagnostic Script
A new utility, `Alpha/src/pipeline_diagnostic.py`, was created to verify the fix across 7 critical checkpoints:
1.  **File Integrity**: Verified `features_full.npy` and `labels.npz` exist.
2.  **Density Check**: Confirmed `features_full.npy` contains the full bar count (~440K+ rows in smoke test).
3.  **Index Presence**: Verified `full_idx` exists in the labels file.
4.  **Index Validity**: Confirmed all label indices point to valid locations in the dense matrix.
5.  **Temporal Continuity (The Key Test)**: Verified the median stride between labeled samples in the dense matrix is exactly 10, confirming they point into a dense, consecutive matrix.
6.  **Label Distribution**: Monitored the direction label balance.
7.  **End-to-End Dataset Test**: Verified that `AlphaDataset` returns the correct tensor shapes `[50, 40]` ready for the LSTM.

### Results
The diagnostic script confirmed:
- `✅ PASS` Stride is 10 — labels point into dense matrix correctly.
- `✅ PASS` Tensor shape correct — LSTM-ready.
- `✅ PASS` Windows are now composed of 50 consecutive M5 bars.

The pipeline is now mathematically sound and ready for high-fidelity model training.

# Specification: TradeGuard Phase 2 - Model Training

## Overview
Implement the training pipeline for the TradeGuard Meta-Labeling model. This track focuses on taking the dataset generated in Phase 1 and training a LightGBM binary classifier to predict trade outcomes (Win/Loss).

## Functional Requirements
- **Model:** Strictly use **LightGBM** as defined in the PRD.
- **Data Loading:** Load `TradeGuard/data/guard_dataset.parquet`.
- **Time-Based Splitting:**
    - **Hold-out Validation Set:** 2024-01-01 to 2024-12-31.
    - **Development Set (Training + Tuning):** 2016-01-01 to 2023-12-31.
- **Hyperparameter Tuning (Internal Split):**
    - Sub-split the Development Set:
        - Training: 2016 - 2021.
        - Tuning/Validation: 2022 - 2023.
    - Use this sub-split to find optimal hyperparameters (e.g., `num_leaves`, `learning_rate`, `feature_fraction`).
- **Final Training:** Retrain the model using the best hyperparameters on the full Development Set (2016-2023).
- **Threshold Optimization:** Determine the optimal probability threshold to achieve >60% precision while maintaining acceptable recall.

## Non-Functional Requirements
- **Reproducibility:** Use a fixed seed (42) for all randomized operations.
- **Performance:** Ensure training completes in a reasonable timeframe (utilizing LightGBM's efficiency).

## Evaluation & Outputs
The pipeline must generate the following artifacts in `TradeGuard/models/`:
1. **Model File:** `guard_model.txt` (LightGBM booster).
2. **Metadata:** `model_metadata.json` (including the optimal threshold and performance metrics).
3. **Visualizations:**
    - `confusion_matrix.png`
    - `feature_importance.png`
    - `calibration_curve.png`
    - `roc_curve.png`
4. **Console Report:** Summary of AUC, Precision, Recall, and F1 score on the 2024 hold-out set.

## Acceptance Criteria
- [ ] `train_guard.py` executes successfully from end-to-end.
- [ ] Model achieves an AUC > 0.65 on the 2024 hold-out set.
- [ ] All specified plots and metadata files are generated.
- [ ] The optimal threshold is saved and documented.

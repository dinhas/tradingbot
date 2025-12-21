# Implementation Plan: TradeGuard Phase 2 - Model Training

This plan follows the TDD-first workflow. Each implementation task is preceded by a test creation task.

## Phase 1: Data Preparation & Validation [checkpoint: ef9f3a8]
- [x] Task: Create tests for data loading and time-based splitting logic 635461b
- [x] Task: Implement `TradeGuard/src/train_guard.py` skeleton and data loading (2016-2024) 33fb529
- [x] Task: Implement time-based splitting logic (Train: 2016-2023, Val: 2024) 33fb529
- [x] Task: Conductor - User Manual Verification 'Data Preparation & Validation' (Protocol in workflow.md) ef9f3a8

## Phase 2: Hyperparameter Tuning (Internal Split) [checkpoint: c63915c]
- [x] Task: Create tests for internal tuning split logic (2016-2021 vs 2022-2023) 13c09f3
- [x] Task: Implement LightGBM hyperparameter tuning using the internal split 2c7bce3
- [x] Task: Verify that best parameters are correctly identified and logged fdb416c
- [x] Task: Conductor - User Manual Verification 'Hyperparameter Tuning' (Protocol in workflow.md) c63915c

## Phase 3: Final Training & Evaluation [checkpoint: 44cc7ce]
- [x] Task: Create tests for final model retraining on full 2016-2023 set 13a0e21
- [x] Task: Implement final training with best parameters f77f486
- [x] Task: Implement 2024 hold-out set evaluation and threshold optimization db87057
- [x] Task: Conductor - User Manual Verification 'Final Training & Evaluation' (Protocol in workflow.md) 0f18294

## Phase 4: Artifact Generation & Visualization [checkpoint: ]
- [ ] Task: Create tests for visualization generation (CM, FI, Calibration, ROC)
- [ ] Task: Implement plotting functions for Confusion Matrix and Feature Importance
- [ ] Task: Implement plotting functions for Calibration Curve and ROC Curve
- [ ] Task: Implement metadata export (`model_metadata.json` and `guard_model.txt`)
- [ ] Task: Conductor - User Manual Verification 'Artifact Generation & Visualization' (Protocol in workflow.md)

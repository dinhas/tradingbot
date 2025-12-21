# Implementation Plan: TradeGuard Phase 2 - Model Training

This plan follows the TDD-first workflow. Each implementation task is preceded by a test creation task.

## Phase 1: Data Preparation & Validation [checkpoint: ]
- [x] Task: Create tests for data loading and time-based splitting logic 635461b
- [~] Task: Implement `TradeGuard/src/train_guard.py` skeleton and data loading (2016-2024)
- [ ] Task: Implement time-based splitting logic (Train: 2016-2023, Val: 2024)
- [ ] Task: Conductor - User Manual Verification 'Data Preparation & Validation' (Protocol in workflow.md)

## Phase 2: Hyperparameter Tuning (Internal Split) [checkpoint: ]
- [ ] Task: Create tests for internal tuning split logic (2016-2021 vs 2022-2023)
- [ ] Task: Implement LightGBM hyperparameter tuning using the internal split
- [ ] Task: Verify that best parameters are correctly identified and logged
- [ ] Task: Conductor - User Manual Verification 'Hyperparameter Tuning' (Protocol in workflow.md)

## Phase 3: Final Training & Evaluation [checkpoint: ]
- [ ] Task: Create tests for final model retraining on full 2016-2023 set
- [ ] Task: Implement final training with best parameters
- [ ] Task: Implement 2024 hold-out set evaluation and threshold optimization
- [ ] Task: Conductor - User Manual Verification 'Final Training & Evaluation' (Protocol in workflow.md)

## Phase 4: Artifact Generation & Visualization [checkpoint: ]
- [ ] Task: Create tests for visualization generation (CM, FI, Calibration, ROC)
- [ ] Task: Implement plotting functions for Confusion Matrix and Feature Importance
- [ ] Task: Implement plotting functions for Calibration Curve and ROC Curve
- [ ] Task: Implement metadata export (`model_metadata.json` and `guard_model.txt`)
- [ ] Task: Conductor - User Manual Verification 'Artifact Generation & Visualization' (Protocol in workflow.md)

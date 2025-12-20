# Implementation Plan: TradeGuard Dataset Generation

This plan follows the TDD-first workflow. Each implementation task is preceded by a test creation task.

## Phase 1: Infrastructure & Data Loading [checkpoint: 193e7b4]
- [x] Task: Initialize `TradeGuard/src/generate_dataset.py` structure and logging [b457951]
- [x] Task: Create tests for historical data loading and alignment [4c81d0e]
- [x] Task: Implement OHLCV data loader for 5 assets (2016-2024) [1d2cd28]
- [x] Task: Create tests for Alpha model loading and inference consistency [e6e06c9]
- [x] Task: Implement Alpha model inference loop to identify trade signals [9adb5a0]
- [ ] Task: Conductor - User Manual Verification 'Infrastructure & Data Loading' (Protocol in workflow.md)

## Phase 2: Feature Engineering - Alpha, Risk & News (Features 1-20+)
- [x] Task: Create tests for Alpha Model Confidence features (1-10) [8f96bde]
- [x] Task: Create tests for Risk Model Output features (New: sl_mult, tp_mult, risk_factor) [487ac25]
- [x] Task: Implement Alpha Model Confidence features [62f3186]
- [x] Task: Implement Risk Model Output features [8cd895c]
- [x] Task: Create tests for Synthetic News Proxies features (11-20) [9b34521]
- [x] Task: Implement Synthetic News Proxies features [30679e3]
- [ ] Task: Conductor - User Manual Verification 'Feature Engineering - Alpha, Risk & News' (Protocol in workflow.md)

## Phase 3: Feature Engineering - Regime & Session (Features 21-40)
- [ ] Task: Create tests for Market Regime features (21-30)
- [ ] Task: Implement Market Regime features
- [ ] Task: Create tests for Session Edge features (31-40)
- [ ] Task: Implement Session Edge features
- [ ] Task: Conductor - User Manual Verification 'Feature Engineering - Regime & Session' (Protocol in workflow.md)

## Phase 4: Feature Engineering - Stats & Price Action (Features 41-60)
- [ ] Task: Create tests for Execution Statistics features (41-50, including SL/TP distances)
- [ ] Task: Implement Execution Statistics features
- [ ] Task: Create tests for Price Action Context features (51-60)
- [ ] Task: Implement Price Action Context features
- [ ] Task: Conductor - User Manual Verification 'Feature Engineering - Stats & Price Action' (Protocol in workflow.md)

## Phase 5: Labeling & Export
- [ ] Task: Create tests for Lookahead Labeling logic (Win/Loss)
- [ ] Task: Implement Ground Truth Labeling logic
- [ ] Task: Create tests for Dataset export integrity
- [ ] Task: Implement final Parquet export pipeline
- [ ] Task: Conductor - User Manual Verification 'Labeling & Export' (Protocol in workflow.md)

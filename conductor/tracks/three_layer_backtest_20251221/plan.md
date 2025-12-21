# Implementation Plan: Three-Layer Backtest

## Phase 1: Infrastructure & Model Loading
- [x] Task: Initialize `backtest_full_system.py` and implement model loader 4f2c26c
    - [ ] Write tests to verify LightGBM model and metadata loading (fail-fast behavior)
    - [ ] Implement `load_tradeguard_model` with schema validation
- [ ] Task: Conductor - User Manual Verification 'Infrastructure & Model Loading' (Protocol in workflow.md)

## Phase 2: Feature Engineering & Parity
- [ ] Task: Implement `TradeGuardFeatureBuilder` using existing preprocessing logic
    - [ ] Write tests to compare generated feature vectors against training dataset schema
    - [ ] Implement extraction logic from `generate_dataset.py` for all 6 feature groups
- [ ] Task: Conductor - User Manual Verification 'Feature Engineering & Parity' (Protocol in workflow.md)

## Phase 3: Integrated Backtest Loop
- [ ] Task: Modify `run_backtest` to include TradeGuard filtering
    - [ ] Write tests to verify that trades are blocked when probability is below threshold
    - [ ] Integrate `evaluate_tradeguard` into the main execution loop
- [ ] Task: Conductor - User Manual Verification 'Integrated Backtest Loop' (Protocol in workflow.md)

## Phase 4: Virtual Trade Simulation
- [ ] Task: Implement outcome simulation for blocked signals
    - [ ] Write tests to verify virtual PnL calculation against known outcomes
    - [ ] Implement logging for `blocked_trades` with theoretical results
- [ ] Task: Conductor - User Manual Verification 'Virtual Trade Simulation' (Protocol in workflow.md)

## Phase 5: Metrics & Comparative Analysis
- [ ] Task: Extend metrics engine for TradeGuard and Shadow Portfolio
    - [ ] Write tests for Net Value-Add and Block Accuracy calculations
    - [ ] Implement comparative metrics (Full System vs. Baseline)
- [ ] Task: Conductor - User Manual Verification 'Metrics & Comparative Analysis' (Protocol in workflow.md)

## Phase 6: Reporting & Visualization
- [ ] Task: Implement comprehensive visualization suite
    - [ ] Write tests for report file generation (JSON/CSV)
    - [ ] Implement all 5 required plot types including Time-of-Day analysis
- [ ] Task: Conductor - User Manual Verification 'Reporting & Visualization' (Protocol in workflow.md)

## Phase 7: Final Validation & CLI
- [ ] Task: Finalize CLI and perform end-to-end validation run
    - [ ] Write integration test for the full CLI command execution
    - [ ] Conduct final performance and quality gate checks
- [ ] Task: Conductor - User Manual Verification 'Final Validation & CLI' (Protocol in workflow.md)

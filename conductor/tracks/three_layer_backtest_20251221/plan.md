# Implementation Plan: Three-Layer Backtest

## Phase 1: Infrastructure & Model Loading [checkpoint: 0e3b665]
- [x] Task: Initialize `backtest_full_system.py` and implement model loader 4f2c26c
- [x] Task: Conductor - User Manual Verification 'Infrastructure & Model Loading' (Protocol in workflow.md) 0e3b665

## Phase 2: Feature Engineering & Parity [checkpoint: 5778811]
- [x] Task: Implement `TradeGuardFeatureBuilder` using existing preprocessing logic e6400ee
- [x] Task: Conductor - User Manual Verification 'Feature Engineering & Parity' (Protocol in workflow.md) 5778811

## Phase 3: Integrated Backtest Loop [checkpoint: 37facd6]
- [x] Task: Modify `run_backtest` to include TradeGuard filtering 37facd6
    - [x] Write tests to verify that trades are blocked when probability is below threshold 37facd6
    - [x] Integrate `evaluate_tradeguard` into the main execution loop 37facd6
- [x] Task: Conductor - User Manual Verification 'Integrated Backtest Loop' (Protocol in workflow.md) 37facd6


## Phase 4: Virtual Trade Simulation [checkpoint: c72389e]
- [x] Task: Implement outcome simulation for blocked signals c72389e
    - [x] Write tests to verify virtual PnL calculation against known outcomes c72389e
    - [x] Implement logging for `blocked_trades` with theoretical results c72389e
- [x] Task: Conductor - User Manual Verification 'Virtual Trade Simulation' (Protocol in workflow.md) c72389e


## Phase 5: Metrics & Comparative Analysis [checkpoint: 3042168]
- [x] Task: Extend metrics engine for TradeGuard and Shadow Portfolio 3042168
    - [x] Write tests for Net Value-Add and Block Accuracy calculations 3042168
    - [x] Implement comparative metrics (Full System vs. Baseline) 3042168
- [x] Task: Conductor - User Manual Verification 'Metrics & Comparative Analysis' (Protocol in workflow.md) 3042168


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

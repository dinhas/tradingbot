# Plan: TradeGuard Live Execution System

Implementation of the autonomous live trading system as defined in the PRD, integrating Alpha, Risk, and TradeGuard models with cTrader Open API.

## Phase 1: Core Infrastructure
- [x] Task: Project Scaffolding - Create `LiveExecution/` directory and basic structure. a27273e
- [x] Task: Configuration - Implement `.env` loading and validation for API and Discord credentials. bd67735
- [x] Task: Logging - Implement rotating file logger targeting `conductor/logs/` (DEBUG level). a07b2bb
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core Infrastructure' (Protocol in workflow.md) [checkpoint: 0e6028b]

## Phase 2: cTrader Communication Layer
- [x] Task: Connection - Implement Twisted-based WebSocket client for cTrader Open API. 22c60a3
- [~] Task: Authentication - Implement OAuth2 authentication and account authorization flows.
- [x] Task: Resilience - Implement reconnection logic with exponential backoff (5 retries). 2a18278
- [x] Task: Heartbeat - Implement keep-alive heartbeat to maintain connection. 770034f
- [x] Task: Conductor - User Manual Verification 'Phase 2: cTrader Communication Layer' (Protocol in workflow.md) [checkpoint: 8670df0]

## Phase 3: Data Acquisition & Features
- [x] Task: Event Handling - Implement `ProtoOATrendbarEvent` listener for M5 candle closes. 4d88bb6
- [x] Task: Data Fetching - Implement parallel OHLCV and account summary retrieval for the 5 target assets. 120bb0e
- [x] Task: Feature Engineering - Integrate logic imports from `Alpha`, `Risk`, and `TradeGuard` modules. 5ddc76f
- [x] Task: Conductor - User Manual Verification 'Phase 3: Data Acquisition & Features' (Protocol in workflow.md) [checkpoint: 5ca40ac]

## Phase 4: Inference Pipeline
- [x] Task: Model Loader - Implement loading for SB3 (Alpha, Risk) and LightGBM (TradeGuard) models. 25bbc63
- [x] Task: Logic Chain - Implement sequential inference: Alpha (Direction) -> Risk (Size, SL/TP) -> TradeGuard (Allow/Block). fbf917e
- [x] Task: Position Limits - Implement "Asset Locking" check (max 1 position per asset). bf2a2c9
- [x] Task: Conductor - User Manual Verification 'Phase 4: Inference Pipeline' (Protocol in workflow.md) [checkpoint: d1a6fdb]

## Phase 5: Execution & Notifications
- [ ] Task: Order Submission - Implement Market Order execution via `ProtoOANewOrderReq`.
- [ ] Task: Discord Integration - Implement webhook notifications for trades, blocks, and errors.
- [ ] Task: Error Handling - Implement rejection handling and execution confirmation logging.
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Execution & Notifications' (Protocol in workflow.md)

## Phase 6: Orchestration & Dockerization
- [ ] Task: System Integration - Implement the main event loop orchestrating data, inference, and execution.
- [ ] Task: Recovery Logic - Implement graceful shutdown and resume logic for M5 synchronization.
- [ ] Task: Containerization - Create `Dockerfile` and `docker-compose.yml` for VPS deployment.
- [ ] Task: Conductor - User Manual Verification 'Phase 6: Orchestration & Dockerization' (Protocol in workflow.md)

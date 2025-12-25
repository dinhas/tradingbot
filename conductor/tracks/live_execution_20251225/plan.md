# Plan: TradeGuard Live Execution System

Implementation of the autonomous live trading system as defined in the PRD, integrating Alpha, Risk, and TradeGuard models with cTrader Open API.

## Phase 1: Core Infrastructure
- [x] Task: Project Scaffolding - Create `LiveExecution/` directory and basic structure. a27273e
- [ ] Task: Configuration - Implement `.env` loading and validation for API and Discord credentials.
- [ ] Task: Logging - Implement rotating file logger targeting `conductor/logs/` (DEBUG level).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Core Infrastructure' (Protocol in workflow.md)

## Phase 2: cTrader Communication Layer
- [ ] Task: Connection - Implement Twisted-based WebSocket client for cTrader Open API.
- [ ] Task: Authentication - Implement OAuth2 authentication and account authorization flows.
- [ ] Task: Resilience - Implement reconnection logic with exponential backoff (5 retries).
- [ ] Task: Heartbeat - Implement keep-alive heartbeat to maintain connection.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: cTrader Communication Layer' (Protocol in workflow.md)

## Phase 3: Data Acquisition & Features
- [ ] Task: Event Handling - Implement `ProtoOATrendbarEvent` listener for M5 candle closes.
- [ ] Task: Data Fetching - Implement parallel OHLCV and account summary retrieval for the 5 target assets.
- [ ] Task: Feature Engineering - Integrate logic imports from `Alpha`, `Risk`, and `TradeGuard` modules.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Data Acquisition & Features' (Protocol in workflow.md)

## Phase 4: Inference Pipeline
- [ ] Task: Model Loader - Implement loading for SB3 (Alpha, Risk) and LightGBM (TradeGuard) models.
- [ ] Task: Logic Chain - Implement sequential inference: Alpha (Direction) -> Risk (Size, SL/TP) -> TradeGuard (Allow/Block).
- [ ] Task: Position Limits - Implement "Asset Locking" check (max 1 position per asset).
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Inference Pipeline' (Protocol in workflow.md)

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

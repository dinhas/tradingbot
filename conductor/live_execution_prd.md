# Product Requirements Document (PRD): TradeGuard Live Execution System

**Document Version:** 1.0  
**Last Updated:** 2025-12-25  
**Status:** Draft  
**Owner:** Trading System Development Team

---

## Executive Summary

This PRD defines the requirements for deploying a three-layer Reinforcement Learning trading system (Alpha → Risk → TradeGuard) into live production using the cTrader Open API. The system will operate autonomously on a 5-minute timeframe, managing up to 5 concurrent positions across major forex pairs and gold.

---

## 1. Project Overview

### 1.1 Objective
Deploy an autonomous trading system that combines three trained RL models to execute trades on a cTrader demo account, with the goal of validating live performance before transitioning to real capital.

### 1.2 Scope
- **In Scope:**
  - Real-time data ingestion from cTrader Open API
  - Sequential inference through Alpha, Risk, and TradeGuard models
  - Automated order execution (market orders only)
  - Position monitoring and feature extraction for open trades
  - Discord webhook notifications for all trading events
  - Error handling and graceful shutdown procedures
  - Paper trading validation phase

- **Out of Scope:**
  - Multi-timeframe analysis (only M5)
  - Asset universe expansion beyond 5 pairs
  - Limit orders or advanced order types
  - Portfolio rebalancing across assets
  - Manual trade intervention interface
  - Backtesting functionality (already completed)

### 1.3 Success Criteria
The system will be considered successful when:
1. **Uptime:** ≥ 99% during market hours (24/5)
2. **Latency:** 95th percentile execution time < 5 seconds (target: < 3 seconds)
3. **Performance Metrics (Live Trading):**
   - Sharpe Ratio: ≥ 8.0 (backtest: 11.35)
   - Max Drawdown: ≤ 20% (backtest: 14.4%)
   - Profit Factor: ≥ 2.5 (backtest: 3.79)
4. **Paper Trading Target:** Grow demo account from $10 to $5,000 before live deployment
5. **Zero Critical Failures:** No unhandled exceptions causing data loss or orphaned positions

---

## 2. System Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                     cTrader Open API                        │
│              (Protobuf over WebSocket)                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Event Loop (Twisted Reactor)                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  ProtoOATrendbarEvent Handler (M5 Candle Close)      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Acquisition Layer                     │
│  • Fetch latest OHLCV (5 assets, parallel)                  │
│  • Query account summary (equity, margin, positions)        │
│  • Retrieve open positions for feature engineering          │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                   │
│  ┌────────────┬────────────┬────────────────────────────┐   │
│  │ Alpha (140)│ Risk (165) │ TradeGuard (105)           │   │
│  │ features   │ features   │ features                   │   │
│  └────────────┴────────────┴────────────────────────────┘   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 Inference Chain (Sequential)                │
│  ┌──────────┐      ┌──────────┐      ┌──────────────┐      │
│  │  Alpha   │ ───▶ │   Risk   │ ───▶ │  TradeGuard  │      │
│  │ (8.03)   │      │  (2.15)  │      │   (manual)   │      │
│  │          │      │          │      │              │      │
│  │ Signal:  │      │ Output:  │      │ Decision:    │      │
│  │Direction │      │Size,SL/TP│      │ Allow/Block  │      │
│  └──────────┘      └──────────┘      └──────────────┘      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  Execution Layer                            │
│  • Asset locking check (1 position per asset max)           │
│  • Market order submission (allowed trades only)            │
│  • Order confirmation handling                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│              Notification & Logging Layer                   │
│  • Discord webhook (trade events, errors, status)           │
│  • File logging (DEBUG level, rotating logs)                │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Specifications

#### 2.2.1 Model Layer
| Model | Version | Input Features | Output | Framework |
|-------|---------|----------------|--------|-----------|
| Alpha | 8.03.zip | 140 | Direction (Long/Short/Hold) | SB3 PPO |
| Risk | 2.15.zip | 165 | Size, SL/TP | SB3 PPO |
| TradeGuard | manual_test_model.zip | 105 | Allow/Block | LightGBM |

**Feature Calculation:**
- Feature engineering logic will be extracted from existing implementations:
  - Alpha: `e:\tradingbot\Alpha\src\trading_env.py`
  - Risk: `e:\tradingbot\Risk\src\risk_env.py`
  - TradeGuard: `e:\tradingbot\TradeGuard\src\generate_dataset.py`

#### 2.2.2 API Integration
- **Protocol:** cTrader Open API (Protobuf over WebSocket)
- **Authentication:** OAuth2 with existing credentials
  - `CT_APP_ID`
  - `CT_APP_SECRET`
  - `CT_ACCESS_TOKEN`
  - `CT_ACCOUNT_ID`
- **Key Events:**
  - `ProtoOATrendbarEvent` (candle close trigger)
  - `ProtoOAExecutionEvent` (order confirmations)
  - `ProtoOAAccountAuthRes` (authentication)

---

## 3. Functional Requirements

### 3.1 Trading Logic

#### FR-1: Asset Universe
**Priority:** P0  
**Description:** The system shall trade exactly 5 assets:
1. EURUSD
2. GBPUSD
3. USDJPY
4. USDCHF
5. XAUUSD

**Acceptance Criteria:**
- No trades executed outside these 5 symbols
- System validates symbol before order submission

#### FR-2: Timeframe Synchronization
**Priority:** P0  
**Description:** The system shall execute its inference loop immediately upon M5 candle close.

**Acceptance Criteria:**
- Event handler triggers within 100ms of `ProtoOATrendbarEvent`
- Timestamp logging confirms candle close alignment

#### FR-3: Position Limits
**Priority:** P0  
**Description:** The system shall enforce a maximum of 1 open position per asset.

**Acceptance Criteria:**
- Before order submission, query open positions via API
- Skip trade if asset already has an open position
- Log skipped trades with reason "ASSET_LOCKED"

#### FR-4: Order Execution
**Priority:** P0  
**Description:** The system shall use market orders for instant execution.

**Acceptance Criteria:**
- No limit orders or pending orders
- Order type = `MARKET` in all API calls
- Execution confirmation received within 2 seconds

#### FR-5: Model Inference Chain
**Priority:** P0  
**Description:** The system shall execute models sequentially: Alpha → Risk → TradeGuard.

**Acceptance Criteria:**
- Alpha generates trade direction (Long/Short/Hold)
- Risk determines position size and SL/TP levels based on portfolio state
- TradeGuard makes final allow/block decision
- Blocked trades are logged but not executed

#### FR-6: Open Position Monitoring
**Priority:** P1  
**Description:** The system shall query open positions and include them in feature engineering.

**Acceptance Criteria:**
- Open positions retrieved on every candle close
- Position data (PnL, duration, direction) fed to Risk and TradeGuard features
- System does NOT modify or close positions from previous sessions
- Monitoring is read-only for feature extraction

---

### 3.2 Error Handling & Recovery

#### FR-7: API Connection Resilience
**Priority:** P0  
**Description:** The system shall handle API disconnections gracefully.

**Acceptance Criteria:**
- On connection drop, attempt reconnection with exponential backoff
- Retry up to 5 times with delays: 5s, 10s, 20s, 40s, 80s
- After 5 failed attempts, shut down gracefully with error notification
- Log all reconnection attempts

#### FR-8: Order Rejection Handling
**Priority:** P0  
**Description:** The system shall handle rejected orders without crashing.

**Acceptance Criteria:**
- Parse rejection reason from API response
- Send Discord notification with rejection details
- Skip the trade and continue monitoring next candle
- Do NOT retry rejected orders

#### FR-9: Crash Recovery
**Priority:** P1  
**Description:** On restart, the system shall resume from the next available M5 candle.

**Acceptance Criteria:**
- No attempt to "re-adopt" legacy trades
- State is reset (fresh start)
- Open positions from previous sessions are monitored but not managed
- System logs restart event with timestamp

---

### 3.3 Notifications & Monitoring

#### FR-10: Discord Notifications
**Priority:** P0  
**Description:** The system shall send Discord webhook notifications for all critical events.

**Events to Notify:**
- Trade executed (symbol, direction, size, SL/TP, entry price)
- Trade blocked by TradeGuard (symbol, reason)
- Order rejected (symbol, rejection reason)
- System startup/shutdown
- API connection errors
- Critical exceptions

**Acceptance Criteria:**
- Webhook URL configurable via environment variable
- Notifications sent within 500ms of event
- Failed webhook calls logged but do not block trading

#### FR-11: Logging
**Priority:** P1  
**Description:** The system shall maintain detailed logs for debugging and audit.

**Acceptance Criteria:**
- Log level: DEBUG
- Rotating file handler (max 50MB per file, keep 10 files)
- Log directory: `e:\tradingbot\conductor\logs\`
- Include timestamps, log levels, and contextual data

---

## 4. Non-Functional Requirements

### 4.1 Performance

#### NFR-1: Latency Budget (The "Golden Window")
**Priority:** P0  
**Target:** 95th percentile execution time < 3 seconds

**Breakdown:**
| Phase | Target Time | Description |
|-------|-------------|-------------|
| T+0ms | Trigger | `ProtoOATrendbarEvent` received |
| T+500ms | Data Fetch | OHLCV + account summary (parallel) |
| T+800ms | Feature Eng | Calculate 140+165+105 features |
| T+1000ms | Inference | Alpha → Risk → TradeGuard |
| T+1500ms | Order Submit | Market order to cTrader |
| T+2000ms | Notification | Discord webhook sent |

**Acceptance Criteria:**
- Log execution time for each phase
- Alert if total time exceeds 5 seconds
- Optimize bottlenecks if 95th percentile > 3s

#### NFR-2: Uptime
**Priority:** P0  
**Target:** ≥ 99% uptime during market hours (24/5)

**Acceptance Criteria:**
- Automated restart on crash (systemd/supervisor)
- Heartbeat monitoring every 5 minutes
- Downtime logged and reported weekly

### 4.2 Scalability
**Priority:** P2  
**Description:** System is designed for 5 assets only. No horizontal scaling required.

### 4.3 Security
**Priority:** P1  
**Requirements:**
- API credentials stored in environment variables (never hardcoded)
- Discord webhook URL in environment variables
- Logs do NOT contain sensitive credentials
- VPS access restricted to SSH key authentication

---

## 5. Technical Stack

### 5.1 Infrastructure
- **Hosting:** Cloud VPS (24/5 operation)
- **OS:** Linux (Ubuntu 22.04 LTS recommended)
- **Python Version:** 3.10+
- **Process Manager:** systemd or supervisor (auto-restart on failure)

### 5.2 Dependencies
```
stable-baselines3==2.0.0
lightgbm==4.0.0
twisted==23.10.0
protobuf==4.24.0
numpy==1.24.3
pandas==2.0.3
requests==2.31.0  # For Discord webhooks
python-dotenv==1.0.0
```

### 5.3 API & Protocols
- **cTrader Open API:** Protobuf over WebSocket
- **Discord Webhooks:** HTTPS POST requests

---

## 6. Data Requirements

### 6.1 Historical Data
- **Source:** cTrader Open API
- **Lookback:** Sufficient bars to calculate all features (typically 200+ M5 candles)
- **Storage:** In-memory circular buffer (no persistent storage required)

### 6.2 Feature Engineering
Features will be calculated using logic from:
- **Alpha (140 features):** `Alpha/src/trading_env.py`
- **Risk (165 features):** `Risk/src/risk_env.py`
- **TradeGuard (105 features):** `TradeGuard/src/generate_dataset.py`

**Critical:** Feature calculation must match training-time logic exactly to avoid distribution shift.

---

## 7. Deployment & Operations

### 7.1 Deployment Phases

#### Phase 1: Paper Trading (Demo Account)
**Duration:** Until account grows from $10 to $5,000  
**Objectives:**
- Validate live execution logic
- Confirm latency targets are met
- Monitor for unexpected errors or edge cases
- Verify performance metrics align with backtest

**Exit Criteria:**
- Demo account reaches $5,000 equity
- Sharpe ratio ≥ 8.0 over evaluation period
- Max drawdown ≤ 20%
- Zero critical failures in last 30 days

#### Phase 2: Live Trading (Real Account)
**Prerequisites:**
- Successful completion of Phase 1
- Manual review of all Phase 1 logs
- Confirmation of risk parameters

**Initial Capital:** [TBD]  
**Risk Limits:** Same as demo (1 position per asset, model-driven sizing)

### 7.2 Monitoring & Maintenance
- **Daily:** Review Discord notifications for anomalies
- **Weekly:** Analyze performance metrics (Sharpe, drawdown, win rate)
- **Monthly:** Model performance review (consider retraining if drift detected)

### 7.3 Rollback Plan
**Trigger Conditions:**
- Live Sharpe ratio < 5.0 for 7 consecutive days
- Max drawdown exceeds 25%
- Critical bug discovered

**Rollback Steps:**
1. Pause trading immediately (kill process)
2. Close all open positions manually via cTrader
3. Investigate root cause
4. Fix and redeploy to demo account
5. Re-validate before resuming live trading

---

## 8. Testing Strategy

### 8.1 Unit Tests
- Feature engineering functions (match training logic)
- Order validation logic (position limits, symbol checks)
- Error handling (API failures, rejections)

### 8.2 Integration Tests
- End-to-end inference chain (Alpha → Risk → TradeGuard)
- API connection and reconnection logic
- Discord webhook delivery

### 8.3 Paper Trading Validation
- Run on demo account for [DURATION] or until $[TARGET] reached
- Monitor for:
  - Execution errors
  - Latency violations
  - Feature calculation bugs
  - Unexpected model behavior

---

## 9. Risk Management

### 9.1 Trading Risks
| Risk | Mitigation |
|------|------------|
| Model overfitting | Paper trading validation phase |
| API downtime | Reconnection logic + graceful shutdown |
| Slippage on market orders | Accept as inherent risk; monitor execution prices |
| Flash crashes | Max 1 position per asset limits exposure |
| Orphaned positions on crash | Monitor-only approach for legacy trades |

### 9.2 Technical Risks
| Risk | Mitigation |
|------|------------|
| VPS failure | Automated restart via systemd |
| Network latency spikes | 5-second timeout tolerance |
| Model loading errors | Validate model files on startup |
| Feature calculation bugs | Unit tests + comparison to backtest |

---

## 10. Open Questions & Assumptions

### 10.1 Open Questions
   - Initial: $10
   - Target: $5,000

2. **VPS Specifications:** Any specific CPU/RAM requirements?
   - Assumption: 2 vCPU, 4GB RAM sufficient

3. **Discord Webhook Setup:** Will credentials be provided before deployment?
   - Assumption: Yes, during implementation phase

### 10.2 Assumptions
- cTrader API credentials are valid and have trading permissions on demo account
- Models (8.03.zip, 2.15.zip, manual_test_model.zip) are accessible and loadable
- Feature engineering code in Alpha/Risk/TradeGuard folders is production-ready
- No regulatory restrictions on automated trading in target jurisdiction

---

## 11. Appendix

### 11.1 Glossary
- **M5:** 5-minute timeframe
- **SL/TP:** Stop Loss / Take Profit
- **PPO:** Proximal Policy Optimization (RL algorithm)
- **VPS:** Virtual Private Server
- **Golden Window:** Target 3-second execution time from candle close to order placement

### 11.2 References
- cTrader Open API Documentation: [https://help.ctrader.com/open-api/](https://help.ctrader.com/open-api/)
- Stable-Baselines3 Docs: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
- Discord Webhook Guide: [https://discord.com/developers/docs/resources/webhook](https://discord.com/developers/docs/resources/webhook)

### 11.3 Backtest Performance Baseline
**Full System Backtest Results (2024-12-24):**
- **Sharpe Ratio:** 11.35
- **Max Drawdown:** -14.4%
- **Profit Factor:** 3.79
- **Total Return:** $248,793.02
- **Initial Capital:** $10

**Live Trading Targets:**
- Sharpe Ratio: ≥ 8.0 (70% of backtest)
- Max Drawdown: ≤ 20% (buffer for live conditions)
- Profit Factor: ≥ 2.5 (conservative target)

---

## 12. Approval & Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Product Owner | [TBD] | | |
| Technical Lead | [TBD] | | |
| Risk Manager | [TBD] | | |

---

**Document Control:**
- **Created:** 2025-12-25
- **Last Modified:** 2025-12-25
- **Next Review:** Upon completion of Phase 1 (Paper Trading)

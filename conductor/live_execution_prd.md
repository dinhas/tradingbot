# Product Requirements Document (PRD): Autonomous Live Execution System

**Document Version:** 2.0  
**Last Updated:** 2026-01-04  
**Status:** Current  
**Owner:** Trading System Development Team

---

## Executive Summary

This PRD defines the requirements for deploying a two-layer Reinforcement Learning trading system (Alpha → Risk) into live production using the cTrader Open API. The system will operate autonomously on a 5-minute timeframe, managing up to 2 concurrent positions across major forex pairs and gold.

---

## 1. Project Overview

### 1.1 Objective
Deploy an autonomous trading system that combines two trained RL models (Alpha and Risk) to execute trades on a cTrader account.

### 1.2 Scope
- **In Scope:**
  - Real-time data ingestion from cTrader Open API
  - Sequential inference through Alpha and Risk models
  - Automated order execution (market orders only)
  - Position monitoring and feature extraction for open trades
  - Discord webhook notifications for all trading events
  - Error handling and graceful shutdown procedures

- **Out of Scope:**
  - Multi-timeframe analysis (only M5)
  - Asset universe expansion beyond 5 pairs
  - Limit orders or advanced order types
  - Portfolio rebalancing across assets
  - Manual trade intervention interface

### 1.3 Success Criteria
The system will be considered successful when:
1. **Uptime:** ≥ 99% during market hours (24/5)
2. **Latency:** 95th percentile execution time < 3 seconds
3. **Performance Metrics:**
   - Sharpe Ratio: ≥ 8.0
   - Max Drawdown: ≤ 20%
   - Profit Factor: ≥ 2.5
4. **Zero Critical Failures:** No unhandled exceptions causing data loss or orphaned positions

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
│  ┌────────────┬────────────┐                                 │
│  │ Alpha (140)│ Risk (165) │                                 │
│  │ features   │ features   │                                 │
│  └────────────┴────────────┘                                 │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                 Inference Chain (Sequential)                │
│  ┌──────────┐      ┌──────────┐                              │
│  │  Alpha   │ ───▶ │   Risk   │                              │
│  │ Signal:  │      │ Output:  │                              │
│  │Direction │      │Size,SL/TP│                              │
│  └──────────┘      └──────────┘                              │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  Execution Layer                            │
│  • Asset locking check (1 position per asset max)           │
│  • System limits (Max 2 concurrent positions)               │
│  • Market order submission                                   │
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
| Model | Input Features | Output | Framework |
|-------|----------------|--------|-----------|
| Alpha | 140 | Direction (Long/Short/Hold) | SB3 PPO |
| Risk | 165 | Size, SL/TP | SB3 PPO |

**Feature Calculation:**
- Feature engineering logic is imported directly from:
  - Alpha: `Alpha/src/feature_engine.py`
  - Risk: `RiskLayer/src/feature_engine.py`

#### 2.2.2 API Integration
- **Protocol:** cTrader Open API (Protobuf over WebSocket)
- **Key Events:**
  - `ProtoOATrendbarEvent` (candle close trigger)
  - `ProtoOAExecutionEvent` (order confirmations)
  - `ProtoOAAccountAuthRes` (authentication)

---

## 3. Functional Requirements

### 3.1 Trading Logic

#### FR-1: Asset Universe
**Priority:** P0  
**Description:** The system shall trade 5 specific assets: EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD.

#### FR-2: Timeframe Synchronization
**Priority:** P0  
**Description:** Loop triggered by M5 candle close events.

#### FR-3: Position Limits
**Priority:** P0  
**Description:** Max 1 open position per asset. Max 2 concurrent positions across the entire system.

#### FR-4: Order Execution
**Priority:** P0  
**Description:** Use market orders for instant execution with automated SL/TP.

#### FR-5: Model Inference Chain
**Priority:** P0  
**Description:** Alpha model determines direction; Risk model determines parameters.

---

### 3.2 Error Handling & Recovery

#### FR-6: API Connection Resilience
**Priority:** P0  
**Description:** Automatic reconnection with exponential backoff.

#### FR-7: Order Rejection Handling
**Priority:** P0  
**Description:** Detailed logging and Discord notification for all rejections.

#### FR-8: Crash Recovery
**Priority:** P1  
**Description:** Automatic restart via system manager.

---

### 3.3 Notifications & Monitoring

#### FR-9: Discord Notifications
**Priority:** P0  
**Description:** Real-time alerts for trades, errors, and system status.

#### FR-10: Logging
**Priority:** P1  
**Description:** Rotating debug logs stored in `logs/` directory.

---

## 4. Technical Stack

- **OS:** Linux/Windows (Containerized)
- **Python Version:** 3.10+
- **RL Framework:** Stable-Baselines3 (PPO)
- **Asynchronous I/O:** Twisted (for Open API)
- **Containerization:** Docker & Docker Compose
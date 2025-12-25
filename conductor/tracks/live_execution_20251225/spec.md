# Specification: TradeGuard Live Execution System

## 1. Overview
Implementation of a production-ready live trading system using the cTrader Open API, integrating the Alpha, Risk, and TradeGuard RL models into a unified autonomous execution pipeline. The system will operate on a 5-minute timeframe (M5) for 5 specific forex/gold assets.

## 2. Functional Requirements
- **FR-1: Asset Universe:** Trade EURUSD, GBPUSD, USDJPY, USDCHF, and XAUUSD only.
- **FR-2: M5 Synchronization:** Trigger inference loop immediately upon `ProtoOATrendbarEvent` (candle close).
- **FR-3: Position Management:** Enforce a maximum of 1 open position per asset. Query API to ensure no "Asset Locking" violations.
- **FR-4: Execution:** Execute Market Orders via cTrader Open API.
- **FR-5: Inference Chain:**
    1. **Alpha (v8.03):** Generate Direction (Long/Short/Hold).
    2. **Risk (v2.15):** Calculate Position Size and SL/TP.
    3. **TradeGuard (manual):** Final Allow/Block decision.
- **FR-6: Logic Reuse:** Import feature engineering and environment logic directly from `Alpha/`, `Risk/`, and `TradeGuard/` directories.
- **FR-7: Crash Recovery:** Resume from the next M5 candle on restart. Monitoring-only mode for positions opened in previous sessions.
- **FR-8: Notifications:** Send trade events, errors, and status updates via Discord Webhook.

## 3. Non-Functional Requirements
- **NFR-1: Latency:** Aim for execution within 3 seconds of candle close (max 5 seconds).
- **NFR-2: Reliability:** Implement connection resilience with exponential backoff (up to 5 retries).
- **NFR-3: Portability:** Containerize the application using Docker for deployment to a Linux VPS.
- **NFR-4: Security:** Use environment variables for all API credentials and Webhook URLs.

## 4. Technical Constraints
- **API:** cTrader Open API (Protobuf over WebSocket) using `twisted`.
- **Runtime:** Python 3.10+ in a Docker container.
- **Credentials:** 
    - `DISCORD_WEBHOOK_URL` (Provided: https://discord.com/api/webhooks/1453576365176787138/...)
    - `CT_APP_ID`, `CT_APP_SECRET`, `CT_ACCESS_TOKEN`, `CT_ACCOUNT_ID` (To be configured via `.env`).

## 5. Acceptance Criteria
- [ ] System successfully authenticates with cTrader API.
- [ ] System triggers on M5 candle close for the 5 target assets.
- [ ] Sequential inference (Alpha -> Risk -> TradeGuard) produces a valid execution decision.
- [ ] Market orders are successfully placed on a cTrader demo account.
- [ ] Discord notifications are received for trades and system status.
- [ ] Application runs stably within a Docker container.

## 6. Out of Scope
- Management of legacy positions (trades not opened by the current session).
- Limit or Pending orders.
- Asset expansion beyond the specified 5 pairs.

# AI Trading Agent - Current Progress

**Version:** 1.0
**Date:** October 20, 2025

---

## 1. Project Goal

The primary objective is to build an autonomous, AI-powered scalping system that uses Google Gemini for trading decisions and executes trades via the cTrader Open API, as detailed in `projectreport.md`.

---

## 2. Current Status: Foundation Complete, Logic as Placeholders

The initial phase of development is complete. The legacy scripts have been successfully refactored into a robust, service-oriented architecture. The core services have been created with placeholder logic, establishing a clear framework for future implementation.

### Key Achievements:
- **Modular Architecture:** The codebase is now organized into distinct services (`services/`), utilities (`utils/`), and a central orchestrator (`main.py`).
- **Configuration:** All settings are managed externally in `config.json`.
- **cTrader Connection:** A stable, reusable `CTraderClient` in `services/cTrader/client.py` handles the connection, authentication, and message dispatching with the cTrader API.
- **Event Loop:** The application is built correctly on the Twisted framework, resolving previous event loop conflicts.
- **Core Services (Placeholders):**
    - `MarketDataService`: Can successfully fetch the list of all available symbols but lacks real-time data streaming and candle construction.
    - `AIEngine`: Exists as a placeholder and does not yet integrate with the Google Gemini API.
    - `RiskManager`: Contains placeholder methods for trade validation and position sizing without implementing the specific risk rules.
    - `TradeExecutor`: Outlines the structure for trade execution but does not yet create or send orders to the cTrader API.
- **Runnable Application:** The `main.py` script successfully starts, connects to cTrader, runs for 60 seconds, and shuts down cleanly, demonstrating the viability of the architecture.

**The project's foundational structure is now solid and ready for the implementation of core trading logic.**

---

## 3. Next Steps: Implementing Core Logic

The next phase of development will focus on replacing the placeholder logic with functional code as specified in the `projectreport.md`. The implementation will proceed in the following order:

1.  **Real-Time Market Data (`MarketDataService`):**
    *   Implement subscription to live spot price events (`ProtoOASubscribeSpotsReq`).
    *   Develop logic to aggregate incoming tick data into 5-minute candles.
    *   Cache the last 50 candles for technical analysis.

2.  **Technical Indicator Calculation (`utils/indicators.py`):**
    *   Create a new utility file for calculating technical indicators.
    *   Implement functions to calculate EMAs, RSI, Stochastics, Bollinger Bands, and ATR using `pandas-ta` on the 5-minute candle data.

3.  **AI Engine Integration (`AIEngine`):**
    *   Integrate the Google Gemini API client.
    *   Construct the minimal prompt for scalping decisions as defined in the project report.
    *   Implement the `get_decision` method to call the Gemini API with a strict 3-second timeout.
    *   Parse the JSON response from the AI into a structured decision object.

4.  **Risk Management (`RiskManager`):**
    *   Implement the `validate_trade` method to enforce the risk rules (confidence threshold, daily loss, max positions, etc.).
    *   Implement the `calculate_position_size` method based on the risk-per-trade percentage and stop loss.

5.  **Trade Execution (`TradeExecutor`):**
    *   Implement the `execute_trade` method to create and send `ProtoOACreateOrderReq` messages for market orders.
    *   Ensure that stop loss and take profit levels are included in the order request.

6.  **Position Monitoring (`main.py` or a new `PositionManager` service):**
    *   Implement a monitoring loop that runs every 5 seconds.
    *   Add logic to check open positions for breakeven adjustments (move stop to entry after 5 pips profit).
    *   Implement partial closing of positions at the first take-profit level.
    *   Add a time-based stop to close positions held for longer than 60 minutes.

7.  **Testing & Monitoring:**
    *   Develop unit tests for each service's logic.
    *   Begin development of the real-time monitoring dashboard as a separate FastAPI application.

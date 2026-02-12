# Instructions for Upgrading LiveExecution System

The goal of this task is to upgrade the `LiveExecution` module of the trading bot. This involves transitioning the Risk Model from Reinforcement Learning (RL) to Supervised Learning (SL), moving the notification system from Discord to Telegram, implementing a local logging system for trade tracing, and creating a web dashboard for monitoring.

## Core Tasks

### 1. Risk Model Upgrade (RL -> SL)
The current system uses a PPO-based RL model for risk. You must replace this with the new Supervised Learning model defined in `RiskLayer/src/risk_model_sl.py`.

*   **Model Loading:** Update `LiveExecution/src/models.py` to load the PyTorch `RiskModelSL`. You will need to load `models/risk_model_sl_best.pth` and the `models/sl_risk_scaler.pkl` scaler.
*   **Feature Engineering:** Update `LiveExecution/src/features.py` to produce the **60-dimensional observation** expected by the SL model.
    *   [25] Asset Features: (Current price action, ATR, indicators).
    *   [15] Global Features: (Session, hour, cross-asset correlations).
    *   [5] Account State: `[equity_norm, drawdown, margin_usage, risk_cap_mult, padding]`.
    *   [5] History PnL: Realized PnL % of the last 5 trades for the specific asset.
    *   [10] History Actions: The `sl_mult` and `tp_mult` used in the last 5 trades.
*   **Inference Integration:** Update `Orchestrator.run_inference_chain` to handle the dictionary output of the SL model (`sl`, `tp`, `size`).
    *   `sl_mult`: Used to calculate Stop Loss distance.
    *   `tp_mult`: Used to calculate Take Profit distance.
    *   `size`: Used as the "Confidence" or "Risk Percentage" (replaces the previous `risk_raw` logic).

### 2. Telegram Messaging System
Replace the existing Discord-based notification system with Telegram.

*   **Implementation:** Create `LiveExecution/src/notifications_telegram.py` using `python-telegram-bot` (or a similar library).
*   **Functionality:** 
    *   Send alerts for trade openings and closures (with realized PnL).
    *   Send system error alerts.
    *   Provide a `/status` command to check current balance and open positions.
*   **Config:** Add `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` to `.env` and `config.py`.

### 3. Local Logging & 1-Minute Polling
Implement a robust logging system to trace trades and account state for the dashboard.

*   **Poller:** Add a background task in `CTraderClient` or `Orchestrator` that runs every 60 seconds.
*   **Data to Fetch:** Current balance, equity, and a list of all open positions.
*   **Storage:** Save this data to a local SQLite database (`live_trading.db`).
    *   Table `account_history`: `timestamp, balance, equity, margin_used`.
    *   Table `trades`: `trade_id, symbol, side, entry_price, size, sl, tp, status (OPEN/CLOSED), exit_price, pnl`.

### 4. Web Dashboard
Create a lightweight web dashboard to monitor the bot's performance.

*   **Backend:** Use **FastAPI** to serve both the dashboard and a small API.
*   **Features:**
    *   **Live Overview:** Current Equity, Balance, and Today's PnL.
    *   **Equity Chart:** Visualize account equity over time (using data from SQLite).
    *   **Active Positions:** A table showing currently open trades with real-time unrealized PnL.
    *   **Trade History:** A searchable table of past trades.
*   **Visuals:** Use a modern, dark-themed UI (Tailwind CSS or Bootstrap).

### 5. Notification System Improvements (Ideas to Implement)
Enhance the Telegram bot to be more proactive:
*   **Daily Summary:** Automatically send a message at the end of each trading day with a summary of trades and total PnL.
*   **System Health Check:** Send a "Heartbeat" message every 4-8 hours to confirm the server is running and the cTrader connection is active.
*   **Threshold Alerts:** Notify if drawdown exceeds a certain percentage (e.g., 2%, 5%).

## Technical Notes
*   **Environment:** The system runs on Windows. Ensure all paths and shell commands are compatible.
*   **Async Logic:** `CTraderClient` uses Twisted. Ensure your poller and dashboard logic play nicely with the Twisted reactor (e.g., using `crochet` or running the dashboard in a separate process/thread).
*   **Matching Backtest:** Ensure the SL/TP and Lot calculation logic in `orchestrator.py` exactly matches the labeling logic in `RiskLayer/generate_sl_dataset.py` for consistency.

Please implement these changes incrementally and verify each part.

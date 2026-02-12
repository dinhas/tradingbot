# Live Execution Upgrade Plan: Web Dashboard, Telegram & SL Risk Model

This document serves as a comprehensive instruction set for the AI agent responsible for upgrading the `LiveExecution` system.

## 🚀 Overview
The upgrade transforms the current live execution environment into a more robust, traceable, and user-friendly system. Key changes include moving to Supervised Learning for risk management, replacing Discord with Telegram, and adding a local monitoring dashboard.

## 🛠️ Task 1: SL Risk Model Integration
**Goal:** Replace the PPO (RL) risk model with the `RiskModelSL` (PyTorch).

- **Implementation:**
  - Update `LiveExecution/src/models.py` to support PyTorch model loading.
  - Load `models/risk_model_sl_best.pth` and `models/sl_risk_scaler.pkl`.
  - Update `LiveExecution/src/features.py` to generate the required **60-feature vector**.
  - Update `Orchestrator.py` to use multi-head outputs: `sl_mult`, `tp_mult`, and `size`.

## 📱 Task 2: Telegram Messaging System
**Goal:** Migrate notifications from Discord to Telegram for better accessibility and command support.

- **Features:**
  - Real-time trade opening/closing alerts.
  - System health and error notifications.
  - Commands: `/status` (Account summary), `/positions` (Open trades).
- **Library:** `python-telegram-bot`.

## 📊 Task 3: Local Logging & 1-Minute Polling
**Goal:** Create a persistent record of account state and trades for the dashboard.

- **Mechanism:** Implement a 60-second polling loop in the orchestrator.
- **Storage:** Use a local **SQLite** database (`live_trading.db`).
- **Data Points:** Equity, Balance, Drawdown, Margin, and full trade lifecycle (Entry -> SL/TP -> Exit).

## 🖥️ Task 4: Web Dashboard
**Goal:** Provide a visual interface for monitoring the bot.

- **Tech Stack:** FastAPI (Backend) + Jinja2/Tailwind (Frontend).
- **Components:**
  - **Account Summary:** Real-time balance/equity/drawdown cards.
  - **Performance Chart:** Equity curve over time.
  - **Trade Log:** Table of recent and active trades.
  - **Connection Status:** Indicator for cTrader API health.

## 💡 Task 5: Notification Improvements
- **Real-time PnL:** Send updates when a trade hits a certain PnL % or milestone.
- **Daily Summaries:** A report at 00:00 UTC summarizing the day's performance.
- **Pulse Checks:** A message every 4 hours confirming "All systems nominal".

---

### Execution Steps for the AI Agent:
1. **Analyze** `RiskLayer/src/risk_model_sl.py` and `RiskLayer/generate_sl_dataset.py` to ensure feature consistency.
2. **Modify** `LiveExecution/src/models.py` to add PyTorch support.
3. **Refactor** `LiveExecution/src/features.py` for the 60-dim risk observation.
4. **Implement** `LiveExecution/src/notifications.py` (Telegram).
5. **Develop** the SQLite logging service and the 1-min poller.
6. **Build** the FastAPI dashboard in a new `LiveExecution/dashboard/` directory.
7. **Verify** that inference logic matches backtest labeling exactly.

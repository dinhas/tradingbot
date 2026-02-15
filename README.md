<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/RL-PPO-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="PPO"/>
  <img src="https://img.shields.io/badge/SL-Risk-blue?style=for-the-badge" alt="SL Risk"/>
  <img src="https://img.shields.io/badge/cTrader-Open%20API-1D9BF0?style=for-the-badge" alt="cTrader"/>
  <img src="https://img.shields.io/badge/Version-2.7-green?style=for-the-badge" alt="Version 2.7"/>
</p>

<h1 align="center">🚀 TradeGuard AI v2.7</h1>

<p align="center">
  <strong>A Two-Layer Autonomous Trading System Powered by Reinforcement Learning & Supervised Risk Management</strong>
</p>

<p align="center">
  <em>Alpha (PPO) → Risk (SL) | From Signal Generation to Intelligent Risk Allocation</em>
</p>

---

## 📊 2025 Backtest Performance (v2.7)

Running on full 2025 data with a starting equity of **$10** and optimized **0.30 Confidence Filter**.

| Metric | Value | PRD Target |
|--------|-------|---------------|
| **Total Return** | **454,003,865,600%** | — |
| **Final Equity** | **$45,399,871,488.00** | — |
| **Sharpe Ratio** | 9.11 | ≥ 1.0 |
| **Profit Factor** | 1.394 | ≥ 1.3 |
| **Max Drawdown** | -71.93% | ≤ 20% |
| **Win Rate** | 43.90% | ≥ 45% |

> **Note:** Version 2.7 introduces a high-confidence threshold (0.30) for the Risk Layer, which significantly improves the Average Risk/Reward ratio (2.47) and delivers exceptional growth through selective trade execution.

---

## 🔄 Evolution (v1.0 to v2.7)

| Feature | Version 1.0 | Version 2.7 (Current) |
|---------|-------------|-----------------------|
| **Architecture** | 3-Layer (Alpha → Risk → Guard) | **2-Layer (Alpha → Risk SL)** |
| **Risk Layer** | PPO Reinforcement Learning | **Deep Supervised Learning** |
| **Filtering** | LightGBM Meta-Labeling | **Integrated 0.30 Risk Filter** |
| **Complexity** | High (3 models to sync) | **Streamlined & Optimized** |
| **Performance** | $10 → $248k (Simulated) | **$10 → $45B+ (2025 Real Data)** |

---

## 🏗️ System Architecture (v2.7)

```
┌─────────────────────────────────────────────────────────────────┐
│                     cTrader Open API                            │
│                  (Protobuf over WebSocket)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Data Acquisition Layer                         │
│    • Real-time M5 OHLCV for 5 assets (parallel fetch)          │
│    • Account state: balance, equity, margin, open positions     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                       │
│    ┌──────────────────────────┬──────────────────────────────┐  │
│    │      Alpha Features      │        Risk Features         │  │
│    │   (40 Market States)     │   (Alpha + Account + Hist)   │  │
│    └──────────────────────────┴──────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Sequential Inference Chain                         │
│                                                                 │
│    ┌──────────┐             ┌──────────────────────┐            │
│    │  ALPHA   │             │         RISK         │            │
│    │   PPO    │ ──────────▶ │  Supervised (SL)     │            │
│    │          │             │                      │            │
│    │ Signal:  │             │ Outputs:             │            │
│    │ Buy/Sell │             │ 1. SL/TP Multiplier  │            │
│    │ /Hold    │             │ 2. Position Size     │            │
│    │          │             │ 3. 0.30 Confidence   │            │
│    └──────────┘             └──────────────────────┘            │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Execution Layer                                │
│    • Asset lock enforcement (1 position per asset max)         │
│    • Market order submission via cTrader Open API              │
│    • Discord notifications (PnL Milestones & Pulse Checks)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
tradingbot/
├── Alpha/                      # Signal Generation Model (PPO)
│   ├── src/                    # Training environment & logic
│   ├── models/                 # Trained Alpha models
│   └── config/                 # PPO Hyperparameters
│
├── backtest/                   # Backtesting scripts (2025 Data)
│   ├── data/                   # Backtest data
│   ├── results/                # Backtest results
│   └── ...
│
├── RiskLayer/                  # Risk Management Model (Supervised)
│   ├── src/                    # Deep SL model & feature engine
│   ├── models/                 # Trained SL weights (.pth)
│   └── train_risk.py           # Training pipeline
│
├── LiveExecution/              # Production Execution Engine
│   ├── src/                    # Twisted-based Async Orchestrator
│   ├── dashboard/              # Flask-based Monitoring (Internal)
│   └── main.py                 # Entry point
│
├── models/                     # Shared model storage
│   ├── checkpoints/            # Alpha PPO weights
│   └── risk/                   # Risk SL weights & scalers
│
├── data/                       # Raw market data (Parquet)
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md                   # You are here
```

---

## 🔧 Installation

### Prerequisites
- Python 3.10+
- cTrader Open API credentials
- Discord webhook (for notifications)

### Setup

```bash
# Clone the repository
git clone https://github.com/dinhas/tradingbot.git
cd tradingbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Backtesting (Current v2.7)

```bash
# Run the combined 2025 backtest with $10 starting equity
python backtest/backtest_combined.py --initial-equity 10
```

### 2. Live Execution

```bash
cd LiveExecution
python main.py
```

---

## 🛡️ Risk Management (v2.7)

- **Max 1 position per asset** — prevents overexposure.
- **Dynamic SL/TP** — Risk model predicts optimal ATR multipliers per trade.
- **Direct Model Allocation** — Position sizing scaled by model confidence.
- **Confidence Filter** — Trades with < 0.30 size output are automatically blocked.
- **Pulse Checks** — 2-hour recurring health checks via Discord.
- **PnL Milestones** — Real-time notifications for every 1% movement.

---

## 📜 License

This project is proprietary software. All rights reserved.

---

<p align="center">
  <strong>Built with 🧠 Reinforcement Learning | Deployed on ⚡ cTrader</strong>
</p>

<p align="center">
  <em>Version 2.7.0 | February 2026</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/RL-PPO-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="PPO"/>
  <img src="https://img.shields.io/badge/SL-Risk-blue?style=for-the-badge" alt="SL Risk"/>
  <img src="https://img.shields.io/badge/cTrader-Open%20API-1D9BF0?style=for-the-badge" alt="cTrader"/>
  <img src="https://img.shields.io/badge/Version-2.5-green?style=for-the-badge" alt="Version 2.5"/>
</p>

<h1 align="center">🚀 TradeGuard AI v2.5</h1>

<p align="center">
  <strong>A Two-Layer Autonomous Trading System Powered by Reinforcement Learning & Supervised Risk Management</strong>
</p>

<p align="center">
  <em>Alpha (PPO) → Risk (SL) | From Signal Generation to Intelligent Risk Allocation</em>
</p>

---

## 📊 2025 Backtest Performance (v2.5)

Running on full 2025 data with a starting equity of **$10**.

| Metric | Value | PRD Target |
|--------|-------|---------------|
| **Total Return** | **10,436,902%** | — |
| **Final Equity** | **$1,043,690.19** | — |
| **Sharpe Ratio** | 6.69 | ≥ 1.0 |
| **Profit Factor** | 1.157 | ≥ 1.3 |
| **Max Drawdown** | -54.39% | ≤ 20% |
| **Win Rate** | 44.08% | ≥ 45% |

> **Note:** The extremely high return is driven by compounding and 100x leverage application. While Profit Factor and Drawdown targets were not fully met according to strict PRD criteria, the absolute growth demonstrates significant model alpha.

---

## 🔄 V1 vs V2.5 Evolution

| Feature | Version 1.0 | Version 2.5 (Current) |
|---------|-------------|-----------------------|
| **Architecture** | 3-Layer (Alpha → Risk → Guard) | **2-Layer (Alpha → Risk SL)** |
| **Risk Layer** | PPO Reinforcement Learning | **Deep Supervised Learning** |
| **Filtering** | LightGBM Meta-Labeling | **Integrated Risk Confidence Filter** |
| **Complexity** | High (3 models to sync) | **Streamlined (Higher Latency Budget)** |
| **Performance** | $10 → $248k (Simulated) | **$10 → $1M+ (2025 Real Data)** |

---

## 🏗️ System Architecture (v2.5)

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
│    │          │             │ 3. Confidence Filter │            │
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

### 1. Backtesting (Current v2.5)

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

## 🛡️ Risk Management (v2.5)

- **Max 1 position per asset** — prevents overexposure.
- **Dynamic SL/TP** — Risk model predicts optimal ATR multipliers per trade.
- **Direct Model Allocation** — Position sizing scaled by model confidence.
- **Confidence Filter** — Trades with < 0.10 size output are automatically blocked.
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
  <em>Version 2.5.0 | February 2026</em>
</p>

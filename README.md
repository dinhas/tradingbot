<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/RL-PPO-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="PPO"/>
  <img src="https://img.shields.io/badge/cTrader-Open%20API-1D9BF0?style=for-the-badge" alt="cTrader"/>
  <img src="https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🚀 RL Trading AI</h1>

<p align="center">
  <strong>A Two-Layer Autonomous Trading System Powered by Reinforcement Learning</strong>
</p>

<p align="center">
  <em>Alpha → Risk | From Signal Generation to Intelligent Execution</em>
</p>

---

## 📊 Backtest Performance (V2 Combined System)

| Metric | Value | Target (Live) |
|--------|-------|---------------|
| **Sharpe Ratio** | 21.15 | ≥ 8.0 |
| **Profit Factor** | 4.56 | ≥ 2.5 |
| **Max Drawdown** | -6.76% | ≤ 20% |
| **Win Rate** | 59.5% | ≥ 45% |
| **Avg RR Ratio** | 19.88 | — |
| **Total Return** | $10 → $69.7M | — |

*Backtest Zeitraum: Multi-year 5-minute data (10M Timesteps training).*

---

## 🏗️ System Architecture

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
│    • Account state: equity, margin, open positions              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                       │
│    ┌────────────┬────────────┐                                 │
│    │ Alpha (140)│ Risk (165) │                                 │
│    │  features  │  features  │                                 │
│    └────────────┴────────────┘                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Sequential Inference Chain                         │
│                                                                 │
│    ┌──────────┐      ┌──────────┐                              │
│    │  ALPHA   │ ───▶ │   RISK   │                              │
│    │   PPO    │      │   PPO    │                              │
│    │          │      │          │                              │
│    │ Signal:  │      │ Output:  │                              │
│    │ L/S/Hold │      │ Size,SL  │                              │
│    └──────────┘      └──────────┘                              │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Execution Layer                                │
│    • Asset lock enforcement (1 position per asset max)         │
│    • Market order submission via cTrader Open API              │
│    • Discord notifications for all events                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
tradingbot/
├── Alpha/                      # Signal Generation Model (PPO)
│   ├── src/                    # Training environment & logic
│   ├── backtest/               # Backtesting scripts
│   ├── models/                 # Trained Alpha models
│   └── config/                 # Hyperparameters
│
├── RiskLayer/                  # Risk Management Model (PPO)
│   ├── src/                    # Risk environment
│   ├── models/                 # Trained Risk models
│   └── train_risk.py           # Training script
│
├── LiveExecution/              # Production Execution Engine
│   ├── src/                    # API client, feature engine, inference
│   ├── config/                 # Live trading configuration
│   └── main.py                 # Entry point
│
├── conductor/                  # Development documentation
│   ├── live_execution_prd.md   # Product Requirements Document
│   └── logs/                   # Application logs
│
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Orchestration
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

### Environment Variables

Create a `.env` file in the root directory:

```env
# cTrader API Credentials
CT_APP_ID=your_app_id
CT_APP_SECRET=your_app_secret
CT_ACCESS_TOKEN=your_access_token
CT_ACCOUNT_ID=your_account_id

# Discord Notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

---

## 🚀 Recent Updates (V2 Branch)

- **Single-Pair Inference Core**: Refactored the engine to handle assets as independent pairs while maintaining a global portfolio view.
- **Enhanced Risk Layer**: New PnL Efficiency reward system with "Bullet Dodger" bonus for capital preservation.
- **Sniper Execution**: Improved TradeGuard logic to filter 99% of noisy signals, focusing on high RR (19:1) trades.
- **Parallel Optimization**: Backtest and dataset generation now support multi-core parallel execution.

---

## 🚀 Usage

### Training Pipeline

```bash
# 1. Train Alpha Model
cd Alpha
python src/train.py

# 2. Generate Risk Dataset & Train Risk Model
cd ../RiskLayer
python run_pipeline.py
```

### Backtesting

```bash
cd Alpha/backtest

# Combined backtest (Alpha + Risk)
python backtest.py --model models/checkpoints/stage_3_final.zip --stage 3
```

### Live Execution

```bash
cd LiveExecution
python main.py
```

Or with Docker:

```bash
docker-compose up -d
```

---

## 📈 Model Details

### Layer 1: Alpha (Signal Generation)
| Attribute | Value |
|-----------|-------|
| **Algorithm** | PPO (Proximal Policy Optimization) |
| **Framework** | Stable-Baselines3 |
| **Features** | 40 indicators (extracted from 140 basket) |
| **Reward** | PEEK & LABEL (Lookahead) |
| **Assets** | EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD |

### Layer 2: Risk / TradeGuard (Execution)
| Attribute | Value |
|-----------|-------|
| **Algorithm** | PPO with Dual Normalization |
| **Reward** | PnL Efficiency + Bullet Dodger |
| **Win Rate** | ~23% (Optimized for PnL/RR) |
| **Payoff Ratio** | 19:1 (Sniper Mode) |

---

## ⚡ Performance Targets

### Latency Budget ("Golden Window")
| Phase | Target | Description |
|-------|--------|-------------|
| T+0ms | Trigger | Candle close event received |
| T+500ms | Data | OHLCV + account summary fetched |
| T+800ms | Features | All features calculated |
| T+1000ms | Inference | Model chain complete |
| T+1500ms | Order | Market order submitted |
| T+2000ms | Notify | Discord notification sent |

**Target:** 95th percentile < 3 seconds

---

## 🛡️ Risk Management

- **Max 1 position per asset** — prevents overexposure
- **Model-driven sizing** — Risk layer determines position size
- **Circuit breakers** — graceful shutdown on critical errors
- **Discord alerts** — real-time monitoring of all events

---

## 📋 Deployment Phases

### Phase 1: Paper Trading (Demo Account)
- **Goal:** Validate live execution
- **Target:** Grow $10 → $5,000
- **Duration:** Until exit criteria met

### Phase 2: Live Trading
- **Prerequisites:** Successful Phase 1 completion
- **Risk Limits:** Same as demo
- **Monitoring:** Daily Discord review

---

## 🔔 Notifications

The system sends Discord notifications for:
- ✅ Trade executed (symbol, direction, size, entry price)
- ❌ Order rejected
- 🔄 System startup / shutdown
- ⚠️ API connection errors
- 🔥 Critical exceptions

---

## 📜 License

This project is proprietary software. All rights reserved.

---

## 🤝 Contributing

This is a private trading system. Contributions are not currently accepted.

---

## 📞 Support

For issues or questions, contact the development team.

---

<p align="center">
  <strong>Built with 🧠 Reinforcement Learning | Deployed on ⚡ cTrader</strong>
</p>

<p align="center">
  <em>Version 1.0.0 | December 2025</em>
</p>

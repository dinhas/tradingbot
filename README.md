<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/RL-PPO-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="PPO"/>
  <img src="https://img.shields.io/badge/LightGBM-Meta--Labeling-9ACD32?style=for-the-badge" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/cTrader-Open%20API-1D9BF0?style=for-the-badge" alt="cTrader"/>
  <img src="https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🚀 TradeGuard AI</h1>

<p align="center">
  <strong>A Three-Layer Autonomous Trading System Powered by Reinforcement Learning</strong>
</p>

<p align="center">
  <em>Alpha → Risk → TradeGuard | From Signal Generation to Intelligent Execution</em>
</p>

---

## 📊 Backtest Performance

| Metric | Value | Target (Live) |
|--------|-------|---------------|
| **Sharpe Ratio** | 11.35 | ≥ 8.0 |
| **Max Drawdown** | -14.4% | ≤ 20% |
| **Profit Factor** | 3.79 | ≥ 2.5 |
| **Total Return** | $10 → $248,793 | — |

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
│    ┌────────────┬────────────┬─────────────────┐               │
│    │ Alpha (140)│ Risk (165) │ TradeGuard(105) │               │
│    │  features  │  features  │    features     │               │
│    └────────────┴────────────┴─────────────────┘               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Sequential Inference Chain                         │
│                                                                 │
│    ┌──────────┐      ┌──────────┐      ┌──────────────┐        │
│    │  ALPHA   │ ───▶ │   RISK   │ ───▶ │  TRADEGUARD  │        │
│    │   PPO    │      │   PPO    │      │   LightGBM   │        │
│    │          │      │          │      │              │        │
│    │ Signal:  │      │ Output:  │      │ Decision:    │        │
│    │ L/S/Hold │      │ Size,SL  │      │ Allow/Block  │        │
│    └──────────┘      └──────────┘      └──────────────┘        │
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
├── TradeGuard/                 # Meta-Labeling Filter (LightGBM)
│   ├── src/                    # Dataset generation & training
│   ├── models/                 # Trained TradeGuard model
│   └── config/                 # LightGBM config
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

## 🚀 Usage

### Training Pipeline

```bash
# 1. Train Alpha Model
cd Alpha
python src/train.py

# 2. Generate Risk Dataset & Train Risk Model
cd ../RiskLayer
python run_pipeline.py

# 3. Generate TradeGuard Dataset & Train
cd ../TradeGuard
python run_pipeline.py
```

### Backtesting

```bash
cd Alpha/backtest

# Combined backtest (Alpha + Risk + TradeGuard)
python backtest_full_system.py
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
| **Features** | 140 technical indicators |
| **Output** | Direction (Long / Short / Hold) |
| **Assets** | EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD |
| **Timeframe** | M5 (5-minute) |

### Layer 2: Risk (Position Sizing)
| Attribute | Value |
|-----------|-------|
| **Algorithm** | PPO |
| **Framework** | Stable-Baselines3 |
| **Features** | 165 (Alpha features + portfolio state) |
| **Output** | Position size, Stop-Loss, Take-Profit |

### Layer 3: TradeGuard (Meta-Labeling)
| Attribute | Value |
|-----------|-------|
| **Algorithm** | LightGBM Classifier |
| **Features** | 105 (trade context + market regime) |
| **Output** | Allow / Block trade decision |
| **Purpose** | Filter low-quality signals |

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
- **TradeGuard filter** — blocks low-conviction trades
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
- 🚫 Trade blocked by TradeGuard
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

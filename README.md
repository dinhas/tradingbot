---
title: TradeGuard AI
emoji: 📈
colorFrom: green
colorTo: blue
sdk: python
app_port: 7860
pinned: false
---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/LSTM-Attention-FF6F00?style=for-the-badge" alt="LSTM+Attention"/>
  <img src="https://img.shields.io/badge/cTrader-Open%20API-1D9BF0?style=for-the-badge" alt="cTrader"/>
  <img src="https://img.shields.io/badge/Version-2.0-green?style=for-the-badge" alt="Version 2.0"/>
</p>

<h1 align="center">TradeGuard AI v2.0</h1>

<p align="center">
  <strong>Autonomous Trading System with LSTM Attention & Fixed Risk Management</strong>
</p>

<p align="center">
  <em>Alpha (LSTM+Attention) → Execution (Fixed ATR Multipliers) | From Signal Generation to Intelligent Execution</em>
</p>

---

## System Overview

TradeGuard AI is a fully automated forex trading system that operates on 4 major currency pairs (EURUSD, GBPUSD, USDJPY, USDCHF) using M5 candles. The system uses an LSTM model with multi-head attention to generate buy/sell signals, with fixed ATR-based risk management.

---

## Latest Backtest Performance (v2.0)

Running with **Alpha LSTM+Attention** model on 2025 data with **$10,000 initial equity**.

| Metric | Value | Target |
|--------|-------|--------|
| **Profit Factor** | **3.16** | ≥ 1.5 |
| **Sharpe Ratio** | **34.64** | ≥ 2.0 |
| **Win Rate** | **65.8%** | ≥ 55% |
| **Max Drawdown** | **-8.26%** | ≤ 15% |
| **Total Return** | **5,373%** | — |
| **Total Trades** | 1,934 | — |
| **Avg Hold Time** | 111 min | — |

### Per-Asset Breakdown

| Asset | Trades | Win Rate | Profit Factor | Total PnL |
|-------|--------|----------|---------------|-----------|
| GBPUSD | 521 | 64.7% | 3.66 | $17.3M |
| USDCHF | 506 | 66.8% | 3.61 | $16.5M |
| USDJPY | 455 | 64.2% | 2.79 | $10.2M |
| EURUSD | 452 | 67.7% | 2.55 | $9.7M |

---

## Architecture

### Signal Chain

```
┌─────────────────────────────────────────────────────────────────┐
│                     cTrader Open API                            │
│                  (Protobuf over WebSocket)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Data Acquisition Layer                         │
│    • Real-time M5 OHLCV for 4 assets (parallel fetch)          │
│    • Account state: balance, equity, margin, open positions     │
│    • Macro data: FRED API, COT reports, yfinance               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                       │
│    • 31 Market States (Alpha Features)                         │
│    • Kalman Filter denoising                                   │
│    • Cross-pair flow analysis                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Alpha Model Inference                              │
│                                                                 │
│    ┌──────────────────────────────────────────────┐             │
│    │              ALPHA (LSTM + Attention)        │             │
│    │                                              │             │
│    │  Input: 31 features × 25 timesteps          │             │
│    │  Output: Buy Probability (0-1)               │             │
│    │  Threshold: 0.55 (configurable)              │             │
│    └──────────────────────────────────────────────┘             │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Execution Layer                                │
│    • Fixed Risk: SL=2.0x ATR, TP=4.0x ATR                     │
│    • Position Size: 10% of equity                              │
│    • Asset lock enforcement (1 position per asset max)         │
│    • Market order submission via cTrader Open API              │
│    • Discord notifications (PnL Milestones & Pulse Checks)      │
└─────────────────────────────────────────────────────────────────┘
```

### Alpha Model Architecture (V7)

```
Input (25, 31) → Variable Selection Network (Feature Gating)
    → LSTM (128 units, bidirectional, 3 layers)
    → Layer Normalization
    → Learnable Positional Encoding
    → Scaled Dot-Product Multi-Head Attention (4 heads)
    → Gated Residual Network Blocks
    → Asset Embedding (4 assets)
    → Action Head: Dense(1) → Sigmoid [Buy Probability]
```

**Key Features:**
- **Variable Selection Network**: Learns which features matter most per timestep
- **Multi-Head Attention**: Captures long-range dependencies across sequence
- **Gated Residual Networks**: Stable gradient flow with gating mechanisms
- **Asset Embedding**: Shared representation across 4 forex pairs
- **Binary Classification**: Buy vs Hold (sell is inverse of buy)

---

## Project Structure

```
tradingbot/
├── Alpha/                          # Signal Generation Model (LSTM+Attention)
│   ├── src/                        # Model architecture & feature engineering
│   │   ├── model.py               # AlphaSLModel & AlphaSLModelV7
│   │   ├── feature_engine.py      # 31-feature pipeline with Kalman filter
│   │   ├── labeling.py            # Tradeability labels with net-R simulation
│   │   ├── calibration.py         # Temperature scaling for probability calibration
│   │   └── data_loader.py         # Data loading & preprocessing
│   ├── models/                     # Trained model checkpoints
│   │   ├── alpha_model.pth        # V7 model weights
│   │   └── alpha_calibration.json # Calibration parameters
│   ├── run_pipeline.py            # Training pipeline with purged CV
│   └── calibrate.py               # Standalone calibration script
│
├── Filter/                         # RF Ensemble Filter (Optional Gate)
│   ├── src/                        # Feature engine for filter
│   ├── models/                     # Trained RF ensemble
│   └── train_rf.py                # Filter training script
│
├── backtest/                       # Backtesting Framework
│   ├── alpha_lstm_backtest.py      # Vectorized backtest engine
│   ├── data/                       # Historical data (Parquet)
│   └── results/                    # Backtest metrics & charts
│
├── LiveExecution/                  # Production Execution Engine
│   ├── src/                        # Core components
│   │   ├── orchestrator.py        # Twisted-based async orchestrator
│   │   ├── models.py              # Model loader (Alpha + Filter)
│   │   ├── features.py            # Real-time feature manager
│   │   ├── ctrader_client.py      # cTrader WebSocket client
│   │   ├── config.py              # Configuration management
│   │   └── database.py            # SQLite trade logging
│   ├── dashboard/                  # Flask-based monitoring
│   └── main.py                    # Entry point
│
├── scripts/                        # Utility Scripts
│   ├── smoke_test.py              # Component instantiation test
│   ├── validate_features.py       # Feature validation (no lookahead)
│   └── download_*.py              # Data download scripts
│
├── data/                           # Shared Data Storage
├── models/                         # Shared Model Storage
├── shared_constants.py             # Asset definitions & spread config
├── main.py                         # Production entry point
└── requirements.txt                # Python dependencies
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
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

# Set up environment variables
cp .env.example .env
# Edit .env with your cTrader credentials
```

### Environment Variables

```env
CT_APP_ID=your_app_id
CT_APP_SECRET=your_app_secret
CT_ACCOUNT_ID=your_account_id
CT_ACCESS_TOKEN=your_access_token
CT_HOST_TYPE=demo  # or 'live' for real trading
```

---

## Usage

### 1. Training the Alpha Model

```bash
# Quick sanity check (smoke test)
cd Alpha && python run_pipeline.py --smoke-test

# Full training pipeline
cd Alpha && python run_pipeline.py

# Calibrate trained model
python Alpha/calibrate.py
```

### 2. Backtesting

```bash
# Run vectorized backtest
python backtest/alpha_lstm_backtest.py --initial-equity 10000

# With custom parameters
python backtest/alpha_lstm_backtest.py \
  --confidence-thresh 0.55 \
  --sl-mult 2.0 \
  --tp-mult 4.0 \
  --max-hold-bars 18
```

### 3. Live Execution

```bash
# Start live trading (requires .env configuration)
python main.py

# Or from LiveExecution directory
cd LiveExecution && python main.py
```

### 4. Validation Scripts

```bash
# Smoke test - verify all components instantiate
python scripts/smoke_test.py

# Feature validation - check for data leakage
python scripts/validate_features.py
```

---

## Model Details

### Alpha Model (V7)

| Component | Specification |
|-----------|---------------|
| **Architecture** | LSTM + Multi-Head Attention + GRN |
| **Input** | 31 features × 25 timesteps |
| **LSTM Units** | 128 (bidirectional) |
| **Attention Heads** | 4 |
| **Output** | Binary (Buy probability) |
| **Sequence Length** | 25 bars (125 minutes) |
| **Training** | Purged time-series cross-validation |
| **Loss** | BCE with label smoothing (0.01) |

### Feature Set (31 Features)

| Category | Features | Count |
|----------|----------|-------|
| **Core Regime/Momentum** | volatility, atr_norm, hour_sin/cos, regime, return_12_atr, ema_slope_atr, breakout_position, momentum_6, bar_strength, intraday_position, vol_percentile, activity_ratio, trend_momentum, breakout_conviction | 15 |
| **Cross-Pair Flow** | usd_index_return, pair_residual, cross_pair_divergence | 3 |
| **Session Structure** | session_open_dist, asian_range_pos | 2 |
| **Momentum Exhaustion** | consec_dir_bars, wick_ratio, atr_contraction, return_decel | 4 |
| **Volume/Order Flow** | volume_spike, volume_accumulation, volume_climax, volume_session_rel | 4 |
| **Macro Context** | dxy_return_5d, sp500_return_5d | 3 |

### Risk Management

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Stop Loss** | 2.0x ATR | Dynamic based on volatility |
| **Take Profit** | 4.0x ATR | 2:1 reward-to-risk ratio |
| **Max Hold** | 18 bars (90 min) | Time-based exit |
| **Position Size** | 10% of equity | Fixed risk per trade |
| **Max Positions** | 5 total, 1 per asset | Diversification limit |
| **Leverage** | 1:100 | Standard forex leverage |

---

## Key Components

### Feature Engineering

- **Kalman Filter**: Adaptive denoising for regime detection
- **Fractional Differentiation**: Stationarity-preserving feature transformation
- **Multi-Asset Features**: Cross-pair correlations and flow analysis
- **Macro Integration**: FRED API, COT reports, S&P 500, DXY

### Training Pipeline

- **Purged Time-Series Split**: Prevents data leakage across splits
- **Embargo Gap**: Ensures no label horizon crosses split boundaries
- **Gradient Accumulation**: Effective batch size of 512 (128 × 4)
- **Mixed Precision**: bfloat16 on Ampere+ GPUs
- **Class Balancing**: Weighted loss for imbalanced trade labels

### Backtesting Engine

- **Vectorized Processing**: Batch prediction for speed
- **Barrier Exits**: SL/TP hit detection within candles
- **Spread Costs**: Realistic transaction cost modeling
- **Multi-Asset**: Parallel backtesting across 4 pairs

---

## Configuration

### Thresholds (Loaded from `backtest/results/optimal_thresholds.json`)

```json
{
  "alpha_confidence_threshold": 0.55,
  "meta_threshold": 0.7071,
  "qual_threshold": 0.7,
  "risk_threshold": 0.1
}
```

### Default Parameters

| Parameter | Default | Location |
|-----------|---------|----------|
| Sequence Length | 25 | Alpha/model.py |
| LSTM Units | 128 | Alpha/model.py |
| Dropout | 0.25 | Alpha/run_pipeline.py |
| Learning Rate | 1e-4 | Alpha/run_pipeline.py |
| Batch Size | 128 | Alpha/run_pipeline.py |

---

## Monitoring & Notifications

### Discord Integration

- **PnL Milestones**: Real-time notifications for every 1% movement
- **Pulse Checks**: 2-hour recurring health checks
- **Trade Alerts**: Opening/closing of positions

### Dashboard

- **Flask-based UI**: Real-time equity curve and position monitoring
- **WebSocket Updates**: Live trade execution events
- **Database Logging**: SQLite for trade history and account state

---

## Development

### Running Tests

```bash
# Smoke test
python scripts/smoke_test.py

# Feature validation
python scripts/validate_features.py

# Learnability check
python scripts/learnability_check.py
```

### Code Structure

- **Alpha/src/model.py**: Neural network architectures (V6, V7)
- **Alpha/src/feature_engine.py**: 31-feature calculation pipeline
- **Alpha/run_pipeline.py**: Training loop with purged CV
- **LiveExecution/src/orchestrator.py**: Production execution logic
- **backtest/alpha_lstm_backtest.py**: Vectorized backtest engine

---

## License

This project is proprietary software. All rights reserved.

---

<p align="center">
  <strong>Built with PyTorch | LSTM + Attention | cTrader Open API</strong>
</p>

<p align="center">
  <em>Version 3.0.0 | August 2026</em>
</p>

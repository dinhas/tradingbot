# RL Trading Bot

A reinforcement learning-based trading bot using PPO algorithm to trade 5 assets (EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD) on 5-minute timeframes.

## Overview

This project implements a sophisticated trading agent that learns optimal trading strategies through curriculum learning across 3 progressive stages:

- **Stage 1**: Direction Only (Entry/Exit timing)
- **Stage 2**: Direction + Position Sizing
- **Stage 3**: Full Control (Direction + Sizing + SL/TP optimization)

## Project Structure

```
tradingbot/
├── data/                       # Historical OHLCV data (5-minute candles)
│   ├── EURUSD_5m.parquet
│   ├── GBPUSD_5m.parquet
│   ├── USDJPY_5m.parquet
│   ├── USDCHF_5m.parquet
│   └── XAUUSD_5m.parquet
├── src/
│   ├── data_fetcher.py        # cTrader API data fetcher
│   ├── feature_engine.py      # Calculate 140 features
│   ├── trading_env.py         # Gym environment
│   ├── train.py               # PPO training script
│   └── test_data.py           # Data preprocessing validation
├── models/
│   └── checkpoints/           # Model checkpoints
├── logs/
│   └── tensorboard/           # TensorBoard logs
└── PRD.md                     # Product Requirements Document
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training Stage 1 (Direction Only)

```bash
python -m src.train --stage 1 --total_timesteps 1000000
```

### Training Stage 2 (Direction + Sizing)

```bash
python -m src.train --stage 2 --total_timesteps 1500000 --load_model models/checkpoints/stage_1_final.zip
```

### Training Stage 3 (Full Control)

```bash
python -m src.train --stage 3 --total_timesteps 1500000 --load_model models/checkpoints/stage_2_final.zip
```

### Quick Validation (Dry Run)

```bash
python -m src.train --dry-run
```

## Training Parameters

- **Learning Rate**: 3e-4
- **Batch Size**: 64
- **N-Steps**: 2048
- **Parallel Environments**: 8 (SubprocVecEnv)
- **Checkpoint Frequency**: Every 100k steps

## Key Features

- **Curriculum Learning**: Progressive skill development across 3 stages
- **Dynamic Portfolio**: Randomized starting capital ($100, $1k, $10k) and leverage (1:100, 1:200, 1:500)
- **Risk Management**: Hard limits on position size (50%), total exposure (60%), and drawdown (25%)
- **Rich Observation Space**: 140 features including technical indicators, cross-asset correlations, and session data
- **Intelligent Reward Function**: Balances realized/unrealized P&L, risk-reward quality, and drawdown penalties

## Performance Expectations

- **Stage 1 Training**: ~20 hours (1M steps on Kaggle)
- **Stage 2 Training**: ~30 hours (1.5M steps)
- **Stage 3 Training**: ~30 hours (1.5M steps)
- **Data Preprocessing**: ~3-4 minutes (one-time per session)

## Success Metrics

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| **Profit Factor** | > 1.5 | > 1.3 |
| **Max Drawdown** | < 15% | < 20% |
| **Sharpe Ratio** | > 1.5 | > 1.0 |
| **Win Rate** | > 50% | > 45% |

## Documentation

- **PRD.md**: Complete product requirements and technical specifications
- **Implementation Plan**: Detailed architecture and design decisions
- **Walkthrough**: Step-by-step implementation guide

## License

This project is for educational and research purposes.

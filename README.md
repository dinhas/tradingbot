# RL Trading Bot

A reinforcement learning-based trading bot that uses a PPO algorithm to trade five major assets (EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD) on a 5-minute timeframe.

## Overview

This project implements a sophisticated trading agent that learns optimal trading strategies through curriculum learning across three progressive stages:

- **Stage 1**: Direction Only (Entry/Exit timing)
- **Stage 2**: Direction + Position Sizing
- **Stage 3**: Full Control (Direction + Sizing + SL/TP optimization)

## Key Features

- **Curriculum Learning**: Progressive skill development across 3 stages.
- **Dynamic Portfolio**: Randomized starting capital ($100, $1k, $10k) and leverage (1:100, 1:200, 1:500).
- **Risk Management**: Hard limits on position size (50%), total exposure (60%), and drawdown (25%).
- **Rich Observation Space**: 140 features including technical indicators, cross-asset correlations, and session data.
- **Intelligent Reward Function**: Balances realized/unrealized P&L, risk-reward quality, and drawdown penalties.

## Project Structure

```
.
├── backtest/
├── config/
├── src/
├── tests/
├── util/
├── .gitignore
├── README.md
├── requirements.txt
└── validate_fixes.py
```

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/rl-trading-bot.git
    cd rl-trading-bot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Train the Model

The model is trained in three stages, each building on the previous one.

### Stage 1: Direction Only

In this stage, the agent learns the basics of market timing, focusing solely on entry and exit signals.

**Command:**
```bash
python -m src.train --stage 1 --total_timesteps 1000000
```
- **Description:** Trains the model from scratch, focusing only on market direction.
- **Expected Outcome:** A foundational model saved in `models/checkpoints/`.

### Stage 2: Direction + Sizing

Building on the first stage, the agent now learns to size its positions, adding a new layer of complexity to its strategy.

**Command:**
```bash
python -m src.train --stage 2 --total_timesteps 1500000 --load_model models/checkpoints/stage_1_final.zip
```
- **Description:** Loads the Stage 1 model and continues training with position sizing.
- **Expected Outcome:** An enhanced model with improved risk management capabilities.

### Stage 3: Full Control

In the final stage, the agent gains full control over its trades, including optimizing stop-loss and take-profit levels.

**Command:**
```bash
python -m src.train --stage 3 --total_timesteps 1500000 --load_model models/checkpoints/stage_2_final.zip
```
- **Description:** Loads the Stage 2 model and trains it with full control over trading parameters.
- **Expected Outcome:** A fully trained model capable of making complex trading decisions.

### Quick Validation (Dry Run)

To ensure the environment is set up correctly, run a quick validation test.

```bash
python -m src.train --dry-run
```

## License

This project is for educational and research purposes.

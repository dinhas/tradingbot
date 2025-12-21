# Track Spec: TradeGuard Dataset Generation

## Overview
This track focuses on the creation of a high-quality training dataset for the TradeGuard LightGBM model. It involves running the trained Alpha (PPO) model across 8 years of historical data to identify trade signals, calculating 60 predictive features at each signal point, and labeling the outcome (Win/Loss) based on future price action.

## Core Components
1. **Data Loader:** Efficiently load and align 5-minute OHLCV Parquet files for EURUSD, GBPUSD, USDJPY, USDCHF, and XAUUSD.
2. **Alpha Inference Loop:** A script to step through historical data and trigger signals based on the PPO agent's actions.
3. **Feature Engine:** A modular system to calculate 6 categories of features (60 total) as defined in the PRD.
4. **Labeling Logic:** A lookahead simulation to determine if a signal would have hit TP or SL first.
5. **Dataset Exporter:** Save the final structured dataset to Parquet format.

## Features to Implement
- **Group A:** Alpha Model Confidence (Internal state)
- **Group B:** Synthetic News Proxies (Volatility/Volume anomalies)
- **Group C:** Market Regime (Hurst, ADX, Trend)
- **Group D:** Session Edge (Cyclical time, Liquidity windows)
- **Group E:** Execution Statistics (Entry quality, Risk/Reward)
- **Group F:** Price Action Context (Candle structure, Support/Resistance)

## Technical Constraints
- Use `stable-baselines3` for Alpha model inference.
- Use `ta` library for technical indicators.
- Ensure zero feature leakage (no lookahead in features).
- Output must be a single Parquet file optimized for LightGBM training.

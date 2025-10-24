# src/config.py
"""Configuration parameters for the trading system and genetic algorithm."""

# Trading Parameters
INITIAL_CAPITAL = 10000  # Starting capital in USD
POSITION_SIZE_PCT = 0.25  # Use 25% of capital per trade
SPREAD_PIPS = 2  # EUR/USD typical spread (2 pips)
PIP_VALUE = 0.0001  # For 4-decimal pairs like EUR/USD

# Backtest Parameters
MIN_TRADES_REQUIRED = 30  # Minimum trades for valid strategy
MAX_DRAWDOWN_THRESHOLD = 0.30  # 30% max drawdown limit

# Data Split
TRAIN_TEST_SPLIT = 0.70  # 70% training, 30% testing

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
GENERATIONS = 100
CROSSOVER_RATE = 0.70
MUTATION_RATE = 0.15
ELITISM_COUNT = 5
TOURNAMENT_SIZE = 3

# Parameter Ranges for Optimization
# Method 1: Donchian Channel Breakout
DONCHIAN_PERIOD_MIN = 10
DONCHIAN_PERIOD_MAX = 100

# Method 2: ATR Volatility Breakout
ATR_PERIOD_MIN = 10
ATR_PERIOD_MAX = 50
ATR_MULTIPLIER_MIN = 0.5
ATR_MULTIPLIER_MAX = 3.0

# Method 3: Volume-Confirmed Channel Breakout
VOLUME_LOOKBACK_MIN = 10
VOLUME_LOOKBACK_MAX = 100
VOLUME_THRESHOLD_MIN = 1.0
VOLUME_THRESHOLD_MAX = 3.0
VOLUME_MA_PERIOD_MIN = 10
VOLUME_MA_PERIOD_MAX = 50

import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class RiskConfig:
    # Data Settings
    DATA_PATH: str = "data/"
    ASSETS: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD'])
    TIMEFRAME: str = "5m"
    START_YEAR: int = 2016
    END_YEAR: int = 2024

    # Environment Settings
    INITIAL_EQUITY: float = 10000.0
    TERMINATION_THRESHOLD: float = 0.02  # 2% of initial equity (98% drawdown)
    RANDOM_START: bool = True
    WARMUP_PERIOD: int = 100

    # Action Space Bounds
    SL_MULT_MIN: float = 0.5
    SL_MULT_MAX: float = 3.0
    RR_RATIO_MIN: float = 1.0
    RR_RATIO_MAX: float = 4.0
    RISK_PCT_MIN: float = 0.001
    RISK_PCT_MAX: float = 0.02

    # Execution Costs
    SPREAD_PITS: float = 1.5  # Typical spread in pits/points
    COMMISSION_PER_LOT: float = 3.5  # Per side
    SLIPPAGE_STD: float = 0.2  # ATR-scaled slippage

    # Peak/Valley Labeling
    REVERSAL_ATR_MULT: float = 2.0
    LOOKAHEAD_WINDOW: int = 500

    # Entry Filter (Simulated)
    META_THRESHOLD: float = 0.55
    QUALITY_THRESHOLD: float = 0.30

    # SAC Hyperparameters
    GAMMA: float = 0.99
    TAU: float = 0.005
    ALPHA: float = 0.2
    LR: float = 3e-4
    BATCH_SIZE: int = 256
    HIDDEN_DIM: int = 256
    REPLAY_SIZE: int = 1000000

    # Rewards
    K1_PEAK: float = 1.0
    K2_VALLEY: float = 1.0

    # System
    SEED: int = 42
    DEVICE: str = "cpu"  # "cuda" if available

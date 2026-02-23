import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class RiskConfig:
    # Data Path
    DATA_PATH: str = "data/"
    ASSETS: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD"])

    # Trading Parameters
    INITIAL_EQUITY: float = 100_000.0
    TERMINATION_DRAWDOWN: float = 0.98  # 98% drawdown

    # Execution Costs
    SPREAD_Pips: float = 1.5  # Default spread in pips (will be adjusted per asset if needed)
    COMMISSION_PER_LOT: float = 7.0  # Round turn commission per 100k volume
    SLIPPAGE_ATR_RATIO: float = 0.1  # Slippage as percentage of ATR

    # Action Space Bounds
    SL_MULT_MIN: float = 0.5
    SL_MULT_MAX: float = 3.0
    RR_RATIO_MIN: float = 1.0
    RR_RATIO_MAX: float = 4.0
    RISK_PCT_MIN: float = 0.001
    RISK_PCT_MAX: float = 0.02

    # Labeling
    REVERSAL_ATR_MULT: float = 2.0  # ATR multiplier for structural reversal

    # Reward Weights
    K1_PEAK: float = 1.0
    K2_VALLEY: float = 1.0
    DD_PENALTY_MULT: float = 5.0
    TERMINATION_PENALTY: float = -100.0

    # RL Hyperparameters (SAC)
    BATCH_SIZE: int = 256
    BUFFER_SIZE: int = 1_000_000
    GAMMA: float = 0.99
    TAU: float = 0.005
    LEARNING_RATE: float = 3e-4
    ALPHA: float = 0.2  # Entropy coefficient
    HIDDEN_DIM: int = 256

    # Entry Filter (RL doesn't see these, but Env uses them to trigger trades)
    META_THRESHOLD: float = 0.55
    QUALITY_THRESHOLD: float = 0.30

    # Environment
    WARMUP_PERIOD: int = 100  # Number of bars to calculate indicators

    # Paths
    MODEL_SAVE_PATH: str = "models/risklayer/"

# Create singleton instance
config = RiskConfig()

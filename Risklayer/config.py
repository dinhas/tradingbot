import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # Data Paths
    DATA_DIR: str = "data"
    ASSETS: List[str] = field(
        default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD"]
    )

    # Execution Parameters
    INITIAL_EQUITY: float = 10000.0
    SPREADS: dict = field(
        default_factory=lambda: {
            "EURUSD": 0.00012,  # 1.2 pips
            "GBPUSD": 0.00015,
            "USDJPY": 0.012,
            "USDCHF": 0.00015,
            "XAUUSD": 0.25,
        }
    )
    COMMISSION_PER_LOT: float = 0.0  # Standard account - commission included in spread
    SLIPPAGE_STD: float = 0.1  # Multiplier for slippage based on ATR

    # Contract Sizes
    CONTRACT_SIZES: dict = field(
        default_factory=lambda: {
            "EURUSD": 100000,
            "GBPUSD": 100000,
            "USDJPY": 100000,
            "USDCHF": 100000,
            "XAUUSD": 100,
        }
    )

    # Labeling Parameters
    ATR_REVERSAL_MULTIPLIER: float = 3.0  # X * ATR for structural reversal

    # RL Hyperparameters
    STATE_DIM: int = (
        36  # 30 features + ATR + Vol % + Equity % + DD % + Margin + PosState
    )
    ACTION_DIM: int = 3
    HIDDEN_DIM: int = 512
    LR: float = 1e-4
    GAMMA: float = 0.995
    TAU: float = 0.001
    ALPHA: float = 0.2  # Entropy coefficient
    BATCH_SIZE: int = 1024
    BUFFER_SIZE: int = 1_000_000

    # Action Ranges
    SL_MULTIPLIER_MIN: float = 0.5
    SL_MULTIPLIER_MAX: float = 3.0
    RR_RATIO_MIN: float = 1.0
    RR_RATIO_MAX: float = 4.0
    RISK_PERCENT_MIN: float = 0.001
    RISK_PERCENT_MAX: float = 0.02

    # Reward Coefficients
    K_STRUCTURAL_PEAK: float = 0.2
    K_STRUCTURAL_VALLEY: float = 0.2
    K_DRAWDOWN_PENALTY: float = 5.0
    K_MARGIN_PENALTY: float = 2.0
    TERMINATION_PENALTY: float = -100.0

    # Entry Filter Thresholds
    META_SCORE_THRESHOLD: float = 0.55
    QUALITY_SCORE_THRESHOLD: float = 0.30

    # Training Parameters
    TOTAL_STEPS: int = 2_000_000
    SEED: int = 42


config = Config()

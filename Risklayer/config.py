import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class RiskConfig:
    # Data parameters
    ASSET: str = "EURUSD"
    TIMEFRAME: str = "5m"
    DATA_PATH: str = "data/EURUSD_5m.parquet"

    # Entry Filter (Hardcoded as per requirements)
    META_THRESHOLD: float = 0.55
    QUALITY_THRESHOLD: float = 0.30

    # Action Space Bounds
    SL_MULT_MIN: float = 0.5
    SL_MULT_MAX: float = 3.0
    RR_RATIO_MIN: float = 1.0
    RR_RATIO_MAX: float = 4.0
    RISK_PCT_MIN: float = 0.001
    RISK_PCT_MAX: float = 0.02

    # Execution Parameters
    SPREAD_PIPS: float = 1.2
    COMMISSION_PER_LOT: float = 7.0  # USD per 100k
    SLIPPAGE_STD: float = 0.2        # Multiplied by ATR for slippage simulation
    INITIAL_EQUITY: float = 100000.0
    DRAWDOWN_LIMIT_PCT: float = 0.98 # Terminate at 98% drawdown (2% equity)

    # Structural Labeling
    REVERSAL_ATR_MULT: float = 2.0   # ATR multiplier for peak/valley reversal

    # Reward Weights
    K1_STRUCTURAL_PROFIT: float = 1.0
    K2_STRUCTURAL_LOSS: float = 1.0
    DRAWDOWN_PENALTY: float = 5.0
    MARGIN_PENALTY: float = 0.1
    TERMINATION_PENALTY: float = 50.0

    # SAC Agent Parameters
    HIDDEN_DIMS: List[int] = field(default_factory=lambda: [256, 256])
    LR: float = 3e-4
    GAMMA: float = 0.99
    TAU: float = 0.005
    ALPHA: float = 0.2               # Entropy coefficient
    BATCH_SIZE: int = 256
    BUFFER_SIZE: int = 100000

    # Training Parameters
    TOTAL_STEPS: int = 10000
    WARMUP_STEPS: int = 1000
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42

config = RiskConfig()

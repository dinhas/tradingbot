import numpy as np
from .config import RiskConfig
from .execution_engine import TradeResult

class RewardEngine:
    def __init__(self, config: RiskConfig):
        self.config = config

    def calculate_structural_reward(self,
                                    peak_dist: float,
                                    valley_dist: float,
                                    tp_dist: float,
                                    sl_dist: float) -> float:
        """
        Calculates oracle-based structural reward at trade entry.
        """
        # Encourage trades where future peaks are far and valleys are shallow
        # Relative to our selected TP and SL
        r_structural = (self.config.K1_PEAK * (peak_dist / (tp_dist + 1e-9)) -
                        self.config.K2_VALLEY * (valley_dist / (sl_dist + 1e-9)))

        # Clip to avoid extreme rewards
        return np.clip(r_structural, -5.0, 5.0)

    def calculate_trade_reward(self,
                               result: TradeResult,
                               dd_increment: float,
                               margin_usage: float) -> float:
        """
        Calculates reward based on actual trade outcome.
        """
        # Scale net PnL to a reasonable reward range (e.g., relative to initial equity or risk)
        # We'll use a simple scaling for now
        r_trade = result.net_pnl / (self.config.INITIAL_EQUITY * 0.01)

        # Penalties
        r_trade -= dd_increment * 2.0 # Penalty for increasing drawdown
        r_trade -= margin_usage * 0.1 # Penalty for high margin usage

        return r_trade

    def get_termination_penalty(self) -> float:
        return -100.0 # Large penalty for hitting 98% drawdown

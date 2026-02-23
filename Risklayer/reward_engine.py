from .config import config

class RewardEngine:
    def __init__(self):
        self.k1 = config.K1_PEAK
        self.k2 = config.K2_VALLEY
        self.dd_penalty_mult = config.DD_PENALTY_MULT

    def calculate_structural_reward(self,
                                    peak_dist: float,
                                    valley_dist: float,
                                    sl_dist: float,
                                    tp_dist: float) -> float:
        reward = self.k1 * (peak_dist / (tp_dist + 1e-9)) - self.k2 * (valley_dist / (sl_dist + 1e-9))
        return float(reward)

    def calculate_trade_reward(self,
                               net_profit: float,
                               drawdown_increment: float,
                               initial_equity: float) -> float:
        # Normalize net profit by initial equity to keep reward scale stable
        norm_profit = net_profit / initial_equity * 100.0 # profit in % of initial equity

        reward = norm_profit
        reward -= self.dd_penalty_mult * drawdown_increment

        return float(reward)

    def get_termination_penalty(self) -> float:
        return config.TERMINATION_PENALTY

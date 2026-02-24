from Risklayer.config import config

class RewardEngine:
    def __init__(self):
        self.k_peak = config.K_STRUCTURAL_PEAK
        self.k_valley = config.K_STRUCTURAL_VALLEY
        self.k_dd = config.K_DRAWDOWN_PENALTY
        self.k_margin = config.K_MARGIN_PENALTY

    def calculate_structural_reward(
        self,
        peak_distance: float,
        valley_distance: float,
        tp_distance: float,
        sl_distance: float
    ) -> float:
        """Reward based on future potential (labels)."""
        r_peak = self.k_peak * (peak_distance / (tp_distance + 1e-6))
        r_valley = self.k_valley * (valley_distance / (sl_distance + 1e-6))
        return r_peak - r_valley

    def calculate_trade_close_reward(
        self,
        net_profit: float,
        initial_equity: float,
        drawdown_pct: float,
        margin_usage: float
    ) -> float:
        """Reward based on realized outcome and risk metrics."""
        # Normalize net profit relative to equity
        r_profit = net_profit / initial_equity

        # Penalties
        r_dd = -self.k_dd * (drawdown_pct ** 2)
        r_margin = -self.k_margin * margin_usage

        return r_profit + r_dd + r_margin

    def get_termination_penalty(self) -> float:
        return config.TERMINATION_PENALTY

from Risklayer.config import config
from Risklayer.execution_engine import TradeResult

class RewardEngine:
    def __init__(self):
        self.k1 = config.K1_STRUCTURAL_PROFIT
        self.k2 = config.K2_STRUCTURAL_LOSS
        self.drawdown_penalty_weight = config.DRAWDOWN_PENALTY
        self.margin_penalty_weight = config.MARGIN_PENALTY

    def calculate_reward(self,
                         trade_result: TradeResult,
                         peak_distance: float,
                         valley_distance: float,
                         sl_distance: float,
                         tp_distance: float,
                         drawdown_increment: float,
                         equity_used_pct: float) -> float:
        """
        Calculates the total reward for a trade.
        """
        # 1. Structural Reward
        # Reward for how close the chosen TP was to the actual peak
        # and how much buffer we had from the valley.
        # peak_distance/tp_distance: if peak was 100 and TP was 50, ratio is 2.0 (good).
        # If peak was 25 and TP was 50, ratio is 0.5 (bad, we overshot).
        # Actually, if we overshot, the trade would hit SL or force close.
        # valley_distance/sl_distance: if valley was 20 and SL was 50, ratio is 0.4 (good).
        # If valley was 60 and SL was 50, ratio is 1.2 (bad, we hit SL).

        r_structural = (self.k1 * (peak_distance / (tp_distance + 1e-6))) - \
                       (self.k2 * (valley_distance / (sl_distance + 1e-6)))

        # 2. Trade Reward
        # As per prompt: R_trade = + net_profit - spread_cost - commission - slippage_cost ...
        # Note: net_profit in TradeResult already includes commission and spread (via entry/exit prices)
        # We subtract them again as a penalty to encourage efficient trading.

        r_trade = trade_result.pnl_net \
                  - trade_result.spread_cost \
                  - trade_result.commission \
                  - (trade_result.slippage * (trade_result.pnl_raw / (trade_result.exit_price - trade_result.entry_price + 1e-6) if trade_result.exit_price != trade_result.entry_price else 0)) \
                  - (self.drawdown_penalty_weight * drawdown_increment) \
                  - (self.margin_penalty_weight * equity_used_pct)

        # Normalize r_trade? net_profit can be large.
        # Usually rewards should be in a reasonable range for SAC.
        # We'll apply a simple scaling if needed, but for now let's keep it raw.
        # Actually, let's scale by initial equity to keep it around 1.0
        r_trade_scaled = r_trade / (config.INITIAL_EQUITY * 0.01) # reward per 1% of equity

        return r_structural + r_trade_scaled

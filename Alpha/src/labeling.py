import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from shared_constants import DEFAULT_SPREADS
from Alpha.src.trade_simulator import TradeConfig, TradeSimulator

class Labeler:
    def __init__(self, tp_mult: float = 1.0, sl_mult: float = 0.5, ema_window: int = 100,
                 max_bars: int = 12, adx_threshold: float = 25.0, min_edge_r: float = 0.25):
        """
        Trend-Following Triple Barrier Labeler.
        Optimized for V3 Regime-Aware Denoising.
        Args:
            tp_mult: Take Profit multiplier for 5M ATR (Optimized: 1.0).
            sl_mult: Stop Loss multiplier for 5M ATR (Optimized: 0.5).
            ema_window: Window for the 1H Trend EMA (e.g., 100).
            max_bars: Vertical barrier (60m on 5M). Optimized: 12.
            adx_threshold: Minimum trend strength. Optimized: 25.0.
            min_edge_r: Minimum net R buffer required before a setup is labeled tradeable.
        """
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.ema_window = ema_window
        self.max_bars = max_bars
        self.adx_threshold = adx_threshold
        self.min_edge_r = min_edge_r
        self.simulator = TradeSimulator(TradeConfig(
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            max_hold_bars=max_bars,
        ))

    def _get_adx(self, df: pd.DataFrame, asset: str) -> pd.Series:
        """Calculates the 14-period ADX to measure trend strength."""
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        close = df[f"{asset}_close"]
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        return adx.adx()

    def _get_1h_trend(self, df: pd.DataFrame, asset: str) -> pd.Series:
        """Calculates the 1H Trend using 100 EMA on resampled 1H data.

        CAUSAL: the 1H series is shifted by one completed hour before being
        reindexed to 5M, so a 5M bar only ever sees the last fully closed
        1H candle (no intra-hour lookahead).
        """
        close_5m = df[f"{asset}_close"]

        # Resample to 1h to calculate the HTF Trend
        close_1h = close_5m.resample('1h').last().ffill()
        ema_1h = EMAIndicator(close_1h, window=self.ema_window).ema_indicator()

        # Determine Trend: 1 (Bullish), -1 (Bearish); NaN during EMA warmup
        trend_series_1h = pd.Series(np.where(close_1h > ema_1h, 1.0, -1.0), index=close_1h.index)
        trend_series_1h[ema_1h.isna()] = np.nan

        # Shift by one completed hour, then reindex back to 5M
        trend_series_1h = trend_series_1h.shift(1)
        return trend_series_1h.reindex(df.index, method='ffill')

    def label_data(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Labels data using a spread-aware triple-barrier simulation, then converts the
        outcome into a net-tradeability target.

        Output schema:
            - tradeable: 1 if the best long/short candidate clears min_edge_r.
            - direction: 1 for buy, 0 for sell. Meaningful when tradeable=1.
            - net_r: normalized net edge proxy for the best candidate.
            - valid: bars that pass the ADX/ATR/trend hygiene filters.
        """
        close_col = f"{asset}_close"
        open_col = f"{asset}_open"
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        atr_col = f"{asset}_atr"
        adx_col = f"{asset}_adx"

        if any(col not in df.columns for col in (open_col, close_col, high_col, low_col, atr_col, adx_col)):
            raise ValueError(f"Required columns missing for {asset}")
        
        # 1. Calculate the 1H Trend for all timestamps
        trend_1h = self._get_1h_trend(df, asset)
        
        # 2. Use pre-calculated ADX
        adx_series = df[adx_col].fillna(0)
        
        prices_open = df[open_col].values
        prices_close = df[close_col].values
        prices_high = df[high_col].values
        prices_low = df[low_col].values
        atrs_5m = df[atr_col].values
        timestamps = df.index
        
        spread = DEFAULT_SPREADS.get(asset, 0.0)

        n = len(df)
        # Full-length outputs: one row per input bar.
        # Invalid rows are KEPT (tradeable=0, valid=False) so downstream sequence
        # building stays contiguous in time.
        tradeable = np.zeros(n, dtype=np.float32)
        directions = np.zeros(n, dtype=np.float32)
        net_r = np.zeros(n, dtype=np.float32)
        long_net_r = np.zeros(n, dtype=np.float32)
        short_net_r = np.zeros(n, dtype=np.float32)
        long_tradeable = np.zeros(n, dtype=np.float32)
        short_tradeable = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        trend_vals = trend_1h.values
        adx_vals = adx_series.values
        timestamp_values = timestamps.to_numpy(dtype="datetime64[ns]")
        expected_delta = np.timedelta64(5, "m")

        for i in range(n - self.max_bars):
            current_adx = adx_vals[i]
            atr = atrs_5m[i]

            if np.isnan(atr) or atr == 0:
                continue
                
            # ADX Filter: Only train on candles that pass the ADX threshold
            if current_adx < self.adx_threshold:
                continue

            horizon_deltas = np.diff(timestamp_values[i:i + self.max_bars + 1])
            if len(horizon_deltas) != self.max_bars or np.any(horizon_deltas != expected_delta):
                continue

            valid[i] = True

            long_outcome = self.simulator.simulate(
                prices_open, prices_high, prices_low, prices_close, i, 1, atr, spread
            )
            short_outcome = self.simulator.simulate(
                prices_open, prices_high, prices_low, prices_close, i, -1, atr, spread
            )
            if long_outcome is None or short_outcome is None:
                valid[i] = False
                continue

            long_net_r[i] = long_outcome.net_r
            short_net_r[i] = short_outcome.net_r
            long_tradeable[i] = float(long_outcome.net_r > self.min_edge_r)
            short_tradeable[i] = float(short_outcome.net_r > self.min_edge_r)

            if long_net_r[i] >= short_net_r[i]:
                directions[i] = 1.0
                net_r[i] = long_net_r[i]
            else:
                directions[i] = 0.0
                net_r[i] = short_net_r[i]

            tradeable[i] = 1.0 if net_r[i] > self.min_edge_r else 0.0

        # 3-class action target: 0=hold, 1=short, 2=long
        action_class = np.zeros(n, dtype=np.float32)
        for i in range(n):
            s = short_tradeable[i]
            l = long_tradeable[i]
            if s > 0.5 and l > 0.5:
                action_class[i] = 2.0 if long_net_r[i] >= short_net_r[i] else 1.0
            elif s > 0.5:
                action_class[i] = 1.0
            elif l > 0.5:
                action_class[i] = 2.0

        return pd.DataFrame({
            'tradeable': tradeable,
            'direction': directions,
            'net_r': net_r,
            'long_tradeable': long_tradeable,
            'short_tradeable': short_tradeable,
            'long_net_r': long_net_r,
            'short_net_r': short_net_r,
            'action_class': action_class,
            'valid': valid,
        }, index=timestamps)

if __name__ == "__main__":
    pass

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from shared_constants import DEFAULT_SPREADS

class Labeler:
    def __init__(self, tp_mult: float = 1.0, sl_mult: float = 0.5, ema_window: int = 100,
                 max_bars: int = 6, adx_threshold: float = 25.0, min_edge_r: float = 0.10):
        """
        Trend-Following Triple Barrier Labeler.
        Optimized for V3 Regime-Aware Denoising.
        Args:
            tp_mult: Take Profit multiplier for 5M ATR (Optimized: 1.0).
            sl_mult: Stop Loss multiplier for 5M ATR (Optimized: 0.5).
            ema_window: Window for the 1H Trend EMA (e.g., 100).
            max_bars: Vertical barrier (30m on 5M). Optimized: 6.
            adx_threshold: Minimum trend strength. Optimized: 25.0.
            min_edge_r: Minimum net R buffer required before a setup is labeled tradeable.
        """
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.ema_window = ema_window
        self.max_bars = max_bars
        self.adx_threshold = adx_threshold
        self.min_edge_r = min_edge_r

    def _simulate_trade(self, prices_close: np.ndarray, prices_high: np.ndarray, prices_low: np.ndarray,
                        timestamps: pd.Index, asset: str, i: int, direction: int, atr: float,
                        spread: float) -> tuple[float, bool]:
        """Simulates a single long or short candidate and returns a normalized net-R estimate.

        direction: +1 for buy, -1 for sell.
        The target is not a raw barrier event; it is a net edge proxy after spread.
        """
        mid_price = prices_close[i]
        tp_dist = self.tp_mult * atr
        sl_dist = self.sl_mult * atr
        half_spread = spread / 2.0

        entry_price = mid_price + (direction * half_spread)
        sl = entry_price - (direction * sl_dist)
        tp = entry_price + (direction * tp_dist)

        exit_price = None
        for j in range(i + 1, min(i + 1 + self.max_bars, len(prices_close))):
            if direction == 1:
                bid_low = prices_low[j] - half_spread
                bid_high = prices_high[j] - half_spread
                if bid_low <= sl:
                    exit_price = sl
                    break
                if bid_high >= tp:
                    exit_price = tp
                    break
            else:
                ask_high = prices_high[j] + half_spread
                ask_low = prices_low[j] + half_spread
                if ask_high >= sl:
                    exit_price = sl
                    break
                if ask_low <= tp:
                    exit_price = tp
                    break

        if exit_price is None:
            final_idx = min(i + self.max_bars, len(prices_close) - 1)
            exit_price = prices_close[final_idx] - (direction * half_spread)

        # Normalize by initial risk so tradeability can be thresholded consistently.
        gross_r = (exit_price - entry_price) * direction / max(sl_dist, 1e-8)
        roundtrip_spread_r = spread / max(sl_dist, 1e-8)
        fee_r = (entry_price * 0.00004) / max(sl_dist, 1e-8)
        net_r = gross_r - roundtrip_spread_r - fee_r
        return float(net_r), bool(net_r > 0)

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
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        atr_col = f"{asset}_atr"
        adx_col = f"{asset}_adx"

        if close_col not in df.columns or atr_col not in df.columns or adx_col not in df.columns:
            raise ValueError(f"Required columns missing for {asset}")
        
        # 1. Calculate the 1H Trend for all timestamps
        trend_1h = self._get_1h_trend(df, asset)
        
        # 2. Use pre-calculated ADX
        adx_series = df[adx_col].fillna(0)
        
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
        valid = np.zeros(n, dtype=bool)

        trend_vals = trend_1h.values
        adx_vals = adx_series.values

        for i in range(n - 1):
            current_adx = adx_vals[i]
            atr = atrs_5m[i]

            if np.isnan(atr) or atr == 0:
                continue
                
            # ADX Filter: Only train on candles that pass the ADX threshold
            if current_adx < self.adx_threshold:
                continue

            valid[i] = True

            long_net_r, _ = self._simulate_trade(prices_close, prices_high, prices_low, timestamps, asset, i, 1, atr, spread)
            short_net_r, _ = self._simulate_trade(prices_close, prices_high, prices_low, timestamps, asset, i, -1, atr, spread)

            if long_net_r >= short_net_r:
                directions[i] = 1.0
                net_r[i] = long_net_r
            else:
                directions[i] = 0.0
                net_r[i] = short_net_r

            tradeable[i] = 1.0 if net_r[i] > self.min_edge_r else 0.0

        return pd.DataFrame({'tradeable': tradeable, 'direction': directions, 'net_r': net_r, 'valid': valid}, index=timestamps)

if __name__ == "__main__":
    pass

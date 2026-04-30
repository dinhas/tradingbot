import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from shared_constants import DEFAULT_SPREADS

class Labeler:
    def __init__(self, tp_mult: float = 1.0, sl_mult: float = 0.5, ema_window: int = 100, max_bars: int = 6, adx_threshold: float = 25.0, use_research_pipeline: bool = False):
        """
        Trend-Following Triple Barrier Labeler.
        Optimized for V3 Regime-Aware Denoising.
        Args:
            tp_mult: Take Profit multiplier for 5M ATR (Optimized: 1.0).
            sl_mult: Stop Loss multiplier for 5M ATR (Optimized: 0.5).
            ema_window: Window for the 1H Trend EMA (e.g., 100).
            max_bars: Vertical barrier (30m on 5M). Optimized: 6.
            adx_threshold: Minimum trend strength. Optimized: 25.0.
            use_research_pipeline: If True, uses the research-validated Triple Barrier parameters.
        """
        self.use_research_pipeline = use_research_pipeline
        if self.use_research_pipeline:
            # STEP 3: Research Parameters
            self.tp_mult = 1.5
            self.sl_mult = 1.5
            self.max_bars = 20
        else:
            self.tp_mult = tp_mult
            self.sl_mult = sl_mult
            self.max_bars = max_bars

        self.ema_window = ema_window
        self.adx_threshold = adx_threshold

    def _get_adx(self, df: pd.DataFrame, asset: str) -> pd.Series:
        """Calculates the 14-period ADX to measure trend strength."""
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        close = df[f"{asset}_close"]
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        return adx.adx()

    def _get_1h_trend(self, df: pd.DataFrame, asset: str) -> pd.Series:
        """Calculates the 1H Trend using 100 EMA on resampled 1H data."""
        close_5m = df[f"{asset}_close"]
        
        # Resample to 1h to calculate the HTF Trend
        close_1h = close_5m.resample('1h').last().ffill()
        ema_1h = EMAIndicator(close_1h, window=self.ema_window).ema_indicator()
        
        # Determine Trend: 1 (Bullish), -1 (Bearish)
        trend_1h = np.where(close_1h > ema_1h, 1, -1)
        trend_series_1h = pd.Series(trend_1h, index=close_1h.index)
        
        # Reindex back to 5M to align with original data
        return trend_series_1h.reindex(df.index, method='ffill')

    def label_data(self, df: pd.DataFrame, asset: str) -> pd.DataFrame:
        """
        Labels data using Triple Barrier Method.
        If use_research_pipeline is True, uses RANGING-regime research parameters.
        Otherwise uses legacy trend-following logic.
        """
        close_col = f"{asset}_close"
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"

        if self.use_research_pipeline:
            atr_col = f"{asset}_atr_kalman"
        else:
            atr_col = f"{asset}_atr"

        adx_col = f"{asset}_adx"

        if close_col not in df.columns or atr_col not in df.columns or adx_col not in df.columns:
            raise ValueError(f"Required columns missing for {asset} (expected {close_col}, {atr_col}, {adx_col})")
        
        prices_close = df[close_col].values
        prices_high = df[high_col].values
        prices_low = df[low_col].values
        atrs_5m = df[atr_col].values
        timestamps = df.index
        
        labels = []
        indices = []
        
        n = len(df)

        if self.use_research_pipeline:
            # STEP 3: Research Label Generator (Causal)
            for i in range(n - self.max_bars - 1):
                mid_price = prices_close[i]
                atr = atrs_5m[i]
                
                if np.isnan(atr) or atr == 0:
                    continue
                
                pt_barrier = mid_price + self.tp_mult * atr
                sl_barrier = mid_price - self.sl_mult * atr

                direction = 0 # default time expiry
                for j in range(1, self.max_bars + 1):
                    if prices_high[i+j] >= pt_barrier:
                        direction = 1
                        break
                    if prices_low[i+j] <= sl_barrier:
                        direction = -1
                        break

                labels.append({'direction': direction})
                indices.append(timestamps[i])
        else:
            # Legacy logic
            trend_1h = self._get_1h_trend(df, asset)
            adx_series = df[adx_col].fillna(0)
            spread = DEFAULT_SPREADS.get(asset, 0.0)
            
            for i in range(n - 1):
                current_trend = trend_1h.iloc[i]
                current_adx = adx_series.iloc[i]
                mid_price = prices_close[i]
                atr = atrs_5m[i]

                if np.isnan(atr) or atr == 0 or np.isnan(current_trend):
                    continue
                if current_adx < self.adx_threshold:
                    continue

                tp_dist = self.tp_mult * atr
                sl_dist = self.sl_mult * atr
                direction = 0

                if current_trend == 1:
                    entry_ask = mid_price + (spread / 2.0)
                    tp_barrier = entry_ask + tp_dist
                    sl_barrier = entry_ask - sl_dist
                    for j in range(i + 1, min(i + 1 + self.max_bars, n)):
                        bid_high = prices_high[j] - (spread / 2.0)
                        bid_low = prices_low[j] - (spread / 2.0)
                        if bid_high >= tp_barrier:
                            direction = 1; break
                        if bid_low <= sl_barrier:
                            direction = 0; break
                elif current_trend == -1:
                    entry_bid = mid_price - (spread / 2.0)
                    tp_barrier = entry_bid - tp_dist
                    sl_barrier = entry_bid + sl_dist
                    for j in range(i + 1, min(i + 1 + self.max_bars, n)):
                        ask_low = prices_low[j] + (spread / 2.0)
                        ask_high = prices_high[j] + (spread / 2.0)
                        if ask_low <= tp_barrier:
                            direction = -1; break
                        if ask_high >= sl_barrier:
                            direction = 0; break

                labels.append({'direction': direction})
                indices.append(timestamps[i])
            
        return pd.DataFrame(labels, index=indices)

if __name__ == "__main__":
    pass

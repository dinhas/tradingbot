import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator

class Labeler:
    def __init__(self, tp_mult: float = 4.0, sl_mult: float = 2.0, ema_window: int = 100, max_bars: int = 24, adx_threshold: float = 20.0):
        """
        Trend-Following Triple Barrier Labeler.
        Args:
            tp_mult: Take Profit multiplier for 5M ATR (e.g., 4.0).
            sl_mult: Stop Loss multiplier for 5M ATR (e.g., 2.0).
            ema_window: Window for the 1H Trend EMA (e.g., 100).
            max_bars: Vertical barrier (max holding period in bars). Default 24 (2h on 5M).
            adx_threshold: Minimum trend strength to process a label. Default 20.0.
        """
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.ema_window = ema_window
        self.max_bars = max_bars
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
        Labels data using Triple Barrier Method filtered by 1H Trend and a Vertical Barrier.
        """
        close_col = f"{asset}_close"
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        atr_col = f"{asset}_atr"

        if close_col not in df.columns or atr_col not in df.columns:
            raise ValueError(f"Required columns missing for {asset}")
        
        # 1. Calculate the 1H Trend for all timestamps
        trend_1h = self._get_1h_trend(df, asset)
        
        # 2. Calculate ADX for all timestamps
        adx_series = self._get_adx(df, asset).fillna(0)
        
        prices_close = df[close_col].values
        prices_high = df[high_col].values
        prices_low = df[low_col].values
        atrs_5m = df[atr_col].values
        timestamps = df.index
        
        labels = []
        indices = []
        
        n = len(df)
        for i in range(n - 1):
            current_trend = trend_1h.iloc[i]
            current_adx = adx_series.iloc[i]
            entry_price = prices_close[i]
            atr = atrs_5m[i]
            
            if np.isnan(atr) or atr == 0 or np.isnan(current_trend):
                continue
                
            # ADX Filter: Skip choppy markets
            if current_adx < self.adx_threshold:
                continue

            # Define barriers based on 5M ATR
            tp_dist = self.tp_mult * atr
            sl_dist = self.sl_mult * atr
            
            direction = 0 # Default: No barrier hit or Neutral (Vertical Barrier / Timeout)
            
            # --- TRIPLE BARRIER LOGIC ---
            if current_trend == 1: # Bullish Trend -> ONLY Look for BUYS
                upper_barrier = entry_price + tp_dist
                lower_barrier = entry_price - sl_dist
                
                # Look forward until a barrier is hit or max_bars reached
                for j in range(i + 1, min(i + 1 + self.max_bars, n)):
                    if prices_high[j] >= upper_barrier:
                        direction = 1 # TP Hit
                        break
                    if prices_low[j] <= lower_barrier:
                        direction = 0 # SL Hit
                        break
                        
            elif current_trend == -1: # Bearish Trend -> ONLY Look for SELLS
                upper_barrier = entry_price + sl_dist # Stop Loss for Short
                lower_barrier = entry_price - tp_dist # Take Profit for Short
                
                for j in range(i + 1, min(i + 1 + self.max_bars, n)):
                    if prices_low[j] <= lower_barrier:
                        direction = -1 # TP Hit
                        break
                    if prices_high[j] >= upper_barrier:
                        direction = 0 # SL Hit
                        break
            
            labels.append({'direction': direction})
            indices.append(timestamps[i])
            
        return pd.DataFrame(labels, index=indices)

if __name__ == "__main__":
    pass

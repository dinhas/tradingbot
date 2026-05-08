import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator
from shared_constants import DEFAULT_SPREADS

class Labeler:
    def __init__(self, tp_mult: float = 2.0, sl_mult: float = 1.0, ema_window: int = 100, max_bars: int = 7, adx_threshold: float = 25.0):
        """
        Modified Triple Barrier Labeler for 30M data.
        Args:
            tp_mult: Take Profit multiplier for ATR (User: 2.0).
            sl_mult: Stop Loss multiplier for ATR (User: 1.0).
            ema_window: Window for the HTF Trend EMA.
            max_bars: Vertical barrier (User: 7 candles).
            adx_threshold: Minimum trend strength.
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
        Labels data using Triple Barrier Method without regime or trend filters.
        Checks both Buy and Sell opportunities for each candle.
        """
        close_col = f"{asset}_close"
        high_col = f"{asset}_high"
        low_col = f"{asset}_low"
        atr_col = f"{asset}_atr"

        if close_col not in df.columns or atr_col not in df.columns:
            raise ValueError(f"Required columns missing for {asset}")
        
        prices_close = df[close_col].values
        prices_high = df[high_col].values
        prices_low = df[low_col].values
        atrs = df[atr_col].values
        timestamps = df.index
        
        labels = []
        indices = []
        
        spread = DEFAULT_SPREADS.get(asset, 0.0)
        
        n = len(df)
        for i in range(n - 1):
            mid_price = prices_close[i]
            atr = atrs[i]
            
            if np.isnan(atr) or atr == 0:
                continue
                
            tp_dist = self.tp_mult * atr
            sl_dist = self.sl_mult * atr
            
            direction = 0
            
            # --- Buy Barrier setup ---
            entry_ask = mid_price + (spread / 2.0)
            buy_tp = entry_ask + tp_dist
            buy_sl = entry_ask - sl_dist

            # --- Sell Barrier setup ---
            entry_bid = mid_price - (spread / 2.0)
            sell_tp = entry_bid - tp_dist
            sell_sl = entry_bid + sl_dist

            buy_active = True
            sell_active = True

            for j in range(i + 1, min(i + 1 + self.max_bars, n)):
                b_high = prices_high[j] - (spread / 2.0)
                b_low = prices_low[j] - (spread / 2.0)
                a_high = prices_high[j] + (spread / 2.0)
                a_low = prices_low[j] + (spread / 2.0)

                # Evaluate Buy side
                if buy_active:
                    if b_low <= buy_sl:
                        buy_active = False # SL hit
                    elif b_high >= buy_tp:
                        direction = 1
                        break # Success

                # Evaluate Sell side
                if sell_active:
                    if a_high >= sell_sl:
                        sell_active = False # SL hit
                    elif a_low <= sell_tp:
                        direction = -1
                        break # Success
                
                if not buy_active and not sell_active:
                    break
            
            labels.append({'direction': direction})
            indices.append(timestamps[i])
            
        return pd.DataFrame(labels, index=indices)

if __name__ == "__main__":
    pass

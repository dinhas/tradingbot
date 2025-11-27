import pandas as pd
import numpy as np

class TechnicalIndicators:
    """
    A pure pandas/numpy implementation of technical indicators to replace pandas_ta.
    This ensures compatibility with environments like Kaggle where installing custom libraries
    might be restricted.
    """

    @staticmethod
    def sma(series: pd.Series, length: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=length).mean()

    @staticmethod
    def ema(series: pd.Series, length: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=length, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, length: int = 14) -> pd.Series:
        """Relative Strength Index (Wilder's Smoothing)"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).abs()
        loss = (delta.where(delta < 0, 0)).abs()

        # Wilder's Smoothing
        # First value is SMA
        avg_gain = gain.rolling(window=length).mean()
        avg_loss = loss.rolling(window=length).mean()
        
        # Subsequent values are smoothed
        # We can use ewm with alpha=1/length for Wilder's smoothing equivalent
        # But to match pandas_ta exactly, we often use the explicit formula or ewm(alpha=1/L)
        # pandas_ta uses: rma = ewm(alpha=1/length, adjust=False)
        
        avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands"""
        mid = series.rolling(window=length).mean()
        s = series.rolling(window=length).std()
        upper = mid + (std * s)
        lower = mid - (std * s)
        
        # Return with names matching pandas_ta somewhat or just standard names
        return pd.DataFrame({
            f'BBL_{length}_{std}': lower,
            f'BBM_{length}_{std}': mid,
            f'BBU_{length}_{std}': upper
        })

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Moving Average Convergence Divergence"""
        fast_ema = series.ewm(span=fast, adjust=False).mean()
        slow_ema = series.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        
        return pd.DataFrame({
            f'MACD_{fast}_{slow}_{signal}': macd_line,
            f'MACDh_{fast}_{slow}_{signal}': hist, # Histogram
            f'MACDs_{fast}_{slow}_{signal}': signal_line
        })

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
        """Average True Range"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Wilder's Smoothing for ATR
        return tr.ewm(alpha=1/length, adjust=False).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
        """Average Directional Index"""
        up = high.diff()
        down = -low.diff()
        
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        
        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)
        
        # Smoothed DM and TR
        # ATR calculation again for consistency
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/length, adjust=False).mean() / (atr + 1e-9))
        minus_di = 100 * (minus_dm.ewm(alpha=1/length, adjust=False).mean() / (atr + 1e-9))
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        adx = dx.ewm(alpha=1/length, adjust=False).mean()
        
        return pd.DataFrame({
            f'ADX_{length}': adx,
            f'DMP_{length}': plus_di,
            f'DMN_{length}': minus_di
        })

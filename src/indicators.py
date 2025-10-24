import pandas as pd
import numpy as np
from typing import Union

def calculate_donchian_channels(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculate Donchian Channels (highest high and lowest low over period).

    Args:
        df: DataFrame with 'high' and 'low' columns
        period: Lookback period for channels

    Returns:
        DataFrame with added 'donchian_high' and 'donchian_low' columns
    """
    df = df.copy()
    df['donchian_high'] = df['high'].rolling(window=period).max()
    df['donchian_low'] = df['low'].rolling(window=period).min()
    return df

def calculate_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Period for ATR calculation

    Returns:
        DataFrame with added 'atr' column
    """
    df = df.copy()

    # Calculate True Range components
    df['h_l'] = df['high'] - df['low']
    df['h_pc'] = abs(df['high'] - df['close'].shift(1))
    df['l_pc'] = abs(df['low'] - df['close'].shift(1))

    # True Range is the maximum of the three
    df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)

    # ATR is the rolling mean of True Range
    df['atr'] = df['tr'].rolling(window=period).mean()

    # Clean up intermediate columns
    df.drop(['h_l', 'h_pc', 'l_pc', 'tr'], axis=1, inplace=True)

    return df

def calculate_volume_ma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """
    Calculate Simple Moving Average of Volume.

    Args:
        df: DataFrame with 'volume' column
        period: Period for moving average

    Returns:
        DataFrame with added 'volume_ma' column
    """
    df = df.copy()
    df['volume_ma'] = df['volume'].rolling(window=period).mean()
    return df

def add_all_indicators(df: pd.DataFrame,
                       donchian_period: int = 20,
                       atr_period: int = 14,
                       volume_ma_period: int = 20) -> pd.DataFrame:
    """
    Add all technical indicators to DataFrame at once.
    Useful for pre-calculating with default periods.

    Args:
        df: DataFrame with OHLCV data
        donchian_period: Period for Donchian channels
        atr_period: Period for ATR
        volume_ma_period: Period for volume MA

    Returns:
        DataFrame with all indicators added
    """
    df = calculate_donchian_channels(df, donchian_period)
    df = calculate_atr(df, atr_period)
    df = calculate_volume_ma(df, volume_ma_period)
    return df

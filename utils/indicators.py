"""
This module provides functions for calculating various technical indicators
on financial market data. It uses the pandas and pandas-ta libraries
for efficient and accurate calculations.
"""
import pandas as pd
import pandas_ta as ta
from utils.logger import system_logger as log

def calculate_scalping_indicators(candles):
    """
    Calculates scalping-focused technical indicators from a list of candles.

    Args:
        candles (list): A list of candle dictionaries. Each dictionary must
                        contain 'open', 'high', 'low', 'close', and 'volume'.

    Returns:
        dict: A dictionary containing the latest values for each calculated
              indicator, or None if there is not enough data.
    """
    if not candles:
        log.warning("Indicator calculation received an empty candle list.")
        return None

    # The longest period needed for these indicators is 21 (for EMA).
    required_candles = 21
    if len(candles) < required_candles:
        log.info(f"Not enough candle data to calculate indicators. "
                 f"Have {len(candles)}, need {required_candles}.")
        return None

    try:
        # Convert list of dicts to a DataFrame
        df = pd.DataFrame(candles)

        # Set the timestamp as the index (optional, but good practice)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Calculate indicators using pandas-ta
        # EMA (8, 21)
        df.ta.ema(length=8, append=True)
        df.ta.ema(length=21, append=True)

        # RSI (9)
        df.ta.rsi(length=9, append=True)

        # Stochastic (5,3,3)
        df.ta.stoch(k=5, d=3, smooth_k=3, append=True)

        # Bollinger Bands (20,2)
        df.ta.bbands(length=20, std=2, append=True)

        # ATR (14)
        df.ta.atr(length=14, append=True)

        # Get the latest values
        latest_indicators = df.iloc[-1]

        # Return a dictionary with the indicator values
        return {
            "ema_8": latest_indicators.get("EMA_8"),
            "ema_21": latest_indicators.get("EMA_21"),
            "rsi_9": latest_indicators.get("RSI_9"),
            "stoch_k": latest_indicators.get("STOCHk_5_3_3"),
            "stoch_d": latest_indicators.get("STOCHd_5_3_3"),
            "bb_upper": latest_indicators.get("BBU_20_2.0"),
            "bb_lower": latest_indicators.get("BBL_20_2.0"),
            "bb_mid": latest_indicators.get("BBM_20_2.0"),
            "atr": latest_indicators.get("ATRr_14"),
        }
    except Exception as e:
        log.error(f"An error occurred during indicator calculation: {e}")
        return None

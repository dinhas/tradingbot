import unittest
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.indicators import calculate_scalping_indicators

class TestIndicatorCalculations(unittest.TestCase):
    def setUp(self):
        # Create a sample list of 50 candles for testing
        self.candles = []
        start_time = datetime(2025, 1, 1, 0, 0)
        for i in range(50):
            self.candles.append({
                "timestamp": start_time + timedelta(minutes=i*5),
                "open": 1.1 + i*0.001,
                "high": 1.105 + i*0.001,
                "low": 1.095 + i*0.001,
                "close": 1.102 + i*0.001,
                "volume": 100 + i*10
            })

    def test_successful_calculation(self):
        """Test that indicators are calculated successfully with enough data."""
        indicators = calculate_scalping_indicators(self.candles)
        self.assertIsNotNone(indicators)
        self.assertIsInstance(indicators, dict)
        # Check for one of the expected keys
        self.assertIn("ema_8", indicators)
        self.assertIsNotNone(indicators["ema_8"])

    def test_not_enough_data(self):
        """Test that the function returns None when there is not enough data."""
        short_candle_list = self.candles[:10] # Only 10 candles
        indicators = calculate_scalping_indicators(short_candle_list)
        self.assertIsNone(indicators)

    def test_empty_candle_list(self):
        """Test that the function returns None for an empty list."""
        indicators = calculate_scalping_indicators([])
        self.assertIsNone(indicators)

    def test_indicator_values(self):
        """
        Test the actual calculated values against expected values from pandas-ta.
        This provides a more robust check.
        """
        indicators = calculate_scalping_indicators(self.candles)

        # Manually calculate the expected values for comparison
        df = pd.DataFrame(self.candles)
        df.set_index(pd.to_datetime(df['timestamp']), inplace=True)

        # Calculate EMA_8 manually
        expected_ema_8 = df.ta.ema(length=8).iloc[-1]

        # Calculate RSI_9 manually
        expected_rsi_9 = df.ta.rsi(length=9).iloc[-1]

        self.assertAlmostEqual(indicators['ema_8'], expected_ema_8, places=5)
        self.assertAlmostEqual(indicators['rsi_9'], expected_rsi_9, places=5)

if __name__ == '__main__':
    unittest.main()

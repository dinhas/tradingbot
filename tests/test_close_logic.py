import unittest
import pandas as pd
import numpy as np
from src.trading_env import TradingEnv

class TestPositionClosing(unittest.TestCase):
    def setUp(self):
        # Create a dummy environment
        self.env = TradingEnv(stage=1, is_training=False)
        
        # Mock data to ensure we have valid prices
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        self.env.raw_data = pd.DataFrame({
            'EURUSD_close': [1.1000] * 100,
            'EURUSD_atr_14': [0.0010] * 100,
            # Add other assets to avoid errors
            'GBPUSD_close': [1.2000] * 100, 'GBPUSD_atr_14': [0.0010] * 100,
            'USDJPY_close': [150.00] * 100, 'USDJPY_atr_14': [0.1000] * 100,
            'USDCHF_close': [0.9000] * 100, 'USDCHF_atr_14': [0.0010] * 100,
            'XAUUSD_close': [2000.0] * 100, 'XAUUSD_atr_14': [10.000] * 100,
        }, index=dates)
        
        # Mock processed data for session info
        self.env.processed_data = pd.DataFrame({
            'session_london': [1] * 100,
            'session_ny': [0] * 100,
            'session_overlap': [0] * 100
        }, index=dates)
        
        self.env.max_steps = 100
        self.env.reset()

    def test_close_position_logic(self):
        print("\n--- Testing Position Closing Logic ---")
        
        # 1. Open a BUY position on EURUSD
        # Action: [1.0 (Buy), 0, 0, 0, 0] (assuming EURUSD is index 0)
        open_action = np.array([1.0, 0, 0, 0, 0], dtype=np.float32)
        
        print("Step 1: Sending BUY signal (1.0)")
        self.env.step(open_action)
        
        # Verify position is open
        pos = self.env.positions['EURUSD']
        self.assertIsNotNone(pos, "Position should be open after BUY signal")
        self.assertEqual(pos['direction'], 1, "Position direction should be BUY (1)")
        print(f"Position opened successfully: {pos['direction']} at {pos['entry_price']}")
        
        # 2. Send HOLD signal (Same direction)
        # Action: [1.0 (Buy), ...]
        print("Step 2: Sending BUY signal again (1.0) - Should HOLD")
        self.env.step(open_action)
        pos = self.env.positions['EURUSD']
        self.assertIsNotNone(pos, "Position should remain open")
        print("Position held successfully")
        
        # 3. Send CLOSE signal (0.0)
        # Action: [0.0 (Close), ...]
        # Note: Logic is -0.33 < x < 0.33 is CLOSE
        close_action = np.array([0.0, 0, 0, 0, 0], dtype=np.float32)
        
        print("Step 3: Sending CLOSE signal (0.0)")
        self.env.step(close_action)
        
        # Verify position is closed
        pos = self.env.positions['EURUSD']
        self.assertIsNone(pos, "Position should be CLOSED (None) after 0.0 signal")
        print("Position closed successfully!")
        
        # Check trade log
        last_trade = self.env.completed_trades[-1]
        print(f"Trade Logged: {last_trade['asset']} {last_trade['action']} PnL: {last_trade['pnl']}")

if __name__ == '__main__':
    unittest.main()

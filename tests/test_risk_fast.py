import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trading_env import TradingEnv

class FastTradingEnv(TradingEnv):
    def _load_data(self):
        print("Using Dummy Data for Fast Test")
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
        df = pd.DataFrame({
            'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 100
        }, index=dates)
        return {asset: df for asset in self.assets}

def test_risk_fast():
    print("Initializing FastTradingEnv...")
    env = FastTradingEnv(stage=3, is_training=True)
    print("✅ Environment initialized.")

    # Test Reset
    obs, info = env.reset()
    
    # 1. Test Risk Management (Global Exposure)
    print("\nTesting Risk Management (Global Exposure)...")
    
    # Asset 0: Buy, Max Size
    action_0 = np.zeros(20)
    action_0[0] = 1.0 # Buy
    action_0[1] = 1.0 # Max Size (50% equity)
    
    print(f"Attempting to open Position 0 with Max Size (50% Equity)...")
    obs, reward, term, trunc, info = env.step(action_0)
    pos_0 = env.positions[env.assets[0]]
    
    if pos_0:
        print(f"✅ Position 0 opened. Size: {pos_0['size']:.2f}")
    else:
        print("❌ Position 0 failed to open!")
        
    # Asset 1: Buy, Max Size (Should be rejected)
    # Current Exposure = 50%. New Req = 50%. Total = 100% > 60%.
    action_1 = np.zeros(20)
    action_1[0] = 1.0 # Maintain Buy on Asset 0
    action_1[1] = 1.0 # Maintain Max Size on Asset 0
    action_1[4] = 1.0 # Buy Asset 1
    action_1[5] = 1.0 # Max Size
    
    print(f"Attempting to open Position 1 with Max Size (50% Equity)...")
    obs, reward, term, trunc, info = env.step(action_1)
    pos_1 = env.positions[env.assets[1]]
    
    if pos_1 is None:
        print("✅ Position 1 rejected due to Global Exposure limit (Expected).")
    else:
        print(f"❌ Position 1 opened! Risk Management Failed. Size: {pos_1['size']:.2f}")

if __name__ == "__main__":
    test_risk_fast()

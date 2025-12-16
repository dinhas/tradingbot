import numpy as np
import gymnasium as gym
from risk_env import RiskManagementEnv

class MockRiskEnv(RiskManagementEnv):
    """Subclass to mock data for testing logic."""
    def __init__(self):
        # Bypass standard init to manual setup
        # But we need super init to set constants.
        # We'll override _load_data to do nothing first, then populate.
        super().__init__(dataset_path="dummy", initial_equity=100.0, is_training=False)
        
    def _load_data(self):
        # Create dummy data
        self.n_samples = 10
        self.features_array = np.zeros((10, 140), dtype=np.float32)
        self.entry_prices = np.ones(10, dtype=np.float32) * 1.0
        self.atrs = np.ones(10, dtype=np.float32) * 0.001
        self.directions = np.ones(10, dtype=np.float32) # LONG
        self.is_usd_quote_arr = np.ones(10, dtype=bool)
        self.is_usd_base_arr = np.zeros(10, dtype=bool)
        self.close_prices = np.ones(10, dtype=np.float32) * 1.0
        self.hours_to_exits = np.zeros(10, dtype=np.float32)
        self.has_time_limit = False
        
        # Controlled Outcomes
        self.max_profit_pcts = np.zeros(10, dtype=np.float32)
        self.max_loss_pcts = np.zeros(10, dtype=np.float32)
        
        # Index 0: BAD SIGNAL (Crash)
        # Max Loss: -1.0% (Deep drawdown > 0.05% limit)
        # Max Profit: 0.01%
        self.max_loss_pcts[0] = -0.01 
        self.max_profit_pcts[0] = 0.0001
        
        # Index 1: GOOD SIGNAL (Moon)
        # Max Loss: -0.01% (Tiny drawdown < 0.05% limit)
        # Max Profit: 1.0% (Huge gain > 0.1% target)
        self.max_loss_pcts[1] = -0.0001
        self.max_profit_pcts[1] = 0.01
        
    def set_step(self, step_idx):
        self.episode_start_idx = 0
        self.current_step = step_idx

def test_reward_logic():
    env = MockRiskEnv()
    env.reset()
    
    print("--- Test 1: Blocking a BAD SIGNAL (Should be REWARDED) ---")
    env.set_step(0) # Index 0 is BAD
    # Action: SL=1.0, TP=1.0, Risk=-1.0 (Mapped to 0.0 -> Blocked)
    action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Signal: MaxLoss={info.get('max_adv'):.4f}, MaxProfit={info.get('max_fav'):.4f}")
    print(f"Action: BLOCKED. Reward: {reward:.4f}")
    print(f"Info: {info}")
    
    if reward > 0 and info['block_type'] == "GOOD_BLOCK_SAVED":
        print("PASS: Correctly rewarded for blocking bad signal.")
    else:
        print("FAIL: Expected positive reward for Good Block.")

    print("\n--- Test 2: Blocking a GOOD SIGNAL (Should be PENALIZED) ---")
    env.set_step(1) # Index 1 is GOOD
    # Action: Block
    action = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Signal: MaxLoss={info.get('max_adv'):.4f}, MaxProfit={info.get('max_fav'):.4f}")
    print(f"Action: BLOCKED. Reward: {reward:.4f}")
    print(f"Info: {info}")
    
    if reward < 0 and info['block_type'] == "BAD_BLOCK":
        print("PASS: Correctly penalized for blocking good signal.")
    else:
        print("FAIL: Expected negative penalty for Bad Block.")

    print("\n--- Test 3: Trading a BAD SIGNAL (Should be PENALIZED by PnL) ---")
    # For this to generate PnL, we need close price to reflect the crash.
    # In mock, close_price is 1.0 (flat). But let's assume it hit SL?
    # risk_env logic checks for SL hit.
    # Bad signal has MaxLoss -1.0%. Agent sets SL?
    # Let's set SL tight. Action[0] -> SL Mult.
    # ATR is 0.001 (0.1%). Entry 1.0. SL 1ATR = 0.001 dist.
    # Max loss is -0.01 (1%). So it definitely hits SL.
    
    env.set_step(0)
    # Action: Risk 0.5 (Mapped to ~0.2)
    action = np.array([0.0, 0.0, 0.0], dtype=np.float32) 
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Action: TRADED. Reward: {reward:.4f}")
    print(f"Info: {info}")
    
    if reward < 0 and info['exit'] == 'SL':
        print("PASS: Correctly penalized for trading bad signal (Hit SL).")
    else:
        print("FAIL: Expected negative PnL reward.")

if __name__ == "__main__":
    test_reward_logic()

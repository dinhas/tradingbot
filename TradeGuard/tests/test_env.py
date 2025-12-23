import unittest
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from TradeGuard.src.trade_guard_env import TradeGuardEnv
except ImportError:
    # We expect this to fail in the Red phase
    TradeGuardEnv = None

class TestTradeGuardEnv(unittest.TestCase):
    def setUp(self):
        self.dataset_path = 'TradeGuard/data/test_dataset.parquet'
        if not os.path.exists(self.dataset_path):
            from TradeGuard.tests.generate_test_data import generate_test_dataset
            generate_test_dataset()
        
        self.config = {
            'env': {
                'dataset_path': self.dataset_path,
                'reward_scaling': 1.0,
                'penalty_factors': {
                    'missed_win': 0.5,
                    'loss_avoided': 0.1
                },
                'seed': 42
            }
        }

    def test_env_initialization(self):
        self.assertIsNotNone(TradeGuardEnv, "TradeGuardEnv not implemented yet")
        env = TradeGuardEnv(self.config)
        self.assertEqual(env.observation_space.shape, (105,))
        self.assertEqual(env.action_space.n, 2)

    def test_env_reset(self):
        env = TradeGuardEnv(self.config)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (105,))
        self.assertEqual(env.current_step, 0)

    def test_env_step_allow(self):
        env = TradeGuardEnv(self.config)
        env.reset()
        # Action 1: Allow
        obs, reward, terminated, truncated, info = env.step(1)
        
        # Calculate expected reward
        # Row 0 pnl sum
        df = pd.read_parquet(self.dataset_path)
        assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        expected_reward = sum(df.iloc[0][f'pnl_{a}'] for a in assets)
        
        self.assertAlmostEqual(reward, expected_reward, places=5)
        self.assertEqual(env.current_step, 1)

    def test_env_step_block(self):
        env = TradeGuardEnv(self.config)
        env.reset()
        # Action 0: Block
        obs, reward, terminated, truncated, info = env.step(0)
        
        # Calculate expected reward for Block
        df = pd.read_parquet(self.dataset_path)
        assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        row = df.iloc[0]
        expected_reward = 0
        for a in assets:
            pnl = row[f'pnl_{a}']
            target = row[f'target_{a}'] # 1 if it was a win (hit TP)
            if target == 1:
                # Missed Win Penalty: -pnl * 0.5
                expected_reward -= pnl * 0.5
            else:
                # Loss Avoided Incentive: +abs(pnl) * 0.1 if pnl < 0
                if pnl < 0:
                    expected_reward += abs(pnl) * 0.1
        
        self.assertAlmostEqual(reward, expected_reward, places=5)

    def test_env_termination(self):
        env = TradeGuardEnv(self.config)
        env.reset()
        # Step through all rows (10 rows in test dataset)
        for i in range(9):
            obs, reward, terminated, truncated, info = env.step(1)
            self.assertFalse(terminated)
        
        obs, reward, terminated, truncated, info = env.step(1)
        self.assertTrue(terminated)

if __name__ == '__main__':
    unittest.main()

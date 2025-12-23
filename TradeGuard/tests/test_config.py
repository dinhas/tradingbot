import unittest
import yaml
import os

class TestConfig(unittest.TestCase):
    def test_config_exists(self):
        config_path = "TradeGuard/config/ppo_config.yaml"
        self.assertTrue(os.path.exists(config_path), f"Config file {config_path} does not exist")

    def test_config_structure(self):
        config_path = "TradeGuard/config/ppo_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.assertIn('ppo', config)
        self.assertIn('env', config)
        
        # Check PPO params
        ppo = config['ppo']
        self.assertIn('learning_rate', ppo)
        self.assertIn('n_steps', ppo)
        self.assertIn('batch_size', ppo)
        
        # Check Env params
        env = config['env']
        self.assertIn('reward_scaling', env)
        self.assertIn('penalty_factors', env)

if __name__ == '__main__':
    unittest.main()

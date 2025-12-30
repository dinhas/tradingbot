import os
import yaml
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from TradeGuard.src.trade_guard_env import TradeGuardEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradeGuardTrainer:
    def __init__(self, config_path, config_override=None):
        self.config_path = config_path
        self.config = self._load_config()
        if config_override:
            self._update_recursive(self.config, config_override)
        self.env = self._setup_env()
        self.model = self._setup_model()
        
    def _update_recursive(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_recursive(d.get(k, {}), v)
            else:
                d[k] = v
        return d
        
    def _load_config(self):
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_env(self):
        # Allow overriding dataset_path for testing
        env_config = self.config.copy()
        
        # Use SubprocVecEnv for parallel training
        import multiprocessing
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from stable_baselines3.common.monitor import Monitor
        
        n_cpu = multiprocessing.cpu_count()
        # Create a function that returns the environment
        def make_env(rank, seed=0):
            def _init():
                env = TradeGuardEnv(env_config)
                env = Monitor(env)
                env.reset(seed=seed + rank)
                return env
            return _init
            
        logger.info(f"Creating {n_cpu} parallel environments...")
        return SubprocVecEnv([make_env(i) for i in range(n_cpu)])
        
    def _setup_model(self):
        ppo_params = self.config['ppo']
        
        # Check if tensorboard is installed
        tensorboard_log = "TradeGuard/logs/"
        try:
            import tensorboard
        except ImportError:
            logger.warning("Tensorboard not installed. Disabling tensorboard logging.")
            tensorboard_log = None

        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=1,
            learning_rate=ppo_params['learning_rate'],
            n_steps=ppo_params['n_steps'],
            batch_size=ppo_params['batch_size'],
            gamma=ppo_params['gamma'],
            gae_lambda=ppo_params['gae_lambda'],
            ent_coef=ppo_params['ent_coef'],
            vf_coef=ppo_params['vf_coef'],
            max_grad_norm=ppo_params['max_grad_norm'],
            target_kl=ppo_params['target_kl'],
            tensorboard_log=tensorboard_log
        )
        return model
        
    def train(self, total_timesteps=100000):
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        logger.info("Training complete.")
        
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to {path}")

if __name__ == "__main__":
    trainer = TradeGuardTrainer("TradeGuard/config/ppo_config.yaml")
    # You might want to get total_timesteps from config or CLI
    trainer.train(total_timesteps=1000000)
    trainer.save("TradeGuard/models/tradeguard_ppo_final")

import os
import sys
import logging
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from risk_ppo_env import VectorizedRiskEnv
from risk_model_ppo import RiskActorCriticPolicy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
MARKET_DATA_DIR = os.environ.get("MARKET_DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))
PPO_DATASET_PATH = os.environ.get("PPO_DATASET_PATH", os.path.join(os.path.dirname(__file__), "data", "rl_risk_dataset.parquet"))

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

CONFIG_B = dict(
    learning_rate       = 1e-4,       
    n_steps             = 4096,       
    batch_size          = 1024, # Increased for better CPU throughput
    n_epochs            = 10,
    gamma               = 0.995,      
    gae_lambda          = 0.95,
    clip_range          = 0.15,
    ent_coef            = 0.005,      
    vf_coef             = 0.6,        
    max_grad_norm       = 0.5,
    
    policy_kwargs = dict(
        activation_fn = torch.nn.Tanh,
    ),

    total_timesteps     = 3_000_000,
)

class TradingMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TradingMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if "trade/win_rate" in info:
                for key, value in info.items():
                    if isinstance(value, (int, float, np.float32, np.float64)):
                        self.logger.record(f"env/{key}", value)
        return True

def train():
    logger.info(f"Starting Optimized PPO Risk Model Training on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # 1. Load Data
    logger.info("Loading signals and price data for VectorizedRiskEnv...")
    if not os.path.exists(PPO_DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {PPO_DATASET_PATH}. Please run generate_rl_risk_dataset.py first.")

    signals_df = pd.read_parquet(PPO_DATASET_PATH)
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    price_data = {}
    for a in assets:
        price_data[a] = pd.read_parquet(f"{MARKET_DATA_DIR}/{a}_5m.parquet")
        if f'atr_14' not in price_data[a].columns:
             # Fallback to simple ATR if missing
             price_data[a]['atr_14'] = price_data[a]['close'].rolling(14).apply(lambda x: np.max(x)-np.min(x)).fillna(method='bfill')

    # 2. Create Vectorized Environment
    env = VectorizedRiskEnv(signals_df, price_data, n_envs=8)
    
    # 3. Initialize Model
    model = PPO(
        RiskActorCriticPolicy,
        env,
        verbose=1,
        device="auto",
        tensorboard_log=LOG_DIR,
        **{k: v for k, v in CONFIG_B.items() if k != 'total_timesteps'}
    )
    
    # 4. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=MODELS_DIR,
        name_prefix="ppo_risk_model_v2_opt"
    )
    metrics_callback = TradingMetricsCallback()
    
    # 5. Train
    logger.info("Entering training loop...")
    model.learn(
        total_timesteps=CONFIG_B['total_timesteps'],
        callback=[checkpoint_callback, metrics_callback],
        tb_log_name="PPO_Risk_V2_Optimized"
    )
    
    # 6. Save Final Model
    model_path = os.path.join(MODELS_DIR, "ppo_risk_model_final_v2_opt.zip")
    model.save(model_path)
    
    logger.info(f"Training Complete. Model saved to {model_path}")

if __name__ == "__main__":
    train()

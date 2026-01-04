import os
import sys
import logging
import argparse
import yaml
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from TradeGuard.src.download_data import DataFetcherTraining
from TradeGuard.src.generate_dataset import TrainingDatasetGenerator
from TradeGuard.src.train_guard import TradeGuardTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline(force_download=False, force_generate=False, config_path="TradeGuard/config/ppo_config.yaml"):
    # 1. Configuration
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = config['env'].get('dataset_path', "TradeGuard/data/tradeguard_dataset.parquet")
    dataset_file = data_dir
    market_data_dir = os.path.dirname(dataset_file)
    
    # 2. Download Data
    logger.info("--- Step 1: Data Download ---")
    fetcher = DataFetcherTraining(output_dir=market_data_dir, force=force_download)
    # Check if files already exist is handled inside fetcher.start() -> on_connected
    # but since it uses Twisted reactor, we just start it.
    fetcher.start() 
    
    # 3. Generate Dataset
    logger.info("--- Step 2: Dataset Generation ---")
    if os.path.exists(dataset_file) and not force_generate:
        logger.info(f"✅ Dataset file {dataset_file} already exists. Skipping generation.")
    else:
        # We need the Alpha model for generation
        alpha_model_path = "models/checkpoints/alpha/ppo_final_model.zip"
        if not os.path.exists(alpha_model_path):
            logger.error(f"Alpha model not found at {alpha_model_path}. Cannot generate dataset.")
            return
            
        generator = TrainingDatasetGenerator(alpha_model_path, data_dir=market_data_dir)
        # Note: generator.generate saves to its own default or takes an argument
        generator.generate(output_file=dataset_file, chunk_size=50000)

    # 4. Training
    logger.info("--- Step 3: Model Training ---")
    trainer = TradeGuardTrainer(config_path)
    # Train for the amount of steps specified in config or a default
    total_timesteps = config['ppo'].get('total_timesteps', 1000000)
    trainer.train(total_timesteps=total_timesteps)
    
    model_save_path = os.path.join("TradeGuard/models", f"tradeguard_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    trainer.save(model_save_path)
    logger.info(f"✅ Pipeline complete. Model saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradeGuard Full Pipeline")
    parser.add_argument("--force-download", action="store_true", help="Force redownload of market data")
    parser.add_argument("--force-generate", action="store_true", help="Force regeneration of training dataset")
    parser.add_argument("--config", default="TradeGuard/config/ppo_config.yaml", help="Path to PPO config")
    
    args = parser.parse_args()
    run_pipeline(force_download=args.force_download, force_generate=args.force_generate, config_path=args.config)

import numpy as np
import pandas as pd
import random
import torch
import logging
import os
from collections import deque
from Risklayer.config import config
from Risklayer.data_loader import DataLoader
from Risklayer.feature_engineering import FeatureEngine
from Risklayer.peak_labeling import StructuralLabeler
from Risklayer.trading_env import TradingEnv
from Risklayer.sac_agent import SACAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def generate_synthetic_data(assets: list) -> dict:
    """Generates synthetic OHLCV data for testing/sanity purposes."""
    logger.info("Generating synthetic data since real data was not found...")
    data_dict = {}
    dates = pd.date_range(start='2016-01-01', periods=5000, freq='5min')
    for asset in assets:
        # Simple random walk
        close = 1.1 + np.cumsum(np.random.normal(0, 0.0005, len(dates)))
        df = pd.DataFrame({
            'open': close,
            'high': close + 0.0005,
            'low': close - 0.0005,
            'close': close,
            'volume': np.random.randint(10, 100, len(dates))
        }, index=dates)
        data_dict[asset] = df
    return data_dict

def train():
    # 1. Load Data
    loader = DataLoader()
    if not os.path.exists(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
        data_dict = generate_synthetic_data(config.ASSETS)
    else:
        data_dict = loader.load_all_data()

    if not data_dict:
        data_dict = generate_synthetic_data(config.ASSETS)

    aligned_df = loader.align_data(data_dict)

    # 2. Feature Engineering
    fe = FeatureEngine()
    feature_df = fe.calculate_features(aligned_df)

    # 3. Labeling
    labeler = StructuralLabeler(reversal_multiplier=config.ATR_REVERSAL_MULTIPLIER)
    for asset in config.ASSETS:
        aligned_df = labeler.label_data(aligned_df, asset)

    # Combine features and labels into final DF
    final_df = pd.concat([aligned_df, feature_df], axis=1)

    # 4. Initialize Environment & Agent
    env = TradingEnv(final_df, asset='EURUSD')
    agent = SACAgent()
    memory = ReplayBuffer(config.BUFFER_SIZE)

    # 5. Training Loop
    total_steps = 0
    updates = 0
    best_reward = -np.inf

    while total_steps < config.TOTAL_STEPS:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            if total_steps < 1000: # Warmup
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            mask = 1 if not terminated else 0
            memory.push(state, action, reward, next_state, mask)

            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            if len(memory) > config.BATCH_SIZE:
                agent.update_parameters(memory, config.BATCH_SIZE, updates)
                updates += 1

            if total_steps % 1000 == 0:
                logger.info(f"Step: {total_steps}, Equity: {info['equity']:.2f}, DD: {info['drawdown']:.2%}")
                # Print a sample trade if it just closed
                if not env.position:
                    logger.info(f"Sample Trade Result - Equity: {info['equity']:.2f}")

        logger.info(f"Episode Finished. Reward: {episode_reward:.2f}, Steps: {episode_steps}, Avg Reward: {episode_reward/episode_steps:.4f}")

    logger.info("Training Complete.")

if __name__ == "__main__":
    train()

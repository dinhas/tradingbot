import os
import torch
import numpy as np
from .trading_env import TradingEnv
from .sac_agent import SACAgent
from .config import config
from .data_loader import DataLoader
from .feature_engineering import FeatureEngineer
from .peak_labeling import StructuralLabeler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # 1. Initialize Environment
    # For sanity run, we limit the data to speed up preprocessing
    loader = DataLoader()
    raw_data = loader.load_all_data(limit_bars=50000)
    fe = FeatureEngineer()
    data_dict = fe.calculate_features(raw_data)
    labeler = StructuralLabeler()
    for asset in data_dict:
        data_dict[asset] = labeler.label_data(data_dict[asset])

    env = TradingEnv(data_dict=data_dict)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Initialize Agent
    agent = SACAgent(state_dim, action_dim)

    # 3. Training Loop
    total_steps = 10_000
    batch_size = config.BATCH_SIZE

    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    num_episodes = 0

    rewards_history = []
    equity_history = []

    logger.info(f"Starting sanity training loop for {total_steps} steps...")

    for step in range(total_steps):
        action = agent.select_action(state)

        next_state, reward, done, truncated, info = env.step(action)

        agent.memory.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        episode_steps += 1

        if len(agent.memory) > batch_size:
            agent.update_parameters(batch_size)

        if done or truncated:
            num_episodes += 1
            rewards_history.append(episode_reward)
            equity_history.append(info['equity'])

            if num_episodes % 1 == 0:
                logger.info(f"Step: {step+1} | Episode: {num_episodes} | Reward: {episode_reward:.2f} | Equity: {info['equity']:.2f} | DD: {info['drawdown']*100:.2f}%")

            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0

    # Print Summary Metrics
    avg_reward = np.mean(rewards_history) if rewards_history else 0
    final_equity = info['equity'] if 'info' in locals() else config.INITIAL_EQUITY

    print("\n--- SANITY TRAINING SUMMARY ---")
    print(f"Total Steps: {total_steps}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Final Equity: {final_equity:.2f}")
    print(f"Replay Buffer Size: {len(agent.memory)}")
    print("--------------------------------\n")

    # Save Model
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    agent.save_checkpoint(os.path.join(config.MODEL_SAVE_PATH, "sac_sanity_model.pth"))
    logger.info(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()

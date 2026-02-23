import os
import torch
import numpy as np
import logging
from .config import RiskConfig
from .trading_env import RiskTradingEnv
from .sac_agent import SACAgent, ReplayBuffer

def train():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    config = RiskConfig()
    env = RiskTradingEnv(config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim, config)
    replay_buffer = ReplayBuffer(config.REPLAY_SIZE)

    total_steps = 0
    max_steps = 10000
    episode_reward = 0
    episode_steps = 0
    episodes = 0

    state, _ = env.reset()

    logger.info("Starting sanity training loop (10,000 steps)...")

    all_rewards = []
    while total_steps < max_steps:
        if total_steps < 1000: # Warmup with random actions
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, done, truncated, info = env.step(action)

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        all_rewards.append(reward)
        episode_steps += 1
        total_steps += 1

        if info.get("net_pnl", 0) != 0:
            logger.info(f"Step {total_steps}: Trade Executed. Net PnL: {info['net_pnl']:.2f}, Reward: {reward:.2f}")

        if len(replay_buffer) > config.BATCH_SIZE and total_steps > 1000:
            agent.update_parameters(replay_buffer, config.BATCH_SIZE)

        if done or truncated:
            episodes += 1
            logger.info(f"Episode Finished | Total Steps: {total_steps} | Episode Reward: {episode_reward:.2f} | Final Equity: {env.equity:.2f}")

            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0

    logger.info("Training complete.")

    # Final metrics
    logger.info(f"Final Equity: {env.equity:.2f}")
    logger.info(f"Total Episodes: {episodes}")
    logger.info(f"Average Step Reward: {np.mean(all_rewards):.4f}")

if __name__ == "__main__":
    train()

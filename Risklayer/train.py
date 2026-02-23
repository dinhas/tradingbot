import os
import numpy as np
import torch
import logging
from tqdm import tqdm
from Risklayer.config import config
from Risklayer.trading_env import RiskTradingEnv
from Risklayer.sac_agent import SACAgent, ReplayBuffer

def train():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Train")

    # Set Seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # Initialize Environment and Agent
    env = RiskTradingEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = SACAgent(state_dim, action_dim)
    memory = ReplayBuffer(config.BUFFER_SIZE)

    # Training Loop
    total_steps = config.TOTAL_STEPS
    warmup_steps = config.WARMUP_STEPS
    batch_size = config.BATCH_SIZE

    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    num_episodes = 0

    logger.info("Starting training loop...")

    progress_bar = tqdm(range(total_steps), desc="Training")

    for step in progress_bar:
        if step < warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        episode_steps += 1

        # mask = 0 if terminated else 1. We ignore truncated for standard SAC updates
        # but here they are basically the same (end of data)
        mask = 0 if terminated else 1

        memory.push(state, action, reward, next_state, 1 - mask)

        state = next_state

        if step >= warmup_steps:
            agent.update_parameters(memory, batch_size)

        if terminated or truncated:
            if num_episodes % 1 == 0:
                progress_bar.set_postfix({
                    "Reward": f"{episode_reward:.2f}",
                    "Equity": f"{info['equity']:.0f}",
                    "DD": f"{info['drawdown']:.2%}"
                })

            # Print sample trades every now and then
            if num_episodes % 5 == 0:
                logger.info(f"Episode {num_episodes} finished. Steps: {episode_steps}, Reward: {episode_reward:.2f}, Final Equity: {info['equity']:.2f}")

            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            num_episodes += 1

    logger.info("Training complete.")

    # Final Sanity Check
    logger.info("Running final evaluation...")
    state, _ = env.reset()
    total_eval_reward = 0
    for _ in range(20):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_eval_reward += reward
        state = next_state
        print(f"Trade Evaluation - PnL: {info['pnl']:.2f}, Equity: {info['equity']:.2f}, Exit: {info['exit_type']}")
        if terminated or truncated:
            break

    print(f"Average Reward during Eval: {total_eval_reward / 20:.2f}")

if __name__ == "__main__":
    train()

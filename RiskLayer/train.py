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
    dates = pd.date_range(start="2016-01-01", periods=5000, freq="5min")
    for asset in assets:
        # Simple random walk
        close = 1.1 + np.cumsum(np.random.normal(0, 0.0005, len(dates)))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.0005,
                "low": close - 0.0005,
                "close": close,
                "volume": np.random.randint(10, 100, len(dates)),
            },
            index=dates,
        )
        data_dict[asset] = df
    return data_dict


class TrainingMonitor:
    def __init__(self, window_size=50):
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_pnls = deque(maxlen=window_size)
        self.win_history = deque(maxlen=window_size)
        self.drawdowns = deque(maxlen=window_size)
        
        self.training_metrics = [] # List of dicts from agent.update_parameters
        self.steps_history = []
        self.last_dashboard_step = 0

    def add_episode(self, reward, length, equity_delta, info):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_pnls.append(equity_delta)
        self.win_history.append(1 if equity_delta > 0 else 0)
        self.drawdowns.append(info.get('drawdown', 0))

    def add_step_metrics(self, step, metrics):
        self.training_metrics.append(metrics)
        if len(self.training_metrics) > 1000:
            self.training_metrics.pop(0)
        self.steps_history.append(step)
        if len(self.steps_history) > 1000:
            self.steps_history.pop(0)

    def _calc_sharpe(self):
        if len(self.episode_pnls) < 2: return 0.0
        returns = np.array(self.episode_pnls)
        avg = np.mean(returns)
        std = np.std(returns)
        return (avg / (std + 1e-9)) * np.sqrt(252) # Annualized approx

    def _calc_profit_factor(self):
        pnls = np.array(self.episode_pnls)
        pos = pnls[pnls > 0].sum()
        neg = abs(pnls[pnls < 0].sum())
        return pos / (neg + 1e-9)

    def _ascii_bar(self, val, max_val, width=20, color=""):
        if max_val == 0: return "[" + " " * width + "]"
        filled = int(min(val / max_val * width, width))
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def print_dashboard(self, step, total_steps):
        if not self.episode_rewards or not self.training_metrics: return
        
        os.system('cls' if os.name == 'nt' else 'clear')
        
        avg_reward = np.mean(self.episode_rewards)
        win_rate = np.mean(self.win_history)
        pf = self._calc_profit_factor()
        sharpe = self._calc_sharpe()
        max_dd = np.max(self.drawdowns) if self.drawdowns else 0
        
        latest = self.training_metrics[-1]
        
        print("="*80)
        print(f"🚀 SAC TRAINING DASHBOARD | Step: {step}/{total_steps} ({step/total_steps:.1%})")
        print("="*80)
        
        print(f"\n1️⃣  EPISODE METRICS (Rolling Window: {len(self.episode_rewards)})")
        print(f"   Reward:     Mean: {avg_reward:10.2f} | Min: {np.min(self.episode_rewards):10.2f} | Max: {np.max(self.episode_rewards):10.2f}")
        print(f"   Performance: Win Rate: {win_rate:7.1%} | Profit Factor: {pf:6.2f} | Sharpe: {sharpe:6.2f}")
        print(f"   Risk:        Max Drawdown: {max_dd:7.2%} | Avg Length: {np.mean(self.episode_lengths):7.1f}")
        
        print(f"\n2️⃣  CRITIC (Q-NETWORK) METRICS")
        print(f"   Q1 Loss:    {latest['q1_loss']:10.6f} {self._ascii_bar(latest['q1_loss'], max([m['q1_loss'] for m in self.training_metrics]) if self.training_metrics else 1)}")
        print(f"   Q2 Loss:    {latest['q2_loss']:10.6f} {self._ascii_bar(latest['q2_loss'], max([m['q2_loss'] for m in self.training_metrics]) if self.training_metrics else 1)}")
        print(f"   Q1 Mean:    {latest['q1_mean']:10.4f} | Q2 Mean: {latest['q2_mean']:10.4f} | Target Q: {latest['target_q_mean']:10.4f}")
        print(f"   Exp Var:    {latest['explained_variance']:10.4f} (Prediction Quality)")

        print(f"\n3️⃣  ACTOR (POLICY) METRICS")
        print(f"   Actor Loss: {latest['actor_loss']:10.6f} {self._ascii_bar(abs(latest['actor_loss']), max([abs(m['actor_loss']) for m in self.training_metrics]) if self.training_metrics else 1)}")
        print(f"   Entropy:    {latest['policy_entropy']:10.4f} | Alpha: {latest['alpha']:10.4f}")
        print(f"   Actions:    Mean: {latest['mean_action']:10.4f} | Std: {latest['action_std']:10.4f}")

        print(f"\n4️⃣  LEARNING RATES")
        print(f"   Actor LR: {latest['actor_lr']:.2e} | Critic LR: {latest['critic_lr']:.2e} | Alpha LR: {latest['alpha_lr']:.2e}")
        
        print("\n" + "="*80)
        self.last_dashboard_step = step


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
    final_df = final_df.iloc[-5000:] # VERIFICATION: Keep last 5000 rows

    # 4. Initialize Environment & Agent
    env = TradingEnv(final_df, asset="EURUSD")
    agent = SACAgent()
    memory = ReplayBuffer(config.BUFFER_SIZE)
    monitor = TrainingMonitor()

    # 5. Training Loop
    total_steps = 0
    updates = 0

    while total_steps < config.TOTAL_STEPS:
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        initial_equity = env.equity
        done = False

        while not done:
            if total_steps < 1000:  # Warmup
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
                metrics = agent.update_parameters(memory, config.BATCH_SIZE, updates)
                monitor.add_step_metrics(total_steps, metrics)
                updates += 1

            if total_steps % 100 == 0:
                monitor.print_dashboard(total_steps, config.TOTAL_STEPS)
            elif total_steps % 10 == 0:
                logger.info(
                    f"Step: {total_steps}, Equity: {info['equity']:.2f}, DD: {info['drawdown']:.2%}"
                )

        # Episode end
        monitor.add_episode(episode_reward, episode_steps, info['equity'] - initial_equity, info)
        
        if total_steps % 20000 != 0: # Don't overwrite dashboard immediately
            logger.info(
                f"Episode Finished. Reward: {episode_reward:.2f}, Steps: {episode_steps}, Avg Reward: {episode_reward / episode_steps:.4f}"
            )

    logger.info("Training Complete.")


if __name__ == "__main__":
    train()

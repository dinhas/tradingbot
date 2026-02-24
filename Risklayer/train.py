import numpy as np
import pandas as pd
import torch
import logging
import os
import csv
from datetime import datetime
from collections import deque
from Risklayer.config import config
from Risklayer.data_loader import DataLoader
from Risklayer.feature_engineering import FeatureEngine
from Risklayer.peak_labeling import StructuralLabeler
from Risklayer.trading_env import TradingEnv
from Risklayer.sac_agent import SACAgent


def setup_logging(log_dir: str = "logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state_buf = np.zeros((capacity, state_dim), dtype=np.float32)
        self.mask_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def push(self, state, action, reward, next_state, mask):
        self.state_buf[self.ptr] = state
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr, 0] = reward
        self.next_state_buf[self.ptr] = next_state
        self.mask_buf[self.ptr, 0] = mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            self.state_buf[idx],
            self.action_buf[idx],
            self.reward_buf[idx],
            self.next_state_buf[idx],
            self.mask_buf[idx],
        )

    def __len__(self):
        return self.size


class MetricsLogger:
    def __init__(self, log_dir: str = "logs", window_size: int = 100):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = os.path.join(log_dir, f"metrics_{timestamp}.csv")
        self.episodes_file = os.path.join(log_dir, f"episodes_{timestamp}.csv")

        self._init_csv_files()

        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_pnls = deque(maxlen=window_size)
        self.win_history = deque(maxlen=window_size)
        self.drawdowns = deque(maxlen=window_size)

        self.total_episodes = 0
        self.total_trades = 0
        self.wins = 0

    def _init_csv_files(self):
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "q1_loss",
                    "q2_loss",
                    "q1_mean",
                    "q2_mean",
                    "target_q_mean",
                    "actor_loss",
                    "policy_entropy",
                    "alpha",
                    "explained_variance",
                    "mean_action",
                    "action_std",
                ]
            )

        with open(self.episodes_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "total_steps",
                    "reward",
                    "length",
                    "pnl",
                    "win_rate",
                    "max_dd",
                    "sharpe",
                    "profit_factor",
                ]
            )

    def log_training_step(self, step: int, metrics: dict):
        if step % config.METRICS_LOG_INTERVAL != 0:
            return
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    step,
                    f"{metrics.get('q1_loss', 0):.6f}",
                    f"{metrics.get('q2_loss', 0):.6f}",
                    f"{metrics.get('q1_mean', 0):.4f}",
                    f"{metrics.get('q2_mean', 0):.4f}",
                    f"{metrics.get('target_q_mean', 0):.4f}",
                    f"{metrics.get('actor_loss', 0):.6f}",
                    f"{metrics.get('policy_entropy', 0):.4f}",
                    f"{metrics.get('alpha', 0):.4f}",
                    f"{metrics.get('explained_variance', 0):.4f}",
                    f"{metrics.get('mean_action', 0):.4f}",
                    f"{metrics.get('action_std', 0):.4f}",
                ]
            )

    def log_episode(
        self,
        total_steps: int,
        reward: float,
        length: int,
        equity_delta: float,
        info: dict,
    ):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_pnls.append(equity_delta)
        self.win_history.append(1 if equity_delta > 0 else 0)
        self.drawdowns.append(info.get("drawdown", 0))

        self.total_episodes += 1
        if equity_delta > 0:
            self.wins += 1

        win_rate = self.wins / self.total_episodes
        sharpe = self._calc_sharpe()
        profit_factor = self._calc_profit_factor()
        max_dd = max(self.drawdowns) if self.drawdowns else 0

        with open(self.episodes_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.total_episodes,
                    total_steps,
                    f"{reward:.4f}",
                    length,
                    f"{equity_delta:.4f}",
                    f"{win_rate:.4f}",
                    f"{max_dd:.4f}",
                    f"{sharpe:.4f}",
                    f"{profit_factor:.4f}",
                ]
            )

        return {
            "win_rate": win_rate,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "max_dd": max_dd,
        }

    def _calc_sharpe(self) -> float:
        if len(self.episode_pnls) < 2:
            return 0.0
        returns = np.array(self.episode_pnls)
        avg = np.mean(returns)
        std = np.std(returns)
        if std < 1e-9:
            return 0.0
        return (avg / std) * np.sqrt(252)

    def _calc_profit_factor(self) -> float:
        if not self.episode_pnls:
            return 0.0
        pnls = np.array(self.episode_pnls)
        pos = pnls[pnls > 0].sum()
        neg = abs(pnls[pnls < 0].sum())
        if neg < 1e-9:
            return float("inf") if pos > 0 else 0.0
        return pos / neg

    def get_summary(self) -> dict:
        if not self.episode_rewards:
            return {}

        return {
            "episodes": self.total_episodes,
            "avg_reward": np.mean(self.episode_rewards),
            "avg_length": np.mean(self.episode_lengths),
            "win_rate": np.mean(self.win_history),
            "sharpe": self._calc_sharpe(),
            "profit_factor": self._calc_profit_factor(),
            "max_drawdown": max(self.drawdowns) if self.drawdowns else 0,
        }


def generate_synthetic_data(assets: list) -> dict:
    data_dict = {}
    dates = pd.date_range(start="2016-01-01", periods=5000, freq="5min")
    for asset in assets:
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
    logger.info("Generated synthetic data for %d assets", len(assets))
    return data_dict


def train():
    logger.info("=" * 60)
    logger.info("SAC Training Started")
    logger.info("=" * 60)

    loader = DataLoader()
    if not os.path.exists(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
        logger.warning("Data directory not found, generating synthetic data")
        data_dict = generate_synthetic_data(config.ASSETS)
    else:
        data_dict = loader.load_all_data()
        logger.info("Loaded data from %s", config.DATA_DIR)

    if not data_dict:
        logger.warning("No data found, generating synthetic data")
        data_dict = generate_synthetic_data(config.ASSETS)

    aligned_df = loader.align_data(data_dict)
    logger.info("Aligned data shape: %s", aligned_df.shape)

    fe = FeatureEngine()
    feature_df = fe.calculate_features(aligned_df)
    logger.info("Features calculated: %d columns", len(feature_df.columns))

    labeler = StructuralLabeler(reversal_multiplier=config.ATR_REVERSAL_MULTIPLIER)
    for asset in config.ASSETS:
        aligned_df = labeler.label_data(aligned_df, asset)
    logger.info("Structural labels applied")

    final_df = pd.concat([aligned_df, feature_df], axis=1)
    final_df = final_df.iloc[-5000:]
    logger.info("Final dataset: %d rows", len(final_df))

    num_envs = config.NUM_ENVS
    logger.info("Initializing %d parallel environments", num_envs)
    envs = [TradingEnv(final_df.copy(), asset="EURUSD") for _ in range(num_envs)]
    states = [env.reset()[0] for env in envs]

    agent = SACAgent()
    memory = ReplayBuffer(config.BUFFER_SIZE, config.STATE_DIM, config.ACTION_DIM)
    metrics_logger = MetricsLogger()

    logger.info("Configuration:")
    logger.info("  Total steps: %d", config.TOTAL_STEPS)
    logger.info("  Batch size: %d", config.BATCH_SIZE)
    logger.info("  Buffer size: %d", config.BUFFER_SIZE)
    logger.info("  Updates per step: %d", config.UPDATES_PER_STEP)
    logger.info("  Learning rate: %.2e", config.LR)
    logger.info("  Gamma: %.4f", config.GAMMA)
    logger.info("=" * 60)

    total_steps = 0
    updates = 0
    log_interval = config.LOG_INTERVAL
    start_time = datetime.now()
    last_metrics = {}

    while total_steps < config.TOTAL_STEPS:
        states_ready = np.array(states)
        actions = agent.select_action(states_ready)

        next_states = []
        rewards = []
        masks = []
        infos = []
        terminated_mask = []

        for i, env in enumerate(envs):
            next_state, reward, terminated, truncated, info = env.step(actions[i])
            next_states.append(next_state)
            rewards.append(reward)
            masks.append(0 if terminated else 1)
            infos.append(info)
            terminated_mask.append(terminated or truncated)

        for i in range(len(envs)):
            memory.push(states[i], actions[i], rewards[i], next_states[i], masks[i])

            if terminated_mask[i]:
                episode_stats = metrics_logger.log_episode(
                    total_steps,
                    rewards[i],
                    1,
                    infos[i]["equity"] - config.INITIAL_EQUITY,
                    infos[i],
                )
                state, _ = envs[i].reset()
                states[i] = state

                if metrics_logger.total_episodes % 100 == 0:
                    logger.info(
                        "Episode %d | Steps: %d | Reward: %.2f | Win Rate: %.2f%% | Profit Factor: %.2f | Sharpe: %.2f",
                        metrics_logger.total_episodes,
                        total_steps,
                        rewards[i],
                        episode_stats["win_rate"] * 100,
                        episode_stats["profit_factor"],
                        episode_stats["sharpe"],
                    )
            else:
                states[i] = next_states[i]

        total_steps += num_envs

        if len(memory) > config.BATCH_SIZE:
            for _ in range(config.UPDATES_PER_STEP):
                last_metrics = agent.update_parameters(
                    memory, config.BATCH_SIZE, updates
                )
                metrics_logger.log_training_step(total_steps, last_metrics)
                updates += 1

        if total_steps % log_interval == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            summary = metrics_logger.get_summary()

            logger.info("-" * 60)
            logger.info(
                "Step: %d / %d (%.1f%%)",
                total_steps,
                config.TOTAL_STEPS,
                total_steps / config.TOTAL_STEPS * 100,
            )
            logger.info(
                "Speed: %.0f steps/sec | Elapsed: %.1f min", steps_per_sec, elapsed / 60
            )
            if summary:
                logger.info(
                    "Episodes: %d | Win Rate: %.2f%% | Profit Factor: %.2f | Sharpe: %.2f | Max DD: %.2f%%",
                    summary["episodes"],
                    summary["win_rate"] * 100,
                    summary["profit_factor"],
                    summary["sharpe"],
                    summary["max_drawdown"] * 100,
                )
            if last_metrics:
                logger.info(
                    "Q1 Loss: %.4f | Q2 Loss: %.4f | Actor Loss: %.4f",
                    last_metrics.get("q1_loss", 0),
                    last_metrics.get("q2_loss", 0),
                    last_metrics.get("actor_loss", 0),
                )
            logger.info("-" * 60)

    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(
        "Total steps: %d | Total episodes: %d",
        total_steps,
        metrics_logger.total_episodes,
    )

    final_summary = metrics_logger.get_summary()
    logger.info(
        "Final Win Rate: %.2f%% | Final Profit Factor: %.2f | Final Sharpe: %.2f | Final Max DD: %.2f%%",
        final_summary["win_rate"] * 100,
        final_summary["profit_factor"],
        final_summary["sharpe"],
        final_summary["max_drawdown"] * 100,
    )
    logger.info("Logs saved to: %s", metrics_logger.log_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    train()

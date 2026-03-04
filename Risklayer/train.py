import numpy as np
import pandas as pd
import torch
import logging
import os
import csv
import time
from datetime import datetime
from collections import deque
from Risklayer.config import config
from Risklayer.data_loader import DataLoader
from Risklayer.feature_engineering import FeatureEngine
from Risklayer.sac_agent import SACAgent
from Risklayer.generate_alpha_signals import generate_signals
from Risklayer.vectorized_env import VectorizedTradingEnv

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

    def push_batch(self, states, actions, rewards, next_states, masks):
        num = states.shape[0]
        if self.ptr + num <= self.capacity:
            self.state_buf[self.ptr : self.ptr + num] = states
            self.action_buf[self.ptr : self.ptr + num] = actions
            self.reward_buf[self.ptr : self.ptr + num, 0] = rewards
            self.next_state_buf[self.ptr : self.ptr + num] = next_states
            self.mask_buf[self.ptr : self.ptr + num, 0] = masks
            self.ptr = (self.ptr + num) % self.capacity
        else:
            # Wrap around logic (split into two parts)
            part1 = self.capacity - self.ptr
            self.state_buf[self.ptr : self.capacity] = states[:part1]
            self.action_buf[self.ptr : self.capacity] = actions[:part1]
            self.reward_buf[self.ptr : self.capacity, 0] = rewards[:part1]
            self.next_state_buf[self.ptr : self.capacity] = next_states[:part1]
            self.mask_buf[self.ptr : self.capacity, 0] = masks[:part1]

            part2 = num - part1
            self.state_buf[0 : part2] = states[part1:]
            self.action_buf[0 : part2] = actions[part1:]
            self.reward_buf[0 : part2, 0] = rewards[part1:]
            self.next_state_buf[0 : part2] = next_states[part1:]
            self.mask_buf[0 : part2, 0] = masks[part1:]
            self.ptr = part2
        self.size = min(self.size + num, self.capacity)

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
        self.episode_pnls = deque(maxlen=window_size)
        self.win_history = deque(maxlen=window_size)
        self.drawdowns = deque(maxlen=window_size)
        self.total_episodes = 0
        self.total_trades = 0
        self.wins = 0

    def _init_csv_files(self):
        with open(self.metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "q1_loss", "q2_loss", "actor_loss", "alpha", "sharpe"])

    def log_training_step(self, step: int, metrics: dict, sharpe: float):
        if step % config.METRICS_LOG_INTERVAL != 0: return
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, f"{metrics.get('q1_loss', 0):.6f}", f"{metrics.get('q2_loss', 0):.6f}", f"{metrics.get('actor_loss', 0):.6f}", f"{metrics.get('alpha', 0):.4f}", f"{sharpe:.4f}"])

    def log_episode(self, reward: float, equity_delta: float, drawdown: float):
        self.episode_rewards.append(reward)
        self.episode_pnls.append(equity_delta)
        self.win_history.append(1 if equity_delta > 0 else 0)
        self.drawdowns.append(drawdown)
        self.total_episodes += 1
        if equity_delta > 0: self.wins += 1

    def _calc_sharpe(self) -> float:
        if len(self.episode_pnls) < 2: return 0.0
        returns = np.array(self.episode_pnls)
        avg = np.mean(returns)
        std = np.std(returns)
        if std < 1e-9: return 0.0
        return (avg / std) * np.sqrt(252)

    def get_summary(self) -> dict:
        if not self.episode_rewards: return {"episodes": 0, "avg_reward": 0.0, "win_rate": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
        return {"episodes": self.total_episodes, "avg_reward": np.mean(self.episode_rewards), "win_rate": np.mean(self.win_history), "sharpe": self._calc_sharpe(), "max_drawdown": max(self.drawdowns) if self.drawdowns else 0}

def prepare_optimized_data(full_df, signals_df):
    logger.info("Preparing optimized Numpy data structures...")
    price_data = {}
    assets = config.ASSETS
    for asset in assets:
        atr_vals = full_df[f"{asset}_atr_14"] if f"{asset}_atr_14" in full_df else full_df.get(f"{asset}_atr", 0.0001)
        vol_vals = full_df[f"{asset}_vol_percentile"] if f"{asset}_vol_percentile" in full_df else 0.5 * np.ones(len(full_df))
        peak_vals = full_df[f"{asset}_peak_dist"] if f"{asset}_peak_dist" in full_df else np.zeros(len(full_df))
        valley_vals = full_df[f"{asset}_valley_dist"] if f"{asset}_valley_dist" in full_df else np.zeros(len(full_df))

        price_data[asset] = {
            'high': full_df[f"{asset}_high"].values.astype(np.float32),
            'low': full_df[f"{asset}_low"].values.astype(np.float32),
            'close': full_df[f"{asset}_close"].values.astype(np.float32),
            'atr': atr_vals.values.astype(np.float32) if hasattr(atr_vals, 'values') else atr_vals.astype(np.float32),
            'vol': vol_vals.values.astype(np.float32) if hasattr(vol_vals, 'values') else vol_vals.astype(np.float32),
            'peak': peak_vals.values.astype(np.float32) if hasattr(peak_vals, 'values') else peak_vals.astype(np.float32),
            'valley': valley_vals.values.astype(np.float32) if hasattr(valley_vals, 'values') else valley_vals.astype(np.float32)
        }
    fe = FeatureEngine()
    # Only keep signals that are present in full_df
    valid_signals_mask = signals_df.index.isin(full_df.index)
    valid_signals = signals_df[valid_signals_mask]

    signal_assets = valid_signals['asset_name'].values
    signal_indices = np.array([full_df.index.get_loc(idx) for idx in valid_signals.index])
    signal_obs_static = np.zeros((len(valid_signals), 32), dtype=np.float32)
    for asset in assets:
        asset_mask = (signal_assets == asset)
        if not asset_mask.any(): continue
        asset_signal_indices = signal_indices[asset_mask]
        cols = fe.get_observation_cols(asset)
        existing_cols = [c for c in cols if c in full_df.columns]
        signal_obs_static[asset_mask, :len(existing_cols)] = full_df[existing_cols].values[asset_signal_indices].astype(np.float32)
        signal_obs_static[asset_mask, 30] = price_data[asset]['atr'][asset_signal_indices]
        signal_obs_static[asset_mask, 31] = price_data[asset]['vol'][asset_signal_indices]
    signal_data = {'assets': signal_assets, 'indices': signal_indices, 'obs_static': signal_obs_static, 'meta': valid_signals['meta_score'].values.astype(np.float32), 'qual': valid_signals['quality_score'].values.astype(np.float32), 'dir': valid_signals['pred_direction'].values.astype(np.int8)}
    return price_data, signal_data

def train():
    logger.info("=" * 60)
    logger.info("SAC Training Started (Vectorized)")
    logger.info("=" * 60)
    signals_path = os.path.join("data", "alpha_signals_2016_2025.parquet")
    if not os.path.exists(signals_path): generate_signals()
    signals_df = pd.read_parquet(signals_path)
    loader = DataLoader()
    full_df = loader.align_data(loader.load_all_data())
    price_data, signal_data = prepare_optimized_data(full_df, signals_df)
    
    num_envs = config.NUM_ENVS
    v_env = VectorizedTradingEnv(num_envs, price_data, signal_data)
    agent = SACAgent()
    memory = ReplayBuffer(config.BUFFER_SIZE, config.STATE_DIM, config.ACTION_DIM)
    metrics_logger = MetricsLogger()

    total_steps, updates, start_time = 0, 0, time.time()
    last_metrics = {}

    while total_steps < config.TOTAL_STEPS:
        states = v_env.get_observations()
        actions = agent.select_action(states)
        next_states, rewards, terminated, old_states, pnls, current_dds = v_env.step(actions)

        memory.push_batch(old_states, actions, rewards, next_states, 1 - terminated.astype(np.float32))
        for i in range(num_envs):
            metrics_logger.log_episode(rewards[i], pnls[i], current_dds[i])

        total_steps += num_envs
        if len(memory) > config.BATCH_SIZE:
            for _ in range(config.UPDATES_PER_STEP):
                last_metrics = agent.update_parameters(memory, config.BATCH_SIZE, updates)
                updates += 1

        if total_steps % config.LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed if elapsed > 0 else 0
            summary = metrics_logger.get_summary()
            logger.info(f"Step: {total_steps} | Speed: {steps_per_sec:.0f} steps/s | Win Rate: {summary['win_rate']*100:.1f}% | Sharpe: {summary['sharpe']:.2f}")
            metrics_logger.log_training_step(total_steps, last_metrics, summary['sharpe'])

    logger.info("=" * 60)
    logger.info(f"Training Complete. Total Time: {(time.time() - start_time)/60:.1f} min")
    logger.info("=" * 60)

if __name__ == "__main__":
    train()

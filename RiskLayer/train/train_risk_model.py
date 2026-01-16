import yaml
import argparse
import os
import sys
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure project root is in path
sys.path.append(os.getcwd())

from RiskLayer.env.risk_env import RiskTradingEnv


class SaveNormalizerCallback(BaseCallback):
    """Callback to save VecNormalize stats periodically."""
    def __init__(self, save_path, save_freq=50000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            norm_path = os.path.join(self.save_path, f"vec_normalize_{self.n_calls}.pkl")
            self.training_env.save(norm_path)
            if self.verbose > 0:
                logging.info(f"Saved normalizer to {norm_path}")
        return True


class BestPerformanceTracker(BaseCallback):
    """Callback to track and save best performance metrics throughout training."""
    def __init__(self, log_dir, check_freq=5000, verbose=0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.check_freq = check_freq
        self.best_performance_file = self.log_dir / "best_performance.json"
        self.initial_equity = 10000.0
        
        # Initialize best metrics
        self.best_metrics = {
            "best_episode_reward": float('-inf'),
            "best_mean_reward": float('-inf'),
            "best_win_rate": 0.0,
            "best_equity": 0.0,
            "best_avg_reward": float('-inf'),
            "best_total_pnl": float('-inf'),
            "best_sharpe_ratio": float('-inf'),
            "best_return_pct": float('-inf'),
            "best_step": 0,
            "best_timestamp": None,
            "training_start": datetime.now().isoformat(),
            "all_best_updates": []  # Track all improvements
        }
        
        # Load existing best performance if it exists
        if self.best_performance_file.exists():
            try:
                with open(self.best_performance_file, 'r') as f:
                    existing = json.load(f)
                    self.best_metrics.update(existing)
                    logging.info(f"Loaded existing best performance from {self.best_performance_file}")
            except Exception as e:
                logging.warning(f"Could not load existing best performance: {e}")
        
        # Track reward log files
        self.reward_log_dir = Path("RiskLayer/logs")
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._update_best_performance()
            self._log_current_status()
        return True
    
    def _log_current_status(self):
        """Log current portfolio status for real-time monitoring."""
        if self.reward_log_dir.exists():
            reward_files = list(self.reward_log_dir.glob("rewards_env_*.csv"))
            if reward_files:
                try:
                    all_data = []
                    for log_file in reward_files:
                        try:
                            df = pd.read_csv(log_file)
                            if len(df) > 0:
                                all_data.append(df)
                        except Exception:
                            pass
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        combined_df = combined_df.sort_values('step').reset_index(drop=True)
                        latest = combined_df.iloc[-1] if len(combined_df) > 0 else None
                        
                        if latest is not None and 'total_pnl' in combined_df.columns:
                            cumulative_pnl = combined_df['total_pnl'].sum()
                            equity = self.initial_equity + float(cumulative_pnl)
                            return_pct = ((equity - self.initial_equity) / self.initial_equity) * 100.0
                            
                            # Build equity curve for Sharpe
                            equity_curve = [self.initial_equity]
                            for pnl in combined_df['total_pnl'].cumsum():
                                equity_curve.append(self.initial_equity + pnl)
                            sharpe = self._calculate_sharpe_ratio(np.array(equity_curve))
                            
                            win_rate = latest.get('win_rate', 0.0) if pd.notna(latest.get('win_rate', 0.0)) else 0.0
                            
                            # Log current status
                            logging.info(
                                f"📊 Portfolio Status (Step {self.num_timesteps:,}): "
                                f"Equity=${equity:.2f} | Return={return_pct:.2f}% | "
                                f"WinRate={win_rate:.4f} | Sharpe={sharpe:.3f} | "
                                f"Best: WR={self.best_metrics['best_win_rate']:.4f} | "
                                f"SR={self.best_metrics['best_sharpe_ratio']:.3f} | "
                                f"Ret={self.best_metrics['best_return_pct']:.2f}%"
                            )
                except Exception as e:
                    logging.debug(f"Error logging current status: {e}")
    
    def _calculate_sharpe_ratio(self, equity_curve):
        """Calculate Sharpe ratio from equity curve."""
        if len(equity_curve) < 2:
            return 0.0
        
        try:
            # Filter out zero/negative equity values
            equity_curve = np.array(equity_curve)
            equity_curve = equity_curve[equity_curve > 0]
            
            if len(equity_curve) < 2:
                return 0.0
            
            # Calculate period returns
            returns = np.diff(equity_curve) / equity_curve[:-1]
            
            # Filter out invalid returns
            returns = returns[np.isfinite(returns)]
            
            if len(returns) == 0:
                return 0.0
            
            std_returns = np.std(returns)
            if std_returns == 0 or not np.isfinite(std_returns):
                return 0.0
            
            # For 5-minute candles: 288 periods per day (24*60/5), 252 trading days per year
            periods_per_year = 252 * 288
            
            # Annualized Sharpe ratio
            mean_return = np.mean(returns)
            if not np.isfinite(mean_return):
                return 0.0
            
            sharpe = mean_return / std_returns * np.sqrt(periods_per_year)
            
            return float(sharpe) if np.isfinite(sharpe) else 0.0
        except Exception:
            return 0.0
    
    def _update_best_performance(self):
        """Update best performance metrics from training stats and reward logs."""
        updated = False
        
        # Get training stats from episode info buffer (if available)
        if hasattr(self, 'model') and hasattr(self.model, 'ep_info_buffer'):
            if len(self.model.ep_info_buffer) > 0:
                # Get latest episode info
                ep_info = self.model.ep_info_buffer[-1]
                ep_reward = ep_info.get('r', 0)
                
                # Track episode reward
                if ep_reward > self.best_metrics['best_episode_reward']:
                    self.best_metrics['best_episode_reward'] = ep_reward
                    updated = True
                
                # Calculate mean reward from buffer
                all_rewards = [ep.get('r', 0) for ep in self.model.ep_info_buffer]
                if all_rewards:
                    mean_reward = sum(all_rewards) / len(all_rewards)
                    if mean_reward > self.best_metrics['best_mean_reward']:
                        self.best_metrics['best_mean_reward'] = mean_reward
                        updated = True
        
        # Read reward log files from environment
        if self.reward_log_dir.exists():
            reward_files = list(self.reward_log_dir.glob("rewards_env_*.csv"))
            if reward_files:
                try:
                    # Read the most recent entries from all reward log files
                    all_data = []
                    for log_file in reward_files:
                        try:
                            df = pd.read_csv(log_file)
                            if len(df) > 0:
                                all_data.append(df)
                        except Exception as e:
                            logging.debug(f"Could not read {log_file}: {e}")
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        # Sort by step to ensure chronological order
                        combined_df = combined_df.sort_values('step').reset_index(drop=True)
                        latest = combined_df.iloc[-1] if len(combined_df) > 0 else None
                        
                        if latest is not None:
                            # Update metrics from reward logs
                            if 'win_rate' in latest and pd.notna(latest['win_rate']) and latest['win_rate'] > self.best_metrics['best_win_rate']:
                                old_win_rate = self.best_metrics['best_win_rate']
                                self.best_metrics['best_win_rate'] = float(latest['win_rate'])
                                logging.info(f"🏆 NEW BEST WIN RATE: {self.best_metrics['best_win_rate']:.4f} (was {old_win_rate:.4f}) at step {self.num_timesteps:,}")
                                updated = True
                            
                            if 'avg_reward' in latest and pd.notna(latest['avg_reward']) and latest['avg_reward'] > self.best_metrics['best_avg_reward']:
                                self.best_metrics['best_avg_reward'] = float(latest['avg_reward'])
                                updated = True
                            
                            # Note: 'total_pnl' column actually contains period_pnl (R-multiple sum per period)
                            # Calculate cumulative PnL by summing all period values
                            if 'total_pnl' in combined_df.columns:
                                cumulative_pnl = combined_df['total_pnl'].sum()
                                if cumulative_pnl > self.best_metrics['best_total_pnl']:
                                    self.best_metrics['best_total_pnl'] = float(cumulative_pnl)
                                    updated = True
                                
                                # Calculate equity from cumulative PnL (assuming starting equity of 10000)
                                equity = self.initial_equity + float(cumulative_pnl)
                                if equity > self.best_metrics['best_equity']:
                                    old_equity = self.best_metrics['best_equity']
                                    self.best_metrics['best_equity'] = equity
                                    logging.info(f"💰 NEW BEST EQUITY: ${equity:.2f} (was ${old_equity:.2f}) at step {self.num_timesteps:,}")
                                    updated = True
                                
                                # Calculate return percentage
                                return_pct = ((equity - self.initial_equity) / self.initial_equity) * 100.0
                                if return_pct > self.best_metrics['best_return_pct']:
                                    old_return = self.best_metrics['best_return_pct']
                                    self.best_metrics['best_return_pct'] = return_pct
                                    logging.info(f"📈 NEW BEST RETURN: {return_pct:.2f}% (was {old_return:.2f}%) at step {self.num_timesteps:,}")
                                    updated = True
                                
                                # Calculate Sharpe ratio from equity curve
                                # Build equity curve from cumulative PnL
                                equity_curve = [self.initial_equity]
                                for pnl in combined_df['total_pnl'].cumsum():
                                    equity_curve.append(self.initial_equity + pnl)
                                
                                sharpe_ratio = self._calculate_sharpe_ratio(np.array(equity_curve))
                                if sharpe_ratio > self.best_metrics['best_sharpe_ratio']:
                                    old_sharpe = self.best_metrics['best_sharpe_ratio']
                                    self.best_metrics['best_sharpe_ratio'] = sharpe_ratio
                                    logging.info(f"⭐ NEW BEST SHARPE RATIO: {sharpe_ratio:.3f} (was {old_sharpe:.3f}) at step {self.num_timesteps:,}")
                                    updated = True
                except Exception as e:
                    logging.debug(f"Error reading reward logs: {e}")
        
        # Save if updated
        if updated:
            self.best_metrics['best_step'] = self.num_timesteps
            self.best_metrics['best_timestamp'] = datetime.now().isoformat()
            
            # Record this update
            update_record = {
                "step": self.num_timesteps,
                "timestamp": self.best_metrics['best_timestamp'],
                "metrics": {k: v for k, v in self.best_metrics.items() 
                           if k not in ['all_best_updates', 'training_start']}
            }
            self.best_metrics['all_best_updates'].append(update_record)
            
            # Save to file
            try:
                with open(self.best_performance_file, 'w') as f:
                    json.dump(self.best_metrics, f, indent=2)
                if self.verbose > 0:
                    logging.info(f"Updated best performance at step {self.num_timesteps:,}")
            except Exception as e:
                logging.error(f"Failed to save best performance: {e}")
    
    def get_summary(self):
        """Get a summary of best performance metrics."""
        return {
            "best_mean_reward": self.best_metrics.get('best_mean_reward', 'N/A'),
            "best_episode_reward": self.best_metrics.get('best_episode_reward', 'N/A'),
            "best_win_rate": self.best_metrics.get('best_win_rate', 'N/A'),
            "best_equity": self.best_metrics.get('best_equity', 'N/A'),
            "best_avg_reward": self.best_metrics.get('best_avg_reward', 'N/A'),
            "best_total_pnl": self.best_metrics.get('best_total_pnl', 'N/A'),
            "best_sharpe_ratio": self.best_metrics.get('best_sharpe_ratio', 'N/A'),
            "best_return_pct": self.best_metrics.get('best_return_pct', 'N/A'),
            "best_step": self.best_metrics.get('best_step', 'N/A'),
            "best_timestamp": self.best_metrics.get('best_timestamp', 'N/A')
        }


def load_config(config_path="RiskLayer/config/ppo_config.yaml"):
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Map activation function string to torch class
    if 'policy_kwargs' in config and 'activation_fn' in config['policy_kwargs']:
        act_fn_str = config['policy_kwargs']['activation_fn']
        if act_fn_str == "Tanh":
            config['policy_kwargs']['activation_fn'] = nn.Tanh
        elif act_fn_str == "ReLU":
            config['policy_kwargs']['activation_fn'] = nn.ReLU
        elif act_fn_str == "LeakyReLU":
            config['policy_kwargs']['activation_fn'] = nn.LeakyReLU
            
    return config


def train(dry_run=False, steps_override=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config = load_config()
    
    # Overrides
    total_timesteps = steps_override if steps_override else config.get('total_timesteps', 10_000_000)
    
    models_dir = "RiskLayer/models/checkpoints"
    log_dir = "RiskLayer/models/logs"
    reward_log_dir = "RiskLayer/logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(reward_log_dir, exist_ok=True)

    logging.info(f"Starting Training Session. Dry Run: {dry_run}")
    logging.info(f"Total Timesteps: {total_timesteps:,}")
    logging.info(f"Model checkpoints: {os.path.abspath(models_dir)}")
    logging.info(f"Training logs: {os.path.abspath(log_dir)}")
    logging.info(f"Reward logs: {os.path.abspath(reward_log_dir)}")
    
    # 1. Initialize Environment
    # Use 5 parallel envs (one per asset for coverage)
    n_envs = 1 if dry_run else 5
    
    def make_env():
        return RiskTradingEnv(is_training=True)

    if dry_run:
        env = DummyVecEnv([make_env])
    else:
        env = DummyVecEnv([make_env for _ in range(n_envs)])
    
    # Wrap with VecNormalize for observation normalization
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    logging.info("Environment wrapped with VecNormalize")

    # 2. Initialize Model with config hyperparameters
    policy_kwargs = config.get('policy_kwargs', None)
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=config.get('learning_rate', 0.0001),
        n_steps=config.get('n_steps', 4096),
        batch_size=config.get('batch_size', 256),
        n_epochs=config.get('n_epochs', 10),
        gamma=config.get('gamma', 0.995),
        gae_lambda=config.get('gae_lambda', 0.98),
        clip_range=config.get('clip_range', 0.15),
        ent_coef=config.get('ent_coef', 0.005),
        max_grad_norm=config.get('max_grad_norm', 0.5),
        policy_kwargs=policy_kwargs
    )
    
    logging.info(f"Model initialized with policy: {model.policy}")
    
    if dry_run:
        logging.info("Dry Run: Executing a few steps to verify loop...")
        try:
            model.learn(total_timesteps=2000)
            logging.info("Dry Run Complete. System appears functional.")
        except Exception as e:
            logging.error(f"Dry Run Failed: {e}")
            raise e
        finally:
            env.close()
        return

    # 3. Training Loop
    logging.info(f"Starting Full Training for {total_timesteps:,} steps...")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save model every 100k steps
        save_path=models_dir,
        name_prefix="risk_model_ppo"
    )
    
    normalizer_callback = SaveNormalizerCallback(
        save_path=models_dir,
        save_freq=100000,  # Save normalizer every 100k steps
        verbose=1
    )
    
    # Performance tracking callback
    performance_tracker = BestPerformanceTracker(
        log_dir=log_dir,
        check_freq=10000,  # Check every 10k steps
        verbose=1
    )
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=[checkpoint_callback, normalizer_callback, performance_tracker],
            progress_bar=True
        )
        
        # Save final model and normalizer
        model.save(f"{models_dir}/risk_model_final")
        env.save(f"{models_dir}/vec_normalize.pkl")
        logging.info(f"Training Complete. Model saved to {models_dir}/risk_model_final.zip")
        logging.info(f"Normalizer saved to {models_dir}/vec_normalize.pkl")
        
        # Final performance update
        performance_tracker._update_best_performance()
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving current state...")
        model.save(f"{models_dir}/risk_model_interrupted")
        env.save(f"{models_dir}/vec_normalize_interrupted.pkl")
        performance_tracker._update_best_performance()
    finally:
        # Print summary and log locations
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        summary = performance_tracker.get_summary()
        print(f"\nBest Performance Metrics:")
        print(f"  🏆 Best Win Rate: {summary['best_win_rate']:.4f}" if isinstance(summary['best_win_rate'], (int, float)) else f"  Best Win Rate: {summary['best_win_rate']}")
        print(f"  ⭐ Best Sharpe Ratio: {summary['best_sharpe_ratio']:.3f}" if isinstance(summary['best_sharpe_ratio'], (int, float)) else f"  Best Sharpe Ratio: {summary['best_sharpe_ratio']}")
        print(f"  📈 Best Return: {summary['best_return_pct']:.2f}%" if isinstance(summary['best_return_pct'], (int, float)) else f"  Best Return: {summary['best_return_pct']}")
        print(f"  💰 Best Equity: ${summary['best_equity']:.2f}" if isinstance(summary['best_equity'], (int, float)) else f"  Best Equity: {summary['best_equity']}")
        print(f"  Best Mean Reward: {summary['best_mean_reward']}")
        print(f"  Best Episode Reward: {summary['best_episode_reward']}")
        print(f"  Best Avg Reward: {summary['best_avg_reward']}")
        print(f"  Best Total PnL: {summary['best_total_pnl']}")
        print(f"  Achieved at Step: {summary['best_step']:,}")
        print(f"  Timestamp: {summary['best_timestamp']}")
        
        print(f"\nLog File Locations:")
        print(f"  Best Performance: {os.path.abspath(performance_tracker.best_performance_file)}")
        print(f"  Model Checkpoints: {os.path.abspath(models_dir)}")
        print(f"  Training Logs (TensorBoard): {os.path.abspath(log_dir)}")
        print(f"  Reward Logs (CSV): {os.path.abspath(reward_log_dir)}")
        print(f"  Final Model: {os.path.abspath(models_dir)}/risk_model_final.zip")
        print(f"  Normalizer: {os.path.abspath(models_dir)}/vec_normalize.pkl")
        print("="*80 + "\n")
        
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Risk Model (TradeGuard)")
    parser.add_argument("--dry-run", action="store_true", help="Run a short verification test instead of full training")
    parser.add_argument("--steps", type=int, default=0, help="Total training timesteps (overrides config)")
    
    args = parser.parse_args()
    
    steps = args.steps if args.steps > 0 else None
    train(dry_run=args.dry_run, steps_override=steps)

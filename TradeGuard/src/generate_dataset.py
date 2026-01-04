
import os
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
import sys
import gc
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from Alpha.src.trading_env import TradingEnv
from TradeGuard.src.feature_calculator import TradeGuardFeatureCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingDatasetGenerator:
    def __init__(self, model_path, data_dir='TradeGuard/data'):
        self.data_dir = data_dir
        self.model_path = model_path
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Load all data once to slice later
        self.full_data = self._load_all_data()
        
    def _load_all_data(self):
        logger.info("Loading all market data files...")
        data = {}
        common_index = None
        
        for asset in self.assets:
            file_path = os.path.join(self.data_dir, f"{asset}_5m.parquet")
            # Handle potential missing files gracefully or ensure they exist
            if not os.path.exists(file_path):
                 # Fallback to 2025 file if exists (matching TradingEnv logic)
                 file_path_2025 = os.path.join(self.data_dir, f"{asset}_5m_2025.parquet")
                 if os.path.exists(file_path_2025):
                     file_path = file_path_2025
                 else:
                     raise FileNotFoundError(f"Data file for {asset} not found at {file_path}")

            df = pd.read_parquet(file_path)
            # Ensure index is datetime and sorted
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            data[asset] = df
            
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        # Align all assets to the common intersection
        if common_index is None or len(common_index) == 0:
            raise ValueError("No overlapping data found across assets! Cannot generate dataset.")
            
        logger.info(f"Aligning all assets to common time range. Overlapping rows: {len(common_index)}")
        logger.info(f"Data Range: {common_index.min()} to {common_index.max()}")
        
        for asset in self.assets:
            data[asset] = data[asset].loc[common_index]
            
        return data

    def generate(self, output_file='TradeGuard/data/training_dataset.parquet', chunk_size=50000):
        # Load Risk Model and Normalizer
        risk_model_path = "models/checkpoints/risk/model10M.zip"
        risk_norm_path = "models/checkpoints/risk/model10M.pkl"
        
        logger.info(f"Loading Risk Model from {risk_model_path}...")
        try:
            risk_model = PPO.load(risk_model_path, device='cpu')
            
            # Create a dummy env with 45 dims to load the Risk normalizer
            class SimpleRiskEnv(gym.Env):
                def __init__(self):
                    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,))
                    self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
                def reset(self, seed=None): return np.zeros(45), {}
                def step(self, action): return np.zeros(45), 0, False, False, {}

            self.risk_norm_env = VecNormalize.load(risk_norm_path, DummyVecEnv([lambda: SimpleRiskEnv()]))
            self.risk_norm_env.training = False
            self.risk_norm_env.norm_reward = False
            logger.info(f"Loaded Risk Normalizer from {risk_norm_path}")
            
        except Exception as e:
            logger.error(f"Failed to load Risk Model/Normalizer: {e}")
            return

        # Load Alpha model and Normalizer ONCE
        alpha_norm_path = "models/checkpoints/alpha/ppo_final_vecnormalize.pkl"
        dummy_data = {asset: self.full_data[asset].iloc[:100] for asset in self.assets}
        temp_env = DummyVecEnv([lambda: TradingEnv(data=dummy_data, is_training=False, stage=1)])
        
        if os.path.exists(alpha_norm_path):
            alpha_vec_normalize = VecNormalize.load(alpha_norm_path, temp_env)
            alpha_vec_normalize.training = False
            alpha_vec_normalize.norm_reward = False
            logger.info(f"Loaded Alpha Normalizer from {alpha_norm_path}")
        else:
            alpha_vec_normalize = temp_env
        
        alpha_model = PPO.load(self.model_path, env=alpha_vec_normalize)
        logger.info("Alpha model and normalizer loaded.")
        
        # Diagnostic: Check observation space compatibility
        logger.info(f"Alpha model observation space: {alpha_model.observation_space}")
        logger.info(f"Alpha normalizer observation space: {alpha_vec_normalize.observation_space}")

        min_len = min(len(df) for df in self.full_data.values())
        total_rows = min_len
        logger.info(f"Total rows: {total_rows}, Chunk size: {chunk_size}")
        
        dataset = []
        action_history = {asset: [] for asset in self.assets}
        stats = {'total_steps': 0, 'alpha_signals': 0, 'risk_blocked_bad_rr': 0, 'saved_rows': 0}
        
        warmup = 500 
        
        with tqdm(total=total_rows, desc="Generating Dataset", unit="rows") as pbar:
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                slice_start = max(0, start_idx - warmup)
                
                if slice_start >= end_idx: continue

                sliced_data = {asset: self.full_data[asset].iloc[slice_start:end_idx] for asset in self.assets}
                if any(df.empty for df in sliced_data.values()): continue

                raw_env = TradingEnv(data=sliced_data, is_training=False, stage=1)
                raw_env.reset() # Critical: Initialize state variables like self.positions
                tg_calculator = TradeGuardFeatureCalculator(raw_env.data)
                
                steps_to_skip = start_idx - slice_start
                
                for step in range(len(raw_env.processed_data)):
                    raw_env.current_step = step
                    
                    if step < steps_to_skip:
                        # Update action history even during warmup
                        for asset in self.assets:
                            try:
                                raw_env.current_asset = asset
                                obs = raw_env._get_observation()
                                norm_obs = alpha_vec_normalize.normalize_obs(obs)
                                action, _ = alpha_model.predict(norm_obs, deterministic=True)
                                direction = 1 if action[0] > 0.33 else (-1 if action[0] < -0.33 else 0)
                                action_history[asset].append(direction)
                                if len(action_history[asset]) > 20: action_history[asset].pop(0)
                            except Exception as e:
                                logger.error(f"Error during warmup at step {step}, asset {asset}: {e}")
                                raise
                        continue

                    trade_infos = {}
                    for asset in self.assets:
                        raw_env.current_asset = asset
                        obs = raw_env._get_observation()
                        norm_obs = alpha_vec_normalize.normalize_obs(obs)
                        action, _ = alpha_model.predict(norm_obs, deterministic=True)
                        direction = 1 if action[0] > 0.33 else (-1 if action[0] < -0.33 else 0)
                        
                        prev_act = action_history[asset][-1] if action_history[asset] else 0
                        
                        # LINKAGE FIX: Only allow entry if the signal is NEW (avoid duplicates)
                        if direction != 0 and direction != prev_act:
                            stats['alpha_signals'] += 1
                            
                            # Construct 45-dim observation for Risk Model
                            # Alpha features (40) + Account state (5)
                            drawdown = 1.0 - (raw_env.equity / raw_env.peak_equity)
                            account_obs = np.array([1.0, drawdown, 0.0, 1.0, 0.0], dtype=np.float32)
                            risk_obs = np.concatenate([obs, account_obs])
                            
                            norm_risk_obs = self.risk_norm_env.normalize_obs(risk_obs)
                            risk_action, _ = risk_model.predict(norm_risk_obs, deterministic=True)
                            if isinstance(risk_action, np.ndarray): risk_action = risk_action[0]
                            
                            # Risk model now has 2 outputs: [SL_Mult, TP_Mult]
                            sl_mult = np.clip((risk_action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)
                            tp_mult = np.clip((risk_action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)
                            
                            if tp_mult < sl_mult: 
                                stats['risk_blocked_bad_rr'] += 1

                            price = raw_env.close_arrays[asset][step]
                            atr = raw_env.atr_arrays[asset][step]
                            trade_infos[asset] = {
                                'entry': price,
                                'sl': price - (direction * sl_mult * atr),
                                'tp': price + (direction * tp_mult * atr),
                                'direction': direction,
                                'risk_val': 1.0 # Placeholder since new Risk model doesn't output confidence
                            }
                        
                        action_history[asset].append(direction)
                        if len(action_history[asset]) > 20: action_history[asset].pop(0)

                    if trade_infos:
                        parsed_actions_for_state = {a: {'direction': action_history[a][-1]} for a in self.assets}
                        portfolio_state = self._get_portfolio_state(raw_env, parsed_actions_for_state, action_history)
                        tg_features = tg_calculator.get_multi_asset_obs(step, trade_infos, portfolio_state)
                        
                        outcomes = {}
                        for asset in self.assets:
                            outcomes[f'target_{asset}'] = 0.0
                            outcomes[f'pnl_{asset}'] = 0.0
                            
                        for asset, t_info in trade_infos.items():
                            raw_env.positions[asset] = {'direction': t_info['direction'], 'entry_price': t_info['entry'], 'size': 1000, 'sl': t_info['sl'], 'tp': t_info['tp'], 'entry_step': step}
                            result = raw_env._simulate_trade_outcome_with_timing(asset)
                            label = 1 if result['exit_reason'] == 'TP' or (result['pnl'] > 0 and result['closed']) else 0
                            outcomes[f'target_{asset}'] = label
                            outcomes[f'pnl_{asset}'] = result['pnl']
                            
                            raw_env.positions[asset] = None

                        row = {f'f_{i}': val for i, val in enumerate(tg_features)}
                        row.update(outcomes)
                        row['timestamp'] = raw_env._get_current_timestamp()
                        dataset.append(row)
                        stats['saved_rows'] += 1

                    stats['total_steps'] += 1

                del raw_env
                del tg_calculator
                gc.collect()
                pbar.update(end_idx - start_idx)

        logger.info(f"Generation Stats: Steps={stats['total_steps']}, Signals={stats['alpha_signals']}, Saved={stats['saved_rows']}")
        if dataset:
            pd.DataFrame(dataset).to_parquet(output_file)
            logger.info(f"✅ Success! Saved {len(dataset)} training samples to {output_file}")
        else:
            logger.warning("No training samples were collected.")

    def _get_portfolio_state(self, env, parsed_actions, action_history):
        state = {}
        for asset in self.assets:
            hist = action_history[asset]
            persistence = 0
            if hist:
                current_dir = parsed_actions[asset]['direction']
                for a in reversed(hist):
                    if a == current_dir and a != 0:
                        persistence += 1
                    else:
                        break
            
            reversal = 1 if (len(hist) > 0 and hist[-1] != parsed_actions[asset]['direction'] and parsed_actions[asset]['direction'] != 0) else 0
            
            state[asset] = {
                'action_raw': parsed_actions[asset]['direction'],
                'signal_persistence': persistence,
                'signal_reversal': reversal
            }
            
        state['total_drawdown'] = 1 - (env.equity / env.peak_equity)
        state['total_exposure'] = sum(p['size'] for p in env.positions.values() if p is not None) / (env.equity + 1e-6)
        
        return state

if __name__ == "__main__":
    MODEL_PATH = "models/checkpoints/alpha/ppo_final_model.zip"
    generator = TrainingDatasetGenerator(MODEL_PATH)
    generator.generate(chunk_size=50000)

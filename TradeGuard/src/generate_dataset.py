import os
import numpy as np
import pandas as pd
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
import sys
import gc
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
    def __init__(self, alpha_model_path, alpha_norm_path, risk_model_path, risk_norm_path, data_dir='TradeGuard/data'):
        self.data_dir = data_dir
        self.alpha_model_path = alpha_model_path
        self.alpha_norm_path = alpha_norm_path
        self.risk_model_path = risk_model_path
        self.risk_norm_path = risk_norm_path
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Load all data once to slice later
        self.full_data = self._load_all_data()
        
    def _load_all_data(self):
        logger.info("Loading all market data files...")
        data = {}
        common_index = None
        
        for asset in self.assets:
            file_path = os.path.join(self.data_dir, f"{asset}_5m.parquet")
            if not os.path.exists(file_path):
                 file_path_2025 = os.path.join(self.data_dir, f"{asset}_5m_2025.parquet")
                 if os.path.exists(file_path_2025):
                     file_path = file_path_2025
                 else:
                     raise FileNotFoundError(f"Data file for {asset} not found at {file_path}")

            df = pd.read_parquet(file_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            data[asset] = df
            
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        if common_index is None or len(common_index) == 0:
            raise ValueError("No overlapping data found across assets! Cannot generate dataset.")
            
        logger.info(f"Aligning all assets to common time range. Overlapping rows: {len(common_index)}")
        
        for asset in self.assets:
            data[asset] = data[asset].loc[common_index]
            
        return data

    def generate(self, output_file='TradeGuard/data/training_dataset.parquet', chunk_size=50000):
        # Load Risk Model
        logger.info(f"Loading Risk Model from {self.risk_model_path}...")
        try:
            risk_model = PPO.load(self.risk_model_path, device='cpu')
            
            # Load Risk Normalizer if it exists
            risk_norm = None
            if os.path.exists(self.risk_norm_path):
                logger.info(f"Loading Risk Normalizer from {self.risk_norm_path}...")
                # We need a dummy env with the correct observation space (45) to load VecNormalize
                class DummyEnv(gym.Env):
                    def __init__(self):
                        super().__init__()
                        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
                        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
                    def step(self, action): return np.zeros(45), 0, False, False, {}
                    def reset(self, seed=None, options=None): return np.zeros(45), {}
                
                dummy_risk_vec_env = DummyVecEnv([lambda: DummyEnv()])
                risk_norm = VecNormalize.load(self.risk_norm_path, dummy_risk_vec_env)
                risk_norm.training = False
                risk_norm.norm_reward = False
        except Exception as e:
            logger.error(f"Failed to load Risk Model or Normalizer: {e}")
            return

        min_len = min(len(df) for df in self.full_data.values())
        total_rows = min_len
        
        dataset = []
        
        stats = {
            'total_steps': 0,
            'alpha_signals': 0,
            'saved_rows': 0
        }
        
        warmup = 500 
        
        for asset in self.assets:
            logger.info(f"Processing asset: {asset}...")
            action_history = []
            
            with tqdm(total=total_rows, desc=f"Generating {asset}", unit="rows") as pbar:
                for start_idx in range(0, total_rows, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_rows)
                    slice_start = max(0, start_idx - warmup)
                    
                    if slice_start >= end_idx: continue

                    sliced_data = {a: self.full_data[a].iloc[slice_start:end_idx] for a in self.assets}
                    
                    if any(df.empty for df in sliced_data.values()):
                        logger.warning(f"Skipping chunk {start_idx}-{end_idx} due to empty dataframe.")
                        continue

                    raw_env = TradingEnv(data=sliced_data, is_training=False, stage=1)
                    raw_env.set_asset(asset)
                    
                    env = DummyVecEnv([lambda: raw_env])
                    if os.path.exists(self.alpha_norm_path):
                        env = VecNormalize.load(self.alpha_norm_path, env)
                        env.training = False
                        env.norm_reward = False
                    
                    model = PPO.load(self.alpha_model_path, env=env)
                    tg_calculator = TradeGuardFeatureCalculator(raw_env.data)
                    
                    obs = env.reset()
                    current_chunk_step = 0
                    steps_to_skip = start_idx - slice_start
                    
                    while current_chunk_step < len(raw_env.processed_data):
                        action, _ = model.predict(obs, deterministic=True)
                        
                        if current_chunk_step >= steps_to_skip:
                            parsed_act = raw_env._parse_action(action[0])
                            
                            if parsed_act['direction'] != 0:
                                stats['alpha_signals'] += 1
                                
                                # Preparation for Risk Model
                                # IMPORTANT: obs[0] is ALREADY normalized if Alpha normalizer was loaded
                                market_obs = obs[0] 
                                
                                drawdown = 1.0 - (raw_env.equity / raw_env.peak_equity)
                                equity_norm = raw_env.equity / 10000.0
                                risk_cap = max(0.2, 1.0 - (drawdown * 2.0))
                                account_obs = np.array([equity_norm, drawdown, 0.0, risk_cap, 0.0], dtype=np.float32)
                                
                                risk_obs = np.concatenate([market_obs, account_obs])
                                
                                # Apply Risk Normalizer manually if present
                                if risk_norm:
                                    risk_obs_norm = risk_norm.normalize_obs(risk_obs.reshape(1, -1))
                                else:
                                    risk_obs_norm = risk_obs.reshape(1, -1)
                                
                                # Get Risk Action
                                risk_action, _ = risk_model.predict(risk_obs_norm, deterministic=True)
                                
                                sl_mult = np.clip((risk_action[0][0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)
                                tp_mult = np.clip((risk_action[0][1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)
                                risk_val = 0.5
                                
                                price = raw_env.close_arrays[asset][raw_env.current_step]
                                atr = raw_env.atr_arrays[asset][raw_env.current_step]
                                
                                t_info = {
                                    'entry': price,
                                    'sl': price - (parsed_act['direction'] * sl_mult * atr),
                                    'tp': price + (parsed_act['direction'] * tp_mult * atr),
                                    'direction': parsed_act['direction'],
                                    'risk_val': risk_val
                                }
                                
                                portfolio_state = self._get_simplified_portfolio_state(raw_env, parsed_act, action_history, asset)
                                tg_features = tg_calculator.get_single_asset_obs(
                                    asset, 
                                    raw_env.current_step, 
                                    t_info, 
                                    portfolio_state[asset],
                                    portfolio_state
                                )

                                raw_env.positions[asset] = {
                                    'direction': t_info['direction'],
                                    'entry_price': t_info['entry'],
                                    'size': 1000, 
                                    'sl': t_info['sl'],
                                    'tp': t_info['tp'],
                                    'entry_step': raw_env.current_step
                                }
                                
                                result = raw_env._simulate_trade_outcome_with_timing(asset)
                                label = 1 if result['exit_reason'] == 'TP' or (result['pnl'] > 0 and result['closed']) else 0
                                
                                row = {f'f_{i}': val for i, val in enumerate(tg_features)}
                                row.update({
                                    'asset': asset,
                                    'target': label,
                                    'pnl': result['pnl'],
                                    'timestamp': raw_env._get_current_timestamp()
                                })
                                dataset.append(row)
                                stats['saved_rows'] += 1
                                raw_env.positions[asset] = None

                            action_history.append(parsed_act['direction'])
                            if len(action_history) > 20: action_history.pop(0)
                        
                        obs, _, _, _ = env.step(action)
                        current_chunk_step += 1

                    del model, env, raw_env, tg_calculator
                    gc.collect()
                    pbar.update(end_idx - start_idx)

        logger.info(f"Generation complete. Alpha Signals: {stats['alpha_signals']}, Saved Rows: {stats['saved_rows']}")
        if dataset:
            df = pd.DataFrame(dataset)
            df.to_parquet(output_file)
            logger.info(f"✅ Success! Saved {len(df)} samples to {output_file}")

    def _get_simplified_portfolio_state(self, env, parsed_act, action_history, current_asset):
        state = {a: {} for a in self.assets}
        persistence = 0
        if action_history:
            current_dir = parsed_act['direction']
            for a in reversed(action_history):
                if a == current_dir and a != 0: persistence += 1
                else: break
        reversal = 1 if (len(action_history) > 0 and action_history[-1] != parsed_act['direction'] and parsed_act['direction'] != 0) else 0
        state[current_asset] = {
            'action_raw': parsed_act['direction'],
            'signal_persistence': persistence,
            'signal_reversal': reversal
        }
        state['total_drawdown'] = 1 - (env.equity / env.peak_equity)
        state['total_exposure'] = sum(p['size'] for p in env.positions.values() if p is not None) / (env.equity + 1e-6)
        return state

if __name__ == "__main__":
    ALPHA_MODEL = "models/checkpoints/ppo_final_model.zip"
    ALPHA_NORM = "models/checkpoints/ppo_final_vecnormalize.pkl"
    RISK_MODEL = "models/risk/model10M.zip"
    RISK_NORM = "models/risk/model10M.pkl"
    
    # Use the 'data' directory which contains the full 600k row dataset
    generator = TrainingDatasetGenerator(ALPHA_MODEL, ALPHA_NORM, RISK_MODEL, RISK_NORM, data_dir='data')
    generator.generate()

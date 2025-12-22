
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
import sys
import gc

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
        for asset in self.assets:
            file_path = os.path.join(self.data_dir, f"{asset}_5m.parquet")
            data[asset] = pd.read_parquet(file_path)
        return data

    def generate(self, output_file='TradeGuard/data/training_dataset.parquet', chunk_size=50000):
        total_rows = len(self.full_data[self.assets[0]])
        logger.info(f"Starting dataset generation. Total rows: {total_rows}, Chunk size: {chunk_size}")
        
        dataset = []
        action_history = {asset: [] for asset in self.assets}
        
        # We need a small overlap (e.g. 500 bars) for indicators to warm up in each environment
        warmup = 500 
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            
            # Slice data with warmup
            slice_start = max(0, start_idx - warmup)
            sliced_data = {asset: self.full_data[asset].iloc[slice_start:end_idx] for asset in self.assets}
            
            logger.info(f"Processing chunk {start_idx} to {end_idx} (Warmup: {slice_start})")
            
            # Initialize environment for this chunk
            # is_training=False ensures it starts from step 500 or beginning
            raw_env = TradingEnv(data=sliced_data, is_training=False, stage=1)
            env = DummyVecEnv([lambda: raw_env])
            
            # Load model into this env
            model = PPO.load(self.model_path, env=env)
            
            # Feature calculator for this slice
            tg_calculator = TradeGuardFeatureCalculator(raw_env.data)
            
            obs = env.reset()
            # If we are in a middle chunk, we need to skip the warmup part for data collection
            # but run the Alpha model to maintain state if necessary (Alpha state is mostly in observations)
            
            current_chunk_step = 0
            # Steps to skip are the warmup bars
            steps_to_skip = start_idx - slice_start
            
            while current_chunk_step < len(raw_env.processed_data):
                action, _ = model.predict(obs, deterministic=True)
                
                # Only collect data after warmup
                if current_chunk_step >= steps_to_skip:
                    parsed_actions = raw_env._parse_action(action[0])
                    portfolio_state = self._get_portfolio_state(raw_env, parsed_actions, action_history)
                    
                    trade_infos = {}
                    for asset in self.assets:
                        act = parsed_actions[asset]
                        if act['direction'] != 0:
                            price = raw_env.close_arrays[asset][raw_env.current_step]
                            atr = raw_env.atr_arrays[asset][raw_env.current_step]
                            
                            sl_dist = act['sl_mult'] * atr
                            tp_dist = act['tp_mult'] * atr
                            
                            trade_infos[asset] = {
                                'entry': price,
                                'sl': price - (act['direction'] * sl_dist),
                                'tp': price + (act['direction'] * tp_dist),
                                'direction': act['direction']
                            }
                    
                    if trade_infos:
                        tg_features = tg_calculator.get_multi_asset_obs(
                            raw_env.current_step, 
                            trade_infos, 
                            portfolio_state
                        )
                        
                        outcomes = {}
                        for asset, t_info in trade_infos.items():
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
                            outcomes[f'target_{asset}'] = label
                            raw_env.positions[asset] = None

                        row = {f'f_{i}': val for i, val in enumerate(tg_features)}
                        row.update(outcomes)
                        row['timestamp'] = raw_env._get_current_timestamp()
                        dataset.append(row)

                    # Update persistence history
                    for asset in self.assets:
                        action_history[asset].append(parsed_actions[asset]['direction'])
                        if len(action_history[asset]) > 20:
                            action_history[asset].pop(0)

                obs, _, _, _ = env.step(action)
                current_chunk_step += 1

            # Cleanup this chunk
            del model
            del env
            del raw_env
            del tg_calculator
            gc.collect()
            
            # Periodically save to avoid losing all data on crash
            if len(dataset) > 1000:
                logger.info(f"Progress check: {len(dataset)} samples collected so far.")

        # Save Final Dataset
        if dataset:
            df = pd.DataFrame(dataset)
            df.to_parquet(output_file)
            logger.info(f"✅ Success! Saved {len(df)} training samples to {output_file}")
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
    MODEL_PATH = "Alpha/models/checkpoints/8.03.zip"
    generator = TrainingDatasetGenerator(MODEL_PATH)
    generator.generate(chunk_size=50000)

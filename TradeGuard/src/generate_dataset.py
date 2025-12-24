
import os
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
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
        for asset in self.assets:
            data[asset] = data[asset].loc[common_index]
            
        return data

    def generate(self, output_file='TradeGuard/data/training_dataset.parquet', chunk_size=50000):
        # Load Risk Model
        risk_model_path = "RiskLayer/models/2.15.zip"
        logger.info(f"Loading Risk Model from {risk_model_path}...")
        try:
            risk_model = PPO.load(risk_model_path, device='cpu')
        except Exception as e:
            logger.error(f"Failed to load Risk Model: {e}")
            return

        # Determine total rows based on the SHORTEST asset to avoid out-of-bounds slicing
        min_len = min(len(df) for df in self.full_data.values())
        total_rows = min_len
        logger.info(f"Starting dataset generation. Total rows (min across assets): {total_rows}, Chunk size: {chunk_size}")
        
        dataset = []
        action_history = {asset: [] for asset in self.assets}
        
        # Risk Layer State (Global across chunks to maintain continuity)
        from collections import deque
        risk_history_pnl = deque([0.0]*5, maxlen=5)
        risk_history_actions = deque([np.zeros(3) for _ in range(5)], maxlen=5)
        
        # We need a small overlap (e.g. 500 bars) for indicators to warm up in each environment
        warmup = 500 
        
        with tqdm(total=total_rows, desc="Generating Dataset", unit="rows") as pbar:
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                
                # Slice data with warmup
                slice_start = max(0, start_idx - warmup)
                
                # Safety check: Ensure we have enough data
                if slice_start >= end_idx:
                     logger.warning(f"Skipping chunk {start_idx}-{end_idx} (Slice start {slice_start} >= End {end_idx})")
                     continue

                sliced_data = {asset: self.full_data[asset].iloc[slice_start:end_idx] for asset in self.assets}
                
                # Double check that no dataframe is empty
                if any(df.empty for df in sliced_data.values()):
                    logger.warning(f"Skipping chunk {start_idx}-{end_idx} due to empty dataframe in slice.")
                    continue

                # Initialize environment for this chunk
                # is_training=False ensures it starts from step 500 or beginning
                raw_env = TradingEnv(data=sliced_data, is_training=False, stage=1)
                env = DummyVecEnv([lambda: raw_env])
                
                # Load Alpha model into this env
                model = PPO.load(self.model_path, env=env)
                
                # Feature calculator for this slice
                tg_calculator = TradeGuardFeatureCalculator(raw_env.data)
                
                obs = env.reset()
                
                current_chunk_step = 0
                # Steps to skip are the warmup bars
                steps_to_skip = start_idx - slice_start
                
                while current_chunk_step < len(raw_env.processed_data):
                    # 1. Get Alpha Direction
                    action, _ = model.predict(obs, deterministic=True)
                    
                    # Only collect data after warmup
                    if current_chunk_step >= steps_to_skip:
                        parsed_actions = raw_env._parse_action(action[0])
                        portfolio_state = self._get_portfolio_state(raw_env, parsed_actions, action_history)
                        
                        trade_infos = {}
                        
                        # Process each asset
                        for asset in self.assets:
                            act = parsed_actions[asset]
                            if act['direction'] != 0:
                                # 2. Prepare Risk Model Observation (165 dims)
                                # [0..139] Market State (from Alpha Obs)
                                # Note: Alpha Obs is 140 dims. Risk Model expects 140 dims. 
                                # We assume they are compatible (same feature engine).
                                market_obs = obs[0] # Alpha observation
                                
                                # [140..144] Account State
                                # RiskEnv uses: [equity_norm, drawdown, leverage, risk_cap, padding]
                                # We approximate these since we aren't running a full RiskEnv
                                equity_norm = 1.0 # Assuming constant equity for generation or tracking it
                                drawdown = 0.0    # Assuming flat
                                risk_cap = 1.0
                                account_obs = np.array([equity_norm, drawdown, 0.0, risk_cap, 0.0], dtype=np.float32)
                                
                                # [145..164] History
                                hist_pnl = np.array(risk_history_pnl, dtype=np.float32)
                                hist_acts = np.array(risk_history_actions, dtype=np.float32).flatten()
                                
                                risk_obs = np.concatenate([market_obs, account_obs, hist_pnl, hist_acts])
                                
                                # 3. Get Risk Action
                                risk_action, _ = risk_model.predict(risk_obs, deterministic=True)
                                
                                # 4. Decode Risk Action (RiskLayer Logic)
                                # SL: 0.2 - 2.0 ATR
                                sl_mult = np.clip((risk_action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)
                                # TP: 0.5 - 4.0 ATR
                                tp_mult = np.clip((risk_action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)
                                # Risk: used for position size, but here we just need SL/TP outcomes
                                risk_val = np.clip((risk_action[2] + 1) / 2, 0.0, 1.0)

                                # 5. Define Trade
                                price = raw_env.close_arrays[asset][raw_env.current_step]
                                atr = raw_env.atr_arrays[asset][raw_env.current_step]
                                
                                sl_dist = sl_mult * atr
                                tp_dist = tp_mult * atr
                                
                                trade_infos[asset] = {
                                    'entry': price,
                                    'sl': price - (act['direction'] * sl_dist),
                                    'tp': price + (act['direction'] * tp_dist),
                                    'direction': act['direction'],
                                    'sl_mult': sl_mult, # Save for reference
                                    'tp_mult': tp_mult
                                }
                                
                                # Update Risk History (Tentative - using dummy result until simulation)
                                # We update it properly after simulation below
                                
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
                                # 6. Simulate Outcome
                                result = raw_env._simulate_trade_outcome_with_timing(asset)
                                label = 1 if result['exit_reason'] == 'TP' or (result['pnl'] > 0 and result['closed']) else 0
                                outcomes[f'target_{asset}'] = label
                                
                                # 7. Update Risk History with REAL outcome
                                # PnL Ratio (approximate for history)
                                pnl_ratio = 0.01 if label == 1 else -0.01 
                                if result['pnl'] != 0:
                                     # Normalize roughly like RiskEnv
                                     pnl_ratio = result['pnl'] / 1000.0 # based on size 1000
                                
                                risk_history_pnl.append(pnl_ratio)
                                risk_history_actions.append(risk_action) # Store the raw action we just took
                                
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
                
                # Update progress bar
                pbar.update(end_idx - start_idx)
                
                # Periodically save to avoid losing all data on crash
                if len(dataset) > 1000 and len(dataset) % 10000 < 100: # Log less frequently
                    pbar.set_postfix({'samples': len(dataset)})

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

import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

# Add project root to path for Shared import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Shared.execution import ExecutionEngine, TradeConfig

class RiskManagementEnv(gym.Env):
    """
    Risk Management Environment (Sequential Episodic) - SAC Algorithm.
    
    The agent learns to optimize SL/TP placement AND position sizing over a sequence of 100 trades.
    It receives a trade signal (from Alpha model) and decides SL, TP, and position size.
    
    State Space (65):
        [0..39]:    Alpha Features (25 Asset-Specific + 15 Global)
        [40..44]:   Account State (Equity, Drawdown, Leverage, RiskCap, Padding)
        [45..49]:   PnL History (Last 5 trades normalized PnL)
        [50..64]:   Action History (Last 5 actions: [SL, TP, Size] * 5 = 15 features)
        
    Action Space (3):
        0: SL Multiplier (0.5x - 3.0x ATR)
        1: TP Multiplier (1.0x - 10.0x ATR)
        2: Position Size (0.01 - 0.10, i.e., 1% to 10% of portfolio)
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, dataset_path, initial_equity=10000.0, is_training=True):
        super(RiskManagementEnv, self).__init__()
        
        self.dataset_path = dataset_path
        self.initial_equity_base = initial_equity
        self.is_training = is_training
        
        # --- Shared Execution Engine ---
        self.engine = ExecutionEngine()
        self.config = self.engine.config
        
        # Simulation Toggles
        self.ENABLE_SLIPPAGE = False 
        
        # --- Configuration ---
        self.EPISODE_LENGTH = 100
        
        # --- Load Data ---
        self._load_data()
        
        # --- Spaces ---
        # Actions: [SL_Mult, TP_Mult, Position_Size] (Normalized -1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation: 65 features (40 Market + 5 Account + 5 PnL History + 15 Action History)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(65,), dtype=np.float32)
        
        # --- State Variables ---
        self.current_step = 0
        self.episode_start_idx = 0
        self.equity = self.initial_equity_base
        self.peak_equity = self.initial_equity_base
        self.history_pnl = deque(maxlen=5)
        self.history_actions = deque(maxlen=5)
        
    def _load_data(self):
        """
        Load and pre-process the risk dataset with Memory Mapping (mmap) optimization.
        This allows multiple environments to share RAM, preventing crashes.
        """
        import gc
        import time
        
        cache_dir = os.path.splitext(self.dataset_path)[0] + "_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Check timestamps
        parquet_mtime = os.path.getmtime(self.dataset_path)
        cache_valid = True
        required_files = ['features.npy', 'metadata.npy', 'entry_prices.npy', 'atrs.npy', 
                         'directions.npy', 'max_profit_pcts.npy', 'max_loss_pcts.npy', 
                         'close_prices.npy', 'is_usd_quote.npy', 'is_usd_base.npy']
                         
        for f in required_files:
            fp = os.path.join(cache_dir, f)
            if not os.path.exists(fp) or os.path.getmtime(fp) < parquet_mtime:
                cache_valid = False
                break
        
        # Lock file for race conditions in multiprocessing
        lock_file = os.path.join(cache_dir, "generating.lock")
        
        # Wait if another process is generating cache
        if os.path.exists(lock_file):
            print(f"[{os.getpid()}] Waiting for cache generation...")
            while os.path.exists(lock_file):
                time.sleep(1)
            # Re-check validity after wait
            cache_valid = True
        
        if not cache_valid:
            # Create Lock
            try:
                with open(lock_file, 'w') as f: f.write(str(os.getpid()))
                
                print(f"[{os.getpid()}] Generating memory-mapped cache at {cache_dir}...")
                self.df = pd.read_parquet(self.dataset_path)
                
                # --- Sanity Checks & Preprocessing ---
                if 'direction' not in self.df.columns:
                     raise ValueError("Missing 'direction' column")
                     
                chk_cols = ['max_profit_pct', 'max_loss_pct', 'close_1000_price']
                for c in chk_cols:
                    if c not in self.df.columns:
                         raise ValueError(f"Missing {c}")

                # Features
                if 'features' in self.df.columns:
                    features_data = self.df['features'].values
                    if len(features_data) > 0 and isinstance(features_data[0], (list, np.ndarray)):
                        # Stack is heavy, but necessary once
                        features_array = np.stack(features_data).astype(np.float32)
                    else:
                        features_array = np.array(features_data).astype(np.float32)
                else:
                    print("WARNING: Missing features column. Using zeros.")
                    features_array = np.zeros((len(self.df), 40), dtype=np.float32)

                # Metadata checking for non-USDs
                if 'pair' in self.df.columns:
                     pairs = self.df['pair'].astype(str).str.upper()
                     is_usd_quote = pairs.str.endswith('USD').values
                     is_usd_base = (pairs.str.startswith('USD') & ~is_usd_quote).values
                else:
                     # Fallback
                     n = len(self.df)
                     is_usd_quote = np.ones(n, dtype=bool)
                     is_usd_base = np.zeros(n, dtype=bool)

                # Save arrays
                np.save(os.path.join(cache_dir, 'features.npy'), features_array)
                np.save(os.path.join(cache_dir, 'entry_prices.npy'), self.df['entry_price'].values.astype(np.float32))
                np.save(os.path.join(cache_dir, 'atrs.npy'), self.df['atr'].values.astype(np.float32))
                np.save(os.path.join(cache_dir, 'directions.npy'), self.df['direction'].values.astype(np.float32))
                np.save(os.path.join(cache_dir, 'max_profit_pcts.npy'), self.df['max_profit_pct'].values.astype(np.float32))
                np.save(os.path.join(cache_dir, 'max_loss_pcts.npy'), self.df['max_loss_pct'].values.astype(np.float32))
                np.save(os.path.join(cache_dir, 'close_prices.npy'), self.df['close_1000_price'].values.astype(np.float32))
                np.save(os.path.join(cache_dir, 'is_usd_quote.npy'), is_usd_quote)
                np.save(os.path.join(cache_dir, 'is_usd_base.npy'), is_usd_base)
                
                # Optional time limit
                if 'hours_to_exit' in self.df.columns:
                    np.save(os.path.join(cache_dir, 'hours_to_exits.npy'), self.df['hours_to_exit'].values.astype(np.float32))
                
                # Free memory immediately
                del self.df
                del features_array
                gc.collect()
                
            except Exception as e:
                print(f"CRITICAL ERROR generating cache: {e}")
                # Remove lock if failed to avoid deadlock
                if os.path.exists(lock_file): os.remove(lock_file)
                raise e
            finally:
                if os.path.exists(lock_file): os.remove(lock_file)
                
        # --- Load from Cache (Memory Mapped) ---
        try:
            # mmap_mode='r' means Read-Only, Shared Memory. 
            # The OS loads pages once and shares them across processes.
            self.features_array = np.load(os.path.join(cache_dir, 'features.npy'), mmap_mode='r')
            self.entry_prices = np.load(os.path.join(cache_dir, 'entry_prices.npy'), mmap_mode='r')
            self.atrs = np.load(os.path.join(cache_dir, 'atrs.npy'), mmap_mode='r')
            self.directions = np.load(os.path.join(cache_dir, 'directions.npy'), mmap_mode='r')
            self.max_profit_pcts = np.load(os.path.join(cache_dir, 'max_profit_pcts.npy'), mmap_mode='r')
            self.max_loss_pcts = np.load(os.path.join(cache_dir, 'max_loss_pcts.npy'), mmap_mode='r')
            self.close_prices = np.load(os.path.join(cache_dir, 'close_prices.npy'), mmap_mode='r')
            self.is_usd_quote_arr = np.load(os.path.join(cache_dir, 'is_usd_quote.npy'), mmap_mode='r')
            self.is_usd_base_arr = np.load(os.path.join(cache_dir, 'is_usd_base.npy'), mmap_mode='r')
            
            p_time = os.path.join(cache_dir, 'hours_to_exits.npy')
            if os.path.exists(p_time):
                self.hours_to_exits = np.load(p_time, mmap_mode='r')
                self.has_time_limit = True
            else:
                self.has_time_limit = False

            self.n_samples = len(self.entry_prices)
            
        except Exception as e:
            print(f"Error loading mmap cache: {e}. Falling back to clean load.")
            # Verify cache might be corrupted
            import shutil
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
            self._load_data() # Retry (recursive, will regenerate)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.equity = self.initial_equity_base
        self.peak_equity = self.initial_equity_base
        self.current_step = 0
        
        # Reset History (Zero-filled) - Now 3D actions
        self.history_pnl = deque([0.0]*5, maxlen=5)
        self.history_actions = deque([np.zeros(3) for _ in range(5)], maxlen=5)
        
        # Sliding Window / Non-Overlapping Sampling
        if self.is_training:
            # Random block of 100 trades
            max_start = self.n_samples - self.EPISODE_LENGTH - 1
            if max_start > 0:
                self.episode_start_idx = int(self.np_random.integers(0, max_start + 1))
            else:
                self.episode_start_idx = 0
        else:
            # Sequential for testing
            self.episode_start_idx = 0
            
        return self._get_observation(), {}

    def _get_observation(self):
        # 1. Market State (40)
        global_idx = self.episode_start_idx + self.current_step
        if global_idx >= self.n_samples: global_idx = self.n_samples - 1
        market_obs = self.features_array[global_idx] # 40 dims
        
        # 2. Account State (5)
        drawdown = 1.0 - (self.equity / self.peak_equity)
        equity_norm = self.equity / self.initial_equity_base
        risk_cap_mult = 1.0
        
        account_obs = np.array([
            equity_norm,
            drawdown,
            0.0, # Leverage (placeholder)
            risk_cap_mult,
            0.0 # Padding
        ], dtype=np.float32)
        
        # 3. PnL History (5) - Last 5 trades normalized PnL
        pnl_hist = np.array(list(self.history_pnl), dtype=np.float32)
        
        # 4. Action History (15) - Last 5 actions flattened [SL, TP, Size] * 5
        action_hist = np.concatenate([np.array(a, dtype=np.float32) for a in self.history_actions])
        
        # Concatenate: 40 + 5 + 5 + 15 = 65
        obs = np.concatenate([market_obs, account_obs, pnl_hist, action_hist])
        
        # Safety check
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return obs

    def step(self, action):
        # --- 1. Parse Action (SAC 3D: [SL, TP, Position Size]) ---
        # Action Scaling:
        # SL: [-1, 1] -> [0.5, 3.0] ATR
        # TP: [-1, 1] -> [1.0, 10.0] ATR
        # Size: [-1, 1] -> [0.01, 0.10] (1% to 10% of portfolio)
        
        sl_mult = np.clip((action[0] + 1) / 2 * (3.0 - 0.5) + 0.5, 0.5, 3.0)
        tp_mult = np.clip((action[1] + 1) / 2 * (10.0 - 1.0) + 1.0, 1.0, 10.0)
        risk_pct = np.clip((action[2] + 1) / 2 * (0.10 - 0.01) + 0.01, 0.01, 0.10)

        # --- 2. Get Trade Data ---
        global_idx = self.episode_start_idx + self.current_step
        if global_idx >= self.n_samples:
             return self._get_observation(), 0, True, True, {}
             
        entry_price_mid = self.entry_prices[global_idx]  # Raw mid-price from dataset
        atr = self.atrs[global_idx]
        direction = self.directions[global_idx]
        
        is_usd_quote = self.is_usd_quote_arr[global_idx]
        is_usd_base = self.is_usd_base_arr[global_idx]
        
        # --- ENTRY SIMULATION (Shared Engine Source of Truth) ---
        entry_price = self.engine.get_entry_price(
            mid_price=entry_price_mid,
            direction=direction,
            atr=atr,
            enable_slippage=self.ENABLE_SLIPPAGE
        )

        # Use pre-loaded Arrays for Shadow Simulation
        max_favorable = self.max_profit_pcts[global_idx] 
        max_adverse = self.max_loss_pcts[global_idx]

        # --- 3. Calculate Lots using SAC Position Size Action ---
        sl_dist_price = sl_mult * atr
        
        # Use risk_pct from action instead of fixed 25%
        risk_amount_cash = self.equity * risk_pct
        
        # Calculate lots based on risk and SL distance
        if sl_dist_price > 1e-9:
            if is_usd_quote:
                lots = risk_amount_cash / (sl_dist_price * self.config.CONTRACT_SIZE)
            elif is_usd_base:
                lots = (risk_amount_cash * entry_price) / (sl_dist_price * self.config.CONTRACT_SIZE)
            else:
                lots = risk_amount_cash / (sl_dist_price * self.config.CONTRACT_SIZE)
        else:
            lots = 0.0
        
        # Apply leverage constraints
        if is_usd_quote:
            lot_value_usd = self.config.CONTRACT_SIZE * entry_price
        else:
            lot_value_usd = self.config.CONTRACT_SIZE
        
        max_position_value = (self.equity * 0.80) * 400.0  # 80% margin, 1:400 leverage
        max_lots_leverage = max_position_value / max(lot_value_usd, 1e-9)
        lots = min(lots, max_lots_leverage)
        
        # Clip to min/max lots
        lots = float(np.clip(lots, self.config.MIN_LOTS, self.config.MAX_LOTS))
        
        # Check Min Lots
        if lots < self.config.MIN_LOTS:
            reward = 0.0
            self.history_actions.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            self.history_pnl.append(0.0)
            self.current_step += 1
            terminated = False
            truncated = (self.current_step >= self.EPISODE_LENGTH)
            return self._get_observation(), reward, terminated, truncated, {'pnl': 0.0, 'exit': 'SKIPPED_SMALL', 'lots': 0.0, 'equity': self.equity}

        # Calculate Position Value (for info only)
        if is_usd_quote:
             lot_value_usd = self.config.CONTRACT_SIZE * entry_price
        elif is_usd_base:
             lot_value_usd = self.config.CONTRACT_SIZE * 1.0 
        else:
             lot_value_usd = self.config.CONTRACT_SIZE * 1.0
             
        position_val = lots * lot_value_usd
            
        decoded_action = np.array([sl_mult, tp_mult, risk_pct], dtype=np.float32)
        self.history_actions.append(decoded_action)

        # --- 4. Simulate Outcome ---
        tp_dist_price = tp_mult * atr
        
        # SL/TP are calculated from ENTRY price (which already includes spread for longs)
        sl_price = entry_price - (direction * sl_dist_price)
        tp_price = entry_price + (direction * tp_dist_price)
        
        # Calculate raw percentage distances using MID price for comparison with dataset
        sl_pct_dist_raw = abs(sl_dist_price / entry_price_mid)
        tp_pct_dist_raw = abs(tp_dist_price / entry_price_mid)

        if direction == 1: # LONG
             hit_sl = max_adverse <= -sl_pct_dist_raw
             hit_tp = max_favorable >= tp_pct_dist_raw
             
             sl_pct_dist = -sl_pct_dist_raw
             tp_pct_dist = tp_pct_dist_raw
             
        else: # SHORT
             hit_sl = max_favorable >= sl_pct_dist_raw
             hit_tp = max_adverse <= -tp_pct_dist_raw
             
             sl_pct_dist = sl_pct_dist_raw
             tp_pct_dist = -tp_pct_dist_raw

        if hit_sl and hit_tp:
            if abs(sl_pct_dist) < abs(tp_pct_dist):
                hit_tp = False
            else:
                hit_sl = False 
        
        # Determine exit price and apply BID/ASK at exit
        exited_on = 'TIME'
        
        if hit_sl:
            exit_price = sl_price
            exited_on = 'SL'
        elif hit_tp:
            exit_price = tp_price
            exited_on = 'TP'
        else:
            if self.has_time_limit and self.hours_to_exits[global_idx] > 24:
                exited_on = 'TIME_LIMIT'
            
            # EXIT SIMULATION (Shared Engine Source of Truth)
            close_price_mid = self.close_prices[global_idx]
            exit_price = self.engine.get_close_price(
                mid_price=close_price_mid,
                direction=direction,
                atr=atr,
                enable_slippage=self.ENABLE_SLIPPAGE
            )
            
        # --- 5. Calculate P&L (Shared Engine Source of Truth) ---
        net_pnl = self.engine.calculate_pnl(
            entry_price=entry_price,
            exit_price=exit_price,
            lots=lots,
            direction=direction,
            is_usd_quote=bool(is_usd_quote),
            is_usd_base=bool(is_usd_base)
        )
        
        # --- REWARD CALCULATION ---
        # Uses the final prices and net_pnl logic from the Shared Engine.
        # We calculate efficiency based on the actual price delta achieved.
        price_change = exit_price - entry_price
        realized_pct = (price_change / entry_price) * direction
        
        # Update Equity
        prev_equity = self.equity
        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)

        # --- NEW REWARD LOGIC ---
        denom = max(max_favorable, 1e-5)
        pnl_efficiency = (realized_pct / denom) * 10.0

        # --- Whipsaw Penalty ---
        # If SL hit but price went in wanted direction (2 * ATR), increase penalty
        if exited_on == 'SL':
             atr_2x_pct = (2.0 * atr) / entry_price
             if max_favorable >= atr_2x_pct:
                 pnl_efficiency *= 1.5
        
        bullet_bonus = 0.0
        
        if exited_on == 'SL':
            avoided_loss_dist = 0.0
            if direction == 1: 
                 avoided_loss_dist = abs(max_adverse) 
            else: 
                 avoided_loss_dist = abs(max_favorable)
                 
            if avoided_loss_dist > (sl_pct_dist_raw * 2.5):
                saved_ratio = avoided_loss_dist / max(sl_pct_dist_raw, 1e-9)
                bullet_bonus = min(saved_ratio, 3.0) * 2.0
        
        reward = pnl_efficiency + bullet_bonus
        reward = np.clip(reward, -100.0, 100.0)
        
        self.history_pnl.append(net_pnl / max(prev_equity, 1e-6))
        
        # --- 7. Termination ---
        self.current_step += 1
        terminated = False
        truncated = (self.current_step >= self.EPISODE_LENGTH)
        
        if self.equity < (self.initial_equity_base * 0.3): 
            terminated = True
            reward -= 20.0
            reward = np.clip(reward, -100.0, 100.0)
            
        info = {
            'pnl': net_pnl,
            'exit': exited_on,
            'lots': lots,
            'equity': self.equity,
            'efficiency': pnl_efficiency,
            'bullet': bullet_bonus
        }
        
        return self._get_observation(), reward, terminated, truncated, info
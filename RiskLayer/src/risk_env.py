import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

try:
    import config
except ImportError:
    # If called from outside RiskLayer (e.g. combined backtest), 
    # we might need to find config differently. 
    # For now, we assume it's available via sys.path as set by train_risk.py
    config = None

class RiskManagementEnv(gym.Env):
    """
    Risk Management Environment (Sequential Episodic).
    
    The agent learns a policy to manage risk over a sequence of 100 trades.
    It receives a trade signal (from Alpha model) and decides on SL, TP, and Position Size.
    
    State Space (45):
        [0..39]:    Alpha Features (25 Asset-Specific + 15 Global)
        [40..44]:   Account State (Equity, Drawdown, Leverage, RiskCap, Padding)
        
    Action Space (2):
        0: SL Multiplier (0.5x - 2.5x ATR)
        1: TP Multiplier (0.5x - 5.0x ATR)
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, dataset_path=None, initial_equity=None, is_training=True):
        super(RiskManagementEnv, self).__init__()
        
        # Load from config if not provided
        self.dataset_path = dataset_path if dataset_path else getattr(config, 'DATASET_PATH', 'risk_dataset.parquet')
        self.initial_equity_base = initial_equity if initial_equity else getattr(config, 'INITIAL_EQUITY', 10.0)
        self.is_training = is_training
        
        # --- Configuration ---
        self.DRAWDOWN_TERMINATION_THRESHOLD = getattr(config, 'DRAWDOWN_TERMINATION_THRESHOLD', 0.95)
        
        self.MAX_RISK_PER_TRADE = 0.02 # 2% Fixed Risk recommendation
        self.MAX_MARGIN_PER_TRADE_PCT = 0.50 # Max 50% margin
        self.MAX_LEVERAGE = getattr(config, 'MAX_LEVERAGE', 100.0)
        
        # --- Cost Structure ---
        self.TRADING_COST_PCT = 0.0
        self.MIN_LOTS = 0.01
        
        # Slippage & Spread
        self.ENABLE_SLIPPAGE = True
        self.SLIPPAGE_MAX_PIPS = getattr(config, 'SLIPPAGE_MAX_PIPS', 0.5)
        self.SPREAD_MIN_PIPS = getattr(config, 'BASE_SPREAD_PIPS', 1.2)
        self.SPREAD_ATR_FACTOR = 0.05 
        
        import sys
        self.MAX_STEPS = sys.maxsize
        print(f"[{os.getpid()}] RiskEnv Initialized: Threshold={self.DRAWDOWN_TERMINATION_THRESHOLD}, Eq={self.initial_equity_base}")
        
        # --- Load Data ---
        self._load_data()
        
        # --- Spaces ---
        # Actions: [SL_Mult, TP_Mult] (Normalized -1 to 1)
        # Removed Risk Factor (Sizing is now Fixed at 2% or managed by confidence externally)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation: 45 features (25 Asset + 15 Global + 5 Account)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32)
        
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
                         'close_prices.npy', 'is_usd_quote.npy', 'is_usd_base.npy',
                         'pip_scalars.npy', 'contract_sizes.npy']
                         
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

                # Metadata checking for non-USDs and JPY/Gold logic
                n = len(self.df)
                is_usd_quote = np.ones(n, dtype=bool)
                is_usd_base = np.zeros(n, dtype=bool)
                pip_scalars = np.full(n, 0.0001, dtype=np.float32)
                contract_sizes = np.full(n, 100000.0, dtype=np.float32)

                if 'pair' in self.df.columns:
                     pairs = self.df['pair'].astype(str).str.upper()
                     is_usd_quote = pairs.str.endswith('USD').values
                     is_usd_base = (pairs.str.startswith('USD') & ~is_usd_quote).values
                     
                     # --- Fix JPY Error ---
                     # Any pair containing JPY (USDJPY, EURJPY) has pip size 0.01
                     is_jpy = pairs.str.contains('JPY').values
                     pip_scalars[is_jpy] = 0.01
                     
                     # --- Fix Gold Contract Size & Pips ---
                     is_gold = pairs.str.contains('XAU').values
                     pip_scalars[is_gold] = 0.01 # Standard XAU pip is 0.01 (cents) usually
                     contract_sizes[is_gold] = 100.0 # Gold standard contract size
                     
                else:
                     # Fallback if pair missing: Try to infer from price?
                     # Rough heuristic: if price > 50, likely JPY or Gold
                     print("WARNING: 'pair' column missing. Inferring JPY/Gold from prices.")
                     prices = self.df['entry_price'].values
                     is_likely_jpy_or_gold = prices > 50.0
                     pip_scalars[is_likely_jpy_or_gold] = 0.01
                     # Cannot safely infer contract size from price, defaulting to 100k
                     
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
                np.save(os.path.join(cache_dir, 'pip_scalars.npy'), pip_scalars)
                np.save(os.path.join(cache_dir, 'contract_sizes.npy'), contract_sizes)
                
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
            self.pip_scalars = np.load(os.path.join(cache_dir, 'pip_scalars.npy'), mmap_mode='r')
            self.contract_sizes = np.load(os.path.join(cache_dir, 'contract_sizes.npy'), mmap_mode='r')
            
            p_time = os.path.join(cache_dir, 'hours_to_exits.npy')
            if os.path.exists(p_time):
                self.hours_to_exits = np.load(p_time, mmap_mode='r')
                self.has_time_limit = True
            else:
                self.has_time_limit = False

            self.n_samples = len(self.entry_prices)
            # print(f"[{os.getpid()}] Successfully loaded mmap dataset ({self.n_samples} rows)")
            
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
        self.episode_reward = 0.0
        self.episode_len = 0
        
        # Reset History (Zero-filled)
        self.history_pnl = deque([0.0]*5, maxlen=5)
        # BUG FIX 1: Critical History Actions Initialization
        # Now 2 actions: [SL, TP]
        self.history_actions = deque([np.zeros(2) for _ in range(5)], maxlen=5)
        
        # Random start point anywhere in the dataset
        if self.is_training and self.n_samples > 1:
            self.episode_start_idx = int(self.np_random.integers(0, self.n_samples - 1))
        else:
            self.episode_start_idx = 0
            
        return self._get_observation(), {}

    def _get_observation(self):
        # 1. Market State (40)
        global_idx = self.episode_start_idx + self.current_step
        if global_idx >= self.n_samples: global_idx = self.n_samples - 1
        market_obs = self.features_array[global_idx] # Now already 40 dims
        
        # 2. Account State (5)
        drawdown = 1.0 - (self.equity / self.peak_equity)
        equity_norm = self.equity / self.initial_equity_base
        
        # BUG FIX: Risk Cap Formula logic preserved
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0)) 
        
        account_obs = np.array([
            equity_norm,
            drawdown,
            0.0, # Leverage (placeholder)
            risk_cap_mult,
            0.0 # Padding
        ], dtype=np.float32)
        
        # 3. History (20) - REMOVED
        # hist_pnl = np.array(self.history_pnl, dtype=np.float32)
        # hist_acts = np.array(self.history_actions, dtype=np.float32).flatten()
        
        obs = np.concatenate([market_obs, account_obs])
        
        # Safety check
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return obs

    def step(self, action):
        # Initialize variables to prevent NameError in return or edge cases
        lots = 0.0
        net_pnl = 0.0
        reward = 0.0
        terminated = False
        truncated = False
        hit_sl = False
        hit_tp = False
        exited_on = 'UNKNOWN'
        exit_price = 0.0
        
        # --- 1. Parse Action ---
        sl_mult = np.clip((action[0] + 1) / 2 * 2.0 + 0.5, 0.5, 2.5)   # 0.5 - 2.5 ATR
        tp_mult = np.clip((action[1] + 1) / 2 * 4.5 + 0.5, 0.5, 5.0)   # 0.5 - 5.0 ATR
        
        # FIXED RISK: 2.0% per trade (Reverted to Fixed)
        risk_raw = 0.02 

        # --- 2. Get Trade Data (Circular Looping) ---
        global_idx = (self.episode_start_idx + self.current_step) % self.n_samples
             
        entry_price_raw = self.entry_prices[global_idx]
        atr = self.atrs[global_idx]
        direction = self.directions[global_idx]
        is_usd_quote = self.is_usd_quote_arr[global_idx]
        is_usd_base = self.is_usd_base_arr[global_idx]
        pip_scalar = self.pip_scalars[global_idx]
        contract_size = self.contract_sizes[global_idx]
        
        # Outcomes for this signal
        max_favorable = self.max_profit_pcts[global_idx] 
        max_adverse = self.max_loss_pcts[global_idx]

        # --- 3. Execute Logic ---
        # Apply Slippage
        if self.ENABLE_SLIPPAGE:
            slippage_pips = np.random.uniform(self.SLIPPAGE_MIN_PIPS, self.SLIPPAGE_MAX_PIPS)
            entry_price = entry_price_raw + (direction * -1 * slippage_pips * pip_scalar)
        else:
            entry_price = entry_price_raw

        # Apply Spread
        spread_val = (self.SPREAD_MIN_PIPS * pip_scalar) + (self.SPREAD_ATR_FACTOR * atr)
        entry_price += direction * spread_val

        # --- 4. Position Sizing (Alpha Model Style) ---
        # Default size_pct is 25% (0.5 * 0.50) but we use the environment's max margin setting
        size_pct = 0.25 # Base 25% of equity per trade
        
        # Apply Risk Capacitor (Drawdown protection)
        peak_safe = max(self.peak_equity, 1e-9)
        drawdown_val = 1.0 - (self.equity / peak_safe)
        risk_cap_mult = max(0.2, 1.0 - (drawdown_val * 2.0))
        
        margin_allocated = (size_pct * self.equity) * risk_cap_mult
        
        # Minimum position check
        if margin_allocated < 0.1: 
            margin_allocated = 0.1
            
        # Maximum position check (from config/defaults)
        margin_allocated = min(margin_allocated, self.equity * self.MAX_MARGIN_PER_TRADE_PCT)
        
        # Define Leverage
        leverage = self.MAX_LEVERAGE
        
        # Calculate Lots for logging (informational)
        # Notion = margin * leverage. 1 lot = contract_size (usually 100,000 for FX).
        notional_value = margin_allocated * leverage
        lots = notional_value / max(contract_size, 1e-9) 
        
        # --- Trade Execution ---
        tp_dist_price = tp_mult * atr
        sl_dist_price = sl_mult * atr
        
        sl_price = entry_price - (direction * sl_dist_price)
        tp_price = entry_price + (direction * tp_dist_price)
        
        sl_pct_dist_raw = abs(sl_dist_price / entry_price_raw)
        tp_pct_dist_raw = abs(tp_dist_price / entry_price_raw)

        if direction == 1: # LONG
             hit_sl = max_adverse <= -sl_pct_dist_raw
             hit_tp = max_favorable >= tp_pct_dist_raw
        else: # SHORT
             hit_sl = max_favorable >= sl_pct_dist_raw
             hit_tp = max_adverse <= -tp_pct_dist_raw

        if hit_sl and hit_tp:
            if sl_pct_dist_raw < tp_pct_dist_raw: hit_tp = False
            else: hit_sl = False 
        
        exited_on = 'TIME'
        exit_price = self.close_prices[global_idx]
        if hit_sl:
            exit_price, exited_on = sl_price, 'SL'
        elif hit_tp:
            exit_price, exited_on = tp_price, 'TP'
        
        # PnL Calc (Alpha Model Style)
        # price_change_pct * (margin * leverage)
        price_change_pct = (exit_price - entry_price) / entry_price * direction
        gross_pnl_usd = price_change_pct * (margin_allocated * leverage)
        
        # Standard Account: Cost is captured via the Spread in the entry price.
        # Removing the redundant commission fee.
        total_fees = 0.0 
        net_pnl = gross_pnl_usd - total_fees
        
        # Reward Calculation (Remains Risk Model specific for efficiency/quality)
        realized_pct = (exit_price - entry_price) / entry_price * direction
        atr_ratio = atr / entry_price_raw
        efficiency = realized_pct / atr_ratio
        pnl_reward = np.clip(efficiency * 2.0, -5.0, 5.0)
        
        bullet_bonus = 0.0
        if exited_on == 'SL':
            avoided_loss_dist = abs(max_adverse) if direction == 1 else abs(max_favorable)
            # If we exited at SL and price went much further against us
            if avoided_loss_dist > (sl_pct_dist_raw * 1.2):
                saved_ratio = (avoided_loss_dist - sl_pct_dist_raw) / sl_pct_dist_raw
                bullet_bonus = min(saved_ratio * 1.5, 3.0)
        
        # Final Reward Assembly
        # Small step penalty to encourage faster exits or higher quality trades
        # reward = pnl_reward + bullet_bonus - 0.01 
        reward = np.clip(pnl_reward + bullet_bonus, -10.0, 10.0)
        
        # Update Equity
        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)

        # Update Episode Stats
        self.episode_reward += reward
        self.episode_len += 1
        self.current_step += 1
        self.history_actions.append(np.array([sl_mult, tp_mult], dtype=np.float32))
        self.history_pnl.append(net_pnl / max(self.equity, 1e-6))
        
        if self.episode_len % 1000 == 0:
            print(f"  [Trade {self.episode_len:3}] Rew: {reward:6.2f} | PnL: {net_pnl:8.2f} | Eq: {self.equity:8.2f} | Exit: {exited_on}", flush=True)

        # --- Termination Logic ---
        drawdown_end = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        terminated = drawdown_end >= self.DRAWDOWN_TERMINATION_THRESHOLD
        
        # Check for Data Exhaustion
        next_global_idx = self.episode_start_idx + self.current_step
        truncated = next_global_idx >= (self.n_samples - 1)
        
        if terminated:
            # Terminal penalty removed per user request
            pass
            
        if terminated or truncated:
            import sys
            reason = 'STOPOUT' if terminated else ('DATA_END' if truncated else 'UNKNOWN')
            print(f"--- EPISODE FINISHED --- Reward: {self.episode_reward:.2f} | Length: {self.episode_len} | Equity: {self.equity:.2f} | Reason: {reason}", flush=True)
            sys.stdout.flush()
            
        return self._get_observation(), reward, terminated, truncated, {'pnl': net_pnl, 'equity': self.equity, 'lots': lots}


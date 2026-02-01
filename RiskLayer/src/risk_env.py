import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

class RiskManagementEnv(gym.Env):
    """
    Risk Management Environment (Sequential Episodic).
    
    The agent learns a policy to manage risk over a sequence of 100 trades.
    It receives a trade signal (from Alpha model) and decides on SL, TP, and Position Size.
    
    State Space (45):
        [0..39]:    Alpha Features (25 Asset-Specific + 15 Global)
        [40..44]:   Account State (Equity, Drawdown, Leverage, RiskCap, Padding)
        
    Action Space (2):
        0: SL Multiplier (0.2x - 2.0x ATR)
        1: TP Multiplier (0.5x - 4.0x ATR)
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, dataset_path, initial_equity=10.0, is_training=True):
        super(RiskManagementEnv, self).__init__()
        
        self.dataset_path = dataset_path
        self.initial_equity_base = initial_equity
        self.is_training = is_training
        
        # --- Configuration ---
        self.EPISODE_LENGTH = 100
        
        # USER REQUESTED CHANGES:
        self.MAX_RISK_PER_TRADE = 0.40  # 40% Max Risk per trade (Very Agressive)
        self.MAX_MARGIN_PER_TRADE_PCT = 0.80 # Max 80% margin for $10 account survival
        self.MAX_LEVERAGE = 400.0       # 1:400 Leverage
        self.TRADING_COST_PCT = 0.0002  # ~2 pips/ticks roundtrip cost
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000     # Standard Lot
        
        # Slippage Configuration (Realistic Execution Model)
        # Slippage is modeled as 0.5 - 1.5 pips adverse movement on entry
        self.SLIPPAGE_MIN_PIPS = 0.5    # Minimum slippage in pips
        self.SLIPPAGE_MAX_PIPS = 1.5    # Maximum slippage in pips
        self.ENABLE_SLIPPAGE = True     # Toggle for A/B testing
        
        # Spread Configuration (Realistic Market Simulation)
        self.SPREAD_MIN_PIPS = 0.5      # Minimum spread in pips (e.g., 0.5 pips)
        self.SPREAD_ATR_FACTOR = 0.05   # Spread expands by 5% of ATR (Volatility adjustment)
        
        # Tradeable Signal Definitions
        self.TARGET_PROFIT_PCT = 0.001   # 0.1% Target Gain (10 pips)
        self.MAX_DRAWDOWN_PCT = 0.0005   # 0.05% Drawdown Tolerance (5 pips)
        
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
        
        # Reset History (Zero-filled)
        self.history_pnl = deque([0.0]*5, maxlen=5)
        # BUG FIX 1: Critical History Actions Initialization
        # Now 2 actions: [SL, TP]
        self.history_actions = deque([np.zeros(2) for _ in range(5)], maxlen=5)
        
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
        # --- 1. Parse Action ---
        # Clip actions to valid ranges (SCALPING ADJUSTED)
        sl_mult = np.clip((action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)   # 0.2 - 2.0 ATR
        tp_mult = np.clip((action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)   # 0.5 - 4.0 ATR
        
        # FIXED RISK: 2.0% per trade (Reverted to Fixed)
        risk_raw = 0.02 

        # --- 2. Get Trade Data (Moved Before Logic) ---
        global_idx = self.episode_start_idx + self.current_step
        if global_idx >= self.n_samples:
             return self._get_observation(), 0, True, True, {}
             
        entry_price_raw = self.entry_prices[global_idx]
        atr = self.atrs[global_idx]
        direction = self.directions[global_idx]
        
        is_usd_quote = self.is_usd_quote_arr[global_idx]
        is_usd_base = self.is_usd_base_arr[global_idx]
        
        # --- Apply Slippage (Adverse Entry) ---
        # Slippage moves the entry price AGAINST the trade direction
        if self.ENABLE_SLIPPAGE:
            slippage_pips = np.random.uniform(self.SLIPPAGE_MIN_PIPS, self.SLIPPAGE_MAX_PIPS)
            slippage_price = slippage_pips * 0.0001 * entry_price_raw
            entry_price = entry_price_raw + (direction * -1 * slippage_price)
        else:
            entry_price = entry_price_raw

        # --- Dynamic Spread Simulation (Bid/Ask Logic) ---
        # Calculate Spread: Min + (Factor * ATR)
        spread_val = (self.SPREAD_MIN_PIPS * 0.0001) + (self.SPREAD_ATR_FACTOR * atr)
        spread_pct = spread_val / entry_price_raw
        
        # Adjust Entry for Spread
        # For LONG: buy at ask (higher), for SHORT: sell at bid (lower)
        # Bug Fix 1: Directional Spread Application
        entry_price += direction * spread_val

        # Use pre-loaded Arrays for Shadow Simulation
        max_favorable = self.max_profit_pcts[global_idx] 
        max_adverse = self.max_loss_pcts[global_idx] # Always negative (e.g. -0.001)

        # --- 4. Normal Trade Execution ---
        
        # Safer Drawdown Calc
        peak_safe = max(self.peak_equity, 1e-9)
        drawdown = 1.0 - (self.equity / peak_safe)
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        # Actual Risk % (Capped by drawdown)
        actual_risk_pct = risk_raw * risk_cap_mult
        
        # Calculate Lots (Risk Based)
        sl_dist_price = max(sl_mult * atr, 1e-9) # Safety clip
        
        # ATR-based Min SL (SCALPING ADJUSTED)
        min_sl_dist = max(0.0001 * entry_price, 0.2 * atr)
        if sl_dist_price < min_sl_dist: sl_dist_price = min_sl_dist
        
        risk_amount_cash = self.equity * actual_risk_pct
        
        lots = 0.0
        if sl_dist_price > 0:
            if is_usd_quote:
                 lots = risk_amount_cash / (sl_dist_price * self.CONTRACT_SIZE)
            elif is_usd_base:
                 lots = (risk_amount_cash * entry_price) / (sl_dist_price * self.CONTRACT_SIZE)
            else:
                 lots = risk_amount_cash / (sl_dist_price * self.CONTRACT_SIZE)
        
        # Leverage Clamping
        if is_usd_quote:
             lot_value_usd = self.CONTRACT_SIZE * entry_price
        elif is_usd_base:
             lot_value_usd = self.CONTRACT_SIZE * 1.0 
        else:
             lot_value_usd = self.CONTRACT_SIZE * 1.0
        
        max_position_value = (self.equity * self.MAX_MARGIN_PER_TRADE_PCT) * self.MAX_LEVERAGE
        max_lots_leverage = max_position_value / lot_value_usd
        lots = min(lots, max_lots_leverage)
        
        # Check Min Lots
        if lots < self.MIN_LOTS:
            # Skipped/no-trade
            reward = 0.0
            self.history_actions.append(np.array([0.0, 0.0], dtype=np.float32))
            self.history_pnl.append(0.0)
            self.current_step += 1
            terminated = False
            truncated = (self.current_step >= self.EPISODE_LENGTH)
            return self._get_observation(), reward, terminated, truncated, {'pnl': 0.0, 'exit': 'SKIPPED_SMALL', 'lots': 0.0, 'equity': self.equity}

        lots = np.clip(lots, self.MIN_LOTS, 100.0)
        
        # Calculate Position Value
        position_val = lots * lot_value_usd
            
        decoded_action = np.array([sl_mult, tp_mult], dtype=np.float32)
        self.history_actions.append(decoded_action)

        # --- 5. Simulate Outcome (Normal Trade) ---
        tp_dist_price = tp_mult * atr
        
        sl_price = entry_price - (direction * sl_dist_price)
        tp_price = entry_price + (direction * tp_dist_price)
        
        # Calculate raw percentage distances
        sl_pct_dist_raw = abs(sl_dist_price / entry_price_raw)
        tp_pct_dist_raw = abs(tp_dist_price / entry_price_raw)

        # Bug Fix 2: Remove double-counting spread in hit logic
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
        
        exit_price = self.close_prices[global_idx]
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
            exit_price = self.close_prices[global_idx]
            
        # --- 6. Calculate Rewards ---
        price_change = exit_price - entry_price
        gross_pnl_quote = price_change * lots * self.CONTRACT_SIZE * direction
        
        gross_pnl_usd = 0.0
        if is_usd_quote:
             gross_pnl_usd = gross_pnl_quote
        elif is_usd_base:
             gross_pnl_usd = gross_pnl_quote / exit_price 
        else:
             # Non-USD pairs need cross rate - not implemented
             gross_pnl_usd = gross_pnl_quote # INCORRECT - needs fixing

        costs = position_val * self.TRADING_COST_PCT
        net_pnl = gross_pnl_usd - costs
        
        # Update Equity
        prev_equity = self.equity
        self.equity += net_pnl
        prev_peak = self.peak_equity
        self.peak_equity = max(self.peak_equity, self.equity)

        # --- NEW REWARD LOGIC ---
        
        # 1. PnL Efficiency: (Realized / Max_Available) * 10
        # Denominator: Use Max Available Profit, but floored at ATR to prevent explosion on choppy days
        # Max_Available is always positive (distance from entry to best price)
        # Note: max_favorable is ratio (e.g. 0.005). atr is absolute price? No, self.atrs is price.
        # Need to convert ATR to ratio for comparison.
        atr_ratio = atr / entry_price_raw
        
        denom = max(max_favorable, atr_ratio, 1e-5) # Prevent div/0
        
        # Calculate Realized Ratio (PnL / Entry Price equivalent)
        # Approximate: Net PnL / Position Value approx equals Price Change Pct
        # But easier: (exit - entry) / entry * direction
        # Bug Fix 3: Use execution entry price
        realized_pct = (price_change / entry_price) * direction
        
        # Base Efficiency Score
        # Range: -10 to +10 roughly
        pnl_efficiency = (realized_pct / denom) * 10.0
        
        # 2. Bullet Dodger (Bonus for Saving Capital)
        bullet_bonus = 0.0
        # Whipsaw penalty removed as requested
        
        if exited_on == 'SL':
            # Check 2: Bullet Dodger? (Price crashed deep)
            # Bug Fix 6: Bullet Dodger Logic Split
            avoided_loss_dist = 0.0
            if direction == 1: # LONG - Stopped out at Low, check if went Lower
                 avoided_loss_dist = abs(max_adverse) 
            else: # SHORT - Stopped out at High, check if went Higher
                 avoided_loss_dist = abs(max_favorable)
                 
            if avoided_loss_dist > (sl_pct_dist_raw * 1.5):
                # How much did we save?
                saved_ratio = avoided_loss_dist / max(sl_pct_dist_raw, 1e-9)
                # Cap at 3.0, Scale by 2.0 -> Max +6.0
                bullet_bonus = min(saved_ratio, 3.0) * 2.0
        
        # Total Reward Summation
        reward = pnl_efficiency + bullet_bonus
        
        # Final Clip to [-20, 20] to stabilize training
        reward = np.clip(reward, -20.0, 20.0)
        
        self.history_pnl.append(net_pnl / max(prev_equity, 1e-6))
        
        # --- 7. Termination ---
        self.current_step += 1
        terminated = False
        truncated = (self.current_step >= self.EPISODE_LENGTH)
        
        if self.equity < (self.initial_equity_base * 0.3): 
            terminated = True
            reward -= 20.0 # Terminal penalty
            reward = np.clip(reward, -20.0, 20.0)
            
        info = {
            'pnl': net_pnl,
            'exit': exited_on,
            'lots': lots,
            'equity': self.equity,
            'efficiency': pnl_efficiency,
            'bullet': bullet_bonus
        }
        
        return self._get_observation(), reward, terminated, truncated, info

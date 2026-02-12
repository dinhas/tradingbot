import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque

class RiskManagementEnv(gym.Env):
    """
    Risk Management Environment V3 (Sequential Episodic).
    
    The agent learns a policy to manage risk over a sequence of 100 trades.
    It receives a trade signal (from Alpha model) for a SPECIFIC PAIR and decides on SL, TP, position size.
    
    Action Space (2):
        0: SL Multiplier (0.2x - 2.0x ATR)
        1: TP Multiplier (0.5x - 5.0x ATR)
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
        self.MAX_RISKING_PCT = 0.25     # Matching Alpha Model (25% equity margin)
        self.MAX_MARGIN_PER_TRADE_PCT = 0.80 # Max 80% margin for $10 account survival
        self.MAX_LEVERAGE = 100.0       # Matching Alpha Model Leverage
        self.TRADING_COST_PCT = 0.0002  # ~2 pips/ticks roundtrip cost
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000     # Standard Lot
        
        # Slippage Configuration (Realistic Execution Model)
        # Slippage is modeled as 0.5 - 1.5 pips adverse movement on entry
        self.SLIPPAGE_MIN_PIPS = 0.5    # Minimum slippage in pips
        self.SLIPPAGE_MAX_PIPS = 1.5    # Maximum slippage in pips
        self.ENABLE_SLIPPAGE = True     # Toggle for A/B testing
        
        # BLOCKING REWARD PARAMS
        self.BLOCK_REWARD_SCALE = 100.0  # Scale for avoided loss
        self.BLOCK_PENALTY_SCALE = 50.0 # Scale for missed profit (Psychological Asymmetry)
        self.CONSTANT_BLOCK_PENALTY = -0.02 # Small cost per block
        
        # Tradeable Signal Definitions
        self.TARGET_PROFIT_PCT = 0.001   # 0.1% Target Gain (10 pips)
        self.MAX_DRAWDOWN_PCT = 0.0005   # 0.05% Drawdown Tolerance (5 pips)
        
        # --- Load Data ---
        self._load_data()
        
        # --- Spaces ---
        # Actions: [SL_Mult, TP_Mult] (Normalized -1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation: 60 features (40 market + 5 account + 5 PnL history + 10 action history)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32)
        
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
                     
                chk_cols = ['max_profit_pct', 'max_loss_pct', 'close_30_price']
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
                np.save(os.path.join(cache_dir, 'close_prices.npy'), self.df['close_30_price'].values.astype(np.float32))
                np.save(os.path.join(cache_dir, 'is_usd_quote.npy'), is_usd_quote)
                np.save(os.path.join(cache_dir, 'is_usd_base.npy'), is_usd_base)
                
                # V3 New Columns (Optional for backward compatibility with old datasets)
                if 'final_pnl_pct' in self.df.columns:
                     np.save(os.path.join(cache_dir, 'final_pnl_pcts.npy'), self.df['final_pnl_pct'].values.astype(np.float32))
                else:
                     # Fallback to zeros if not present
                     np.save(os.path.join(cache_dir, 'final_pnl_pcts.npy'), np.zeros(len(self.df), dtype=np.float32))

                if 'bars_to_exit' in self.df.columns:
                     np.save(os.path.join(cache_dir, 'bars_to_exits.npy'), self.df['bars_to_exit'].values.astype(np.float32))
                else:
                     np.save(os.path.join(cache_dir, 'bars_to_exits.npy'), np.zeros(len(self.df), dtype=np.float32))
                
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
            
            # V3 New Arrays
            self.final_pnl_pcts = np.load(os.path.join(cache_dir, 'final_pnl_pcts.npy'), mmap_mode='r')
            self.bars_to_exits = np.load(os.path.join(cache_dir, 'bars_to_exits.npy'), mmap_mode='r')
            
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
        # Simplified to 2-dim actions [SL, TP]
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
        # 1. Market State (40 features for CURRENT PAIR)
        # Features are now per-pair (25 asset + 15 global), loaded from dataset
        global_idx = self.episode_start_idx + self.current_step
        if global_idx >= self.n_samples: global_idx = self.n_samples - 1
        market_obs = self.features_array[global_idx]  # (40,) vector
        
        # 2. Account State (5)
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        equity_norm = self.equity / self.initial_equity_base
        
        # Risk cap multiplier based on drawdown
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        # Win streak normalization (track last 5 trades)
        recent_pnls = list(self.history_pnl)[-5:]
        wins = sum(1 for p in recent_pnls if p > 0)
        win_streak_norm = wins / 5.0  # 0.0 to 1.0
        
        account_obs = np.array([
            equity_norm,
            drawdown,
            0.0,  # Leverage placeholder (not used in current impl)
            risk_cap_mult,
            win_streak_norm
        ], dtype=np.float32)
        
        # 3. History PnL (5) - normalized
        hist_pnl = np.array(list(self.history_pnl), dtype=np.float32)
        
        # 4. History Actions (10) - 5 steps × 2 actions
        hist_acts = np.array([act for act in self.history_actions], dtype=np.float32).flatten()
        
        # Combine: [40 market + 5 account + 5 pnl + 10 actions] = 60
        obs = np.concatenate([market_obs, account_obs, hist_pnl, hist_acts])
        
        # Safety: NaN/Inf handling
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs
    def _evaluate_block_reward(self, max_favorable, max_adverse):
        """
        V3: Bounded blocking rewards using tanh scaling.
        All rewards are in [-3, +3] range to match the trade reward scale.
        
        The risk model should learn WHEN to block (bad trades) vs WHEN to trade.
        """
        # 1. Blocking Loss = GOOD (reward positive)
        if max_adverse < -0.03:
            # Catastrophic loss avoided — max reward
            return 3.0, "BLOCKED_CATASTROPHIC"
        
        if max_adverse < -0.015:
            # Major loss avoided — good block
            return 2.0, "BLOCKED_MAJOR_LOSS"
        
        if max_adverse < -0.005:
            # Minor loss avoided — small positive
            return 0.5, "BLOCKED_MINOR_LOSS"
             
        # 2. Blocking Winners = BAD (penalize opportunity cost)
        if max_favorable > 0.02:
            # Missed a huge winner — heavy penalty
            return -3.0, "MISSED_HUGE_WIN"
             
        if max_favorable > 0.01:
            # Missed a decent winner
            return -1.5, "MISSED_BIG_WIN"
             
        if max_favorable > 0.003:
            # Missed a small winner
            return -0.5, "MISSED_SMALL_WIN"
             
        # 3. Noise trade blocked — neutral/slightly positive
        return 0.2, "BLOCKED_NOISE"
    def step(self, action):
        # Simplified: Only 2 actions [SL, TP]
        sl_mult = np.clip((action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)   # 0.2 - 2.0 ATR
        tp_mult = np.clip((action[1] + 1) / 2 * 4.5 + 0.5, 0.5, 5.0)   # 0.5 - 5.0 ATR  
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
        if self.ENABLE_SLIPPAGE:
            slippage_pips = np.random.uniform(self.SLIPPAGE_MIN_PIPS, self.SLIPPAGE_MAX_PIPS)
            slippage_price = slippage_pips * 0.0001 * entry_price_raw
            entry_price = entry_price_raw + (direction * -1 * slippage_price)
        else:
            entry_price = entry_price_raw
        # Use pre-loaded Arrays for Shadow Simulation
        max_favorable = self.max_profit_pcts[global_idx] 
        max_adverse = self.max_loss_pcts[global_idx] # Always negative (e.g. -0.001)
        # --- 3. EXECUTE EVERY TRADE (Blocking Removed) ---
        is_blocked = False
        # --- 4. Alpha-Style Position Sizing (Fixed 25%) ---
        position_size = self.equity * self.MAX_RISKING_PCT
        position_val = position_size * self.MAX_LEVERAGE
        
        # Calculate Lots for tracking
        lot_value_usd = self.CONTRACT_SIZE * entry_price if is_usd_quote else self.CONTRACT_SIZE
        lots = position_val / (lot_value_usd + 1e-9)
        lots = np.clip(lots, self.MIN_LOTS, 100.0)
            
        # Record executed action
        decoded_action = np.array([sl_mult, tp_mult], dtype=np.float32)
        self.history_actions.append(decoded_action)
        # --- 5. Simulate Outcome ---
        sl_dist_price = sl_mult * atr
        tp_dist_price = tp_mult * atr
        
        sl_price = entry_price - (direction * sl_dist_price)
        tp_price = entry_price + (direction * tp_dist_price)
        
        if direction == 1: # LONG
             sl_pct_dist = -abs(sl_dist_price / entry_price)
             tp_pct_dist = abs(tp_dist_price / entry_price)
             hit_sl = max_adverse <= sl_pct_dist 
             hit_tp = max_favorable >= tp_pct_dist
        else: # SHORT
             sl_pct_dist = abs(sl_dist_price / entry_price)
             tp_pct_dist = -abs(tp_dist_price / entry_price)
             hit_sl = max_favorable >= sl_pct_dist
             hit_tp = max_adverse <= tp_pct_dist 
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
             gross_pnl_usd = gross_pnl_quote
        costs = position_val * self.TRADING_COST_PCT
        net_pnl = gross_pnl_usd - costs
        
        prev_equity = self.equity
        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        prev_peak_safe = max(self.peak_equity, 1e-9)
        prev_dd = 1.0 - (prev_equity / prev_peak_safe)
        new_dd  = 1.0 - (self.equity / prev_peak_safe)
        
        dd_increase = max(0.0, new_dd - prev_dd)
        dd_penalty = -(dd_increase ** 2) * 100.0
        
        prev_equity_safe = max(prev_equity, 1e-6)
        pnl_ratio = net_pnl / prev_equity_safe
        raw_pnl_reward = pnl_ratio * 100.0
        pnl_reward = np.tanh(raw_pnl_reward) * 3.0
        
        tp_bonus = 0.0
        if exited_on == 'TP':
            tp_bonus = 0.2
        
        reward = pnl_reward + dd_penalty + tp_bonus
        reward = np.clip(reward, -10.0, 10.0)
        
        self.history_pnl.append(net_pnl / prev_equity_safe)
        
        # --- 7. Termination ---
        self.current_step += 1
        terminated = False
        truncated = (self.current_step >= self.EPISODE_LENGTH)
        
        if self.equity < (self.initial_equity_base * 0.4):
            terminated = True
            reward -= 5.0
            
        info = {
            'pnl': net_pnl,
            'exit': exited_on,
            'lots': lots,
            'equity': self.equity,
            'is_blocked': False,
            'block_reward': 0.0,
            'max_fav': max_favorable,
            'max_adv': max_adverse,
            'size_f': 0.5,
            'dd_pen': dd_penalty,
            'true_outcome': self.final_pnl_pcts[global_idx],
            'ideal_bars': self.bars_to_exits[global_idx]
        }
        
        return self._get_observation(), reward, terminated, truncated, info

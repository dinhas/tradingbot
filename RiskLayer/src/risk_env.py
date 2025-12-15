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
    
    State Space (165):
        [0..139]:   Alpha Features (Market State)
        [140..144]: Account State (Equity, Drawdown, Leverage, RiskCap, Padding)
        [145..149]: History (Last 5 PnL)
        [150..164]: History (Last 5 Actions - flattened [SL_Mult, TP_Mult, Risk_Pct])
        
    Action Space (3):
        0: SL Multiplier (0.5x - 4.0x ATR)
        1: TP Multiplier (1.0x - 8.0x ATR)
        2: Risk Factor (0.01% - 100% of Max Risk) -> Scaled to Actual Risk %
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
        
        # --- Load Data ---
        self._load_data()
        
        # --- Spaces ---
        # Actions: [SL_Mult, TP_Mult, Risk_Factor] (Normalized -1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation: 165 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(165,), dtype=np.float32)
        
        # --- State Variables ---
        self.current_step = 0
        self.episode_start_idx = 0
        self.equity = self.initial_equity_base
        self.peak_equity = self.initial_equity_base
        self.history_pnl = deque(maxlen=5)
        self.history_actions = deque(maxlen=5)
        
    def _load_data(self):
        """Load and pre-process the risk dataset."""
        try:
            self.df = pd.read_parquet(self.dataset_path)
            
            # Validation
            chk_cols = ['max_profit_pct', 'max_loss_pct', 'close_1000_price']
            for c in chk_cols:
                if c not in self.df.columns:
                    raise ValueError(f"Missing column {c} in dataset")
            
            # Handle features column
            if 'features' in self.df.columns:
                # Convert features to numpy stack for speed
                # Features may be stored as lists, convert to array
                features_data = self.df['features'].values
                if isinstance(features_data[0], (list, np.ndarray)):
                    self.features_array = np.stack(features_data).astype(np.float32)
                else:
                    # If already arrays, just convert
                    self.features_array = np.array(features_data).astype(np.float32)
            else:
                # Create dummy features array (140 features) if missing
                # This allows training to continue but agent won't have market context
                print("WARNING: 'features' column missing in dataset. Using zero-filled features.")
                print("         Regenerate dataset with features for proper training.")
                n_samples = len(self.df)
                self.features_array = np.zeros((n_samples, 140), dtype=np.float32)
            
            # BUG FIX 6: Missing Dataset Validation
            if 'direction' not in self.df.columns:
                raise ValueError("Missing 'direction' column in dataset")
            if not self.df['direction'].isin([1, -1]).all():
                raise ValueError("Direction must be 1 (LONG) or -1 (SHORT)")
            
            # BUG FIX: Validate feature shape
            if self.features_array.shape[1] != 140:
                raise ValueError(f"Expected 140 features, got {self.features_array.shape[1]}")

            # Validation for Non-USD and hours_to_exit
            if 'hours_to_exit' not in self.df.columns:
                 print("INFO: Dataset missing 'hours_to_exit'. Time limit feature disabled.")
            
            if 'pair' in self.df.columns:
                 non_usd_mask = ~self.df['pair'].astype(str).str.upper().str.endswith('USD')
                 if non_usd_mask.any():
                     bad_pairs = self.df.loc[non_usd_mask, 'pair'].unique()
                     print(f"WARNING: Non-USD pairs detected: {bad_pairs}. Using approximate $100k valuation.")

            self.n_samples = len(self.df)

            # SPEED OPTIMIZATION: Convert common columns to Numpy arrays
            # Accessing dataframe via .iloc depends on python overhead and is slow. 
            # Numpy arrays are much faster for RL inner loops.
            self.entry_prices = self.df['entry_price'].values.astype(np.float32)
            self.atrs = self.df['atr'].values.astype(np.float32)
            self.directions = self.df['direction'].values.astype(np.float32)
            self.max_profit_pcts = self.df['max_profit_pct'].values.astype(np.float32)
            self.max_loss_pcts = self.df['max_loss_pct'].values.astype(np.float32)
            self.close_prices = self.df['close_1000_price'].values.astype(np.float32)
            
            # Optional Time limit
            if 'hours_to_exit' in self.df.columns:
                self.hours_to_exits = self.df['hours_to_exit'].values.astype(np.float32)
                self.has_time_limit = True
            else:
                self.has_time_limit = False

            # Pre-compute Currency Flags
            pairs = self.df['pair'].astype(str).str.upper()
            self.is_usd_quote_arr = pairs.str.endswith('USD').values
            self.is_usd_base_arr = (pairs.str.startswith('USD') & ~self.is_usd_quote_arr).values
            
        except Exception as e:
            print(f"CRITICAL: Failed to load dataset: {e}")
            # Create dummy data for sanity checks if file missing
            self.features_array = np.zeros((100, 140), dtype=np.float32)
            self.n_samples = 100
            
            # Dummy Arrays
            self.entry_prices = np.ones(100, dtype=np.float32)
            self.atrs = np.full(100, 0.001, dtype=np.float32)
            self.directions = np.ones(100, dtype=np.float32)
            self.max_profit_pcts = np.full(100, 0.01, dtype=np.float32)
            self.max_loss_pcts = np.full(100, -0.01, dtype=np.float32)
            self.close_prices = np.ones(100, dtype=np.float32)
            self.has_time_limit = False
            self.is_usd_quote_arr = np.ones(100, dtype=bool)
            self.is_usd_base_arr = np.zeros(100, dtype=bool)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.equity = self.initial_equity_base
        self.peak_equity = self.initial_equity_base
        self.current_step = 0
        
        # Reset History (Zero-filled)
        self.history_pnl = deque([0.0]*5, maxlen=5)
        # BUG FIX 1: Critical History Actions Initialization
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
        # 1. Market State (140)
        global_idx = self.episode_start_idx + self.current_step
        if global_idx >= self.n_samples: global_idx = self.n_samples - 1
        market_obs = self.features_array[global_idx]
        
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
        
        # 3. History (20)
        hist_pnl = np.array(self.history_pnl, dtype=np.float32)
        hist_acts = np.array(self.history_actions, dtype=np.float32).flatten()
        
        obs = np.concatenate([market_obs, account_obs, hist_pnl, hist_acts])
        
        # Safety check
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return obs

    def step(self, action):
        # --- 1. Parse Action ---
        # Clip actions to valid ranges (SCALPING ADJUSTED)
        sl_mult = np.clip((action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)   # 0.2 - 2.0 ATR
        tp_mult = np.clip((action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)   # 0.5 - 4.0 ATR
        risk_raw = np.clip((action[2] + 1) / 2, 0.0, 1.0)              # 0.0 - 1.0 (Percentage of MAX_RISK)

        # SKIP TRADE Check (Low Risk Request)
        if risk_raw < 1e-3:
             self.history_actions.append(np.array([sl_mult, tp_mult, 0.0], dtype=np.float32))
             self.history_pnl.append(0.0)
             
             self.current_step += 1
             terminated = False
             truncated = (self.current_step >= self.EPISODE_LENGTH)
             
             return self._get_observation(), 0.0, terminated, truncated, {'pnl': 0.0, 'exit': 'SKIPPED', 'lots': 0.0, 'equity': self.equity}
        
        # Get Trade Data
        # Get Trade Data (Optimized Numpy Access)
        global_idx = self.episode_start_idx + self.current_step
        if global_idx >= self.n_samples:
             return self._get_observation(), 0, True, True, {}
             
        entry_price = self.entry_prices[global_idx]
        atr = self.atrs[global_idx]
        direction = self.directions[global_idx]
        
        is_usd_quote = self.is_usd_quote_arr[global_idx]
        is_usd_base = self.is_usd_base_arr[global_idx]
        
        # --- 2. Calculate Position ---
        
        # BUG FIX 5: Safer Drawdown Calc (Div by Zero Protection)
        peak_safe = max(self.peak_equity, 1e-9)
        drawdown = 1.0 - (self.equity / peak_safe)
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        # Actual Risk %
        actual_risk_pct = risk_raw * self.MAX_RISK_PER_TRADE * risk_cap_mult
        
        # Calculate Lots (Risk Based)
        sl_dist_price = sl_mult * atr
        
        # ATR-based Min SL (SCALPING ADJUSTED)
        min_sl_dist = max(0.0001 * entry_price, 0.2 * atr)
        if sl_dist_price < min_sl_dist: sl_dist_price = min_sl_dist
        
        risk_amount_cash = self.equity * actual_risk_pct
        
        lots = 0.0
        if sl_dist_price > 0:
            if is_usd_quote:
                 # Standard: Risk($) / (SL_Dist * Contract)
                 lots = risk_amount_cash / (sl_dist_price * self.CONTRACT_SIZE)
            elif is_usd_base:
                 # Inverted (USDJPY): Risk($) * Price / (SL_Dist * Contract)
                 # Because SL_Dist is in JPY, we need to fund risk in JPY terms ?? 
                 # Wait: Risk(USD) = (SL_Dist(JPY) * Contract * Lots) / Price
                 # -> Lots = Risk(USD) * Price / (SL_Dist(JPY) * Contract)
                 lots = (risk_amount_cash * entry_price) / (sl_dist_price * self.CONTRACT_SIZE)
            else:
                 # Cross-pair fallback (approx)
                 lots = risk_amount_cash / (sl_dist_price * self.CONTRACT_SIZE)
        
        # Leverage Clamping
        # 1. Calculate Value of 1 Lot in USD
        if is_usd_quote:
             lot_value_usd = self.CONTRACT_SIZE * entry_price
        elif is_usd_base:
             lot_value_usd = self.CONTRACT_SIZE * 1.0 
        else:
             lot_value_usd = self.CONTRACT_SIZE * 1.0 # Approx
        
        # 2. Max Lots allowed by Leverage
        max_position_value = (self.equity * self.MAX_MARGIN_PER_TRADE_PCT) * self.MAX_LEVERAGE
        max_lots_leverage = max_position_value / lot_value_usd
        
        # 3. Take the smaller
        lots = min(lots, max_lots_leverage)
        
        # BUG FIX 2: Forced Position Size Check
        # If sensible risk calc results in lots < MIN_LOTS, we SKIP instead of forcing 0.01
        if lots < self.MIN_LOTS:
            # Register as skipped/no-trade
            self.history_actions.append(np.array([sl_mult, tp_mult, 0.0], dtype=np.float32))
            self.history_pnl.append(0.0)
            self.current_step += 1
            terminated = False
            truncated = (self.current_step >= self.EPISODE_LENGTH)
            return self._get_observation(), 0.0, terminated, truncated, {'pnl': 0.0, 'exit': 'SKIPPED_SMALL', 'lots': 0.0, 'equity': self.equity}

        # Safe to clip now
        lots = np.clip(lots, self.MIN_LOTS, 100.0)
        
        # Risk Violation Penalty Calculation (Monitoring only now, since we skip smalls)
        # But we still check if we exceeded intended risk significantly due to rounding up to MIN_LOTS (if close)
        # or max clamp.
        risk_violation_penalty = 0.0
        
        # Re-calc risk based on final lots
        if is_usd_quote:
             actual_risk_cash_pen = lots * sl_dist_price * self.CONTRACT_SIZE
        elif is_usd_base:
             actual_risk_cash_pen = (lots * sl_dist_price * self.CONTRACT_SIZE) / entry_price
        else:
             actual_risk_cash_pen = lots * sl_dist_price * self.CONTRACT_SIZE

        # BUG FIX 7: Percentage-based Penalty (reduced to prevent extreme values)
        if actual_risk_cash_pen > risk_amount_cash * 2.0 and risk_amount_cash > 1e-9:
             excess_risk_pct = (actual_risk_cash_pen - risk_amount_cash) / max(self.equity, 1e-6)
             if excess_risk_pct > 0.05: # 5% of account
                 risk_violation_penalty = -2.0 * excess_risk_pct  # Reduced from -5.0
                 risk_violation_penalty = np.clip(risk_violation_penalty, -10.0, 0.0)
        
        # Calculate Position Value
        position_val = lots * lot_value_usd
            
        decoded_action = np.array([sl_mult, tp_mult, actual_risk_pct], dtype=np.float32)
        self.history_actions.append(decoded_action)

        # --- 3. Simulate Outcome (Oracle) ---
        tp_dist_price = tp_mult * atr
        
        sl_price = entry_price - (direction * sl_dist_price)
        tp_price = entry_price + (direction * tp_dist_price)
        
        # Use pre-loaded Arrays
        max_favorable = self.max_profit_pcts[global_idx] 
        max_adverse = self.max_loss_pcts[global_idx]
     
        
        if direction == 1: # LONG
             sl_pct_dist = -abs(sl_dist_price / entry_price)
             tp_pct_dist = abs(tp_dist_price / entry_price)
             hit_sl = max_adverse <= sl_pct_dist 
             hit_tp = max_favorable >= tp_pct_dist
        else: # SHORT
             # Fixed Direction Logic
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
        # --- 4. Calculate Rewards ---
        price_change = exit_price - entry_price
        gross_pnl_quote = price_change * lots * self.CONTRACT_SIZE * direction
        
        # BUG FIX 3B: PnL Currency Conversion
        gross_pnl_usd = 0.0
        if is_usd_quote:
             gross_pnl_usd = gross_pnl_quote
        elif is_usd_base:
             # Convert Quote (JPY) to USD. Divide by Exit Price.
             gross_pnl_usd = gross_pnl_quote / exit_price 
        else:
             # Fallback
             gross_pnl_usd = gross_pnl_quote

        costs = position_val * self.TRADING_COST_PCT
        net_pnl = gross_pnl_usd - costs
        # BUG FIX: Incremental Drawdown Penalty
        prev_equity = self.equity
        prev_peak = self.peak_equity
        
        # Update Equity
        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)

        # BUG FIX: Use consistent peak for DD calculation
        prev_peak_safe = max(prev_peak, 1e-9)
        prev_dd = 1.0 - (prev_equity / prev_peak_safe)
        new_dd  = 1.0 - (self.equity / prev_peak_safe)
        
        dd_increase = max(0.0, new_dd - prev_dd)
        # Reduced penalty multiplier to prevent extreme values (was 2000.0)
        dd_penalty = -(dd_increase ** 2) * 50.0
        
        # Normalized PnL reward (clip to prevent extreme values)
        # Use log scale for better stability when equity is small
        prev_equity_safe = max(prev_equity, 1e-6)
        pnl_ratio = net_pnl / prev_equity_safe
        
        # Clip and scale reward to reasonable range
        pnl_reward = np.clip(pnl_ratio * 10.0, -50.0, 50.0)
        
        tp_bonus = 0.0
        if exited_on == 'TP':
            tp_bonus = 0.5 * (tp_mult / 8.0)  # Slightly increased TP bonus
        
        reward = pnl_reward + tp_bonus + dd_penalty + risk_violation_penalty
        
        # Final reward clipping to prevent extreme values that break value function
        reward = np.clip(reward, -100.0, 100.0)
        
        self.history_pnl.append(net_pnl / prev_equity)
        
        # --- 5. Termination ---
        self.current_step += 1
        terminated = False
        truncated = (self.current_step >= self.EPISODE_LENGTH)
        
        if self.equity < (self.initial_equity_base * 0.3): 
            terminated = True
            # Reduced terminal penalty to prevent extreme negative rewards
            reward -= 50.0
            reward = np.clip(reward, -100.0, 100.0) 
            
        info = {
            'pnl': net_pnl,
            'exit': exited_on,
            'lots': lots,
            'equity': self.equity
        }
        
        return self._get_observation(), reward, terminated, truncated, info

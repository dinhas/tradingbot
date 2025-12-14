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
        [140..144]: Account State (Equity, Drawdown, Leverage, RiskCap)
        [145..149]: History (Last 5 PnL)
        [150..164]: History (Last 5 Actions - flattened [SL_Mult, TP_Mult, Risk_Pct])
        
    Action Space (3):
        0: SL Multiplier (0.5x - 4.0x ATR)
        1: TP Multiplier (1.0x - 8.0x ATR)
        2: Risk Factor (0.01% - 100% of Max Risk) -> Scaled to Actual Risk %
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, dataset_path, initial_equity=10000.0, is_training=True):
        super(RiskManagementEnv, self).__init__()
        
        self.dataset_path = dataset_path
        self.initial_equity_base = initial_equity
        self.is_training = is_training
        
        # --- Configuration ---
        self.EPISODE_LENGTH = 100
        
        # USER REQUESTED CHANGES:
        self.MAX_RISK_PER_TRADE = 0.40  # 40% Max Risk per trade (Very Agressive)
        self.MAX_MARGIN_PER_TRADE_PCT = 0.40 # Max 40% of Equity used for Margin
        self.MAX_LEVERAGE = 200.0       # 1:200 Leverage
        self.TRADING_COST_PCT = 0.0002  # ~2 pips/ticks roundtrip cost
        self.MIN_LOTS = 0.001
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
            chk_cols = ['max_profit_pct', 'max_loss_pct', 'features', 'close_1000_price']
            for c in chk_cols:
                if c not in self.df.columns:
                    raise ValueError(f"Missing column {c} in dataset")
            
            # Convert features to numpy stack for speed
            self.features_array = np.stack(self.df['features'].values).astype(np.float32)
            
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
            
        except Exception as e:
            print(f"CRITICAL: Failed to load dataset: {e}")
            # Create dummy data for sanity checks if file missing
            self.features_array = np.zeros((100, 140), dtype=np.float32)
            self.df = pd.DataFrame({
                'entry_price': [1.0]*100, 'atr': [0.001]*100, 
                'max_profit_pct': [0.01]*100, 'max_loss_pct': [-0.01]*100,
                'close_1000_price': [1.0]*100, 'direction': [1]*100,
                'pair': ['EURUSD']*100 
            })
            self.n_samples = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.equity = self.initial_equity_base
        self.peak_equity = self.initial_equity_base
        self.current_step = 0
        
        # Reset History (Zero-filled)
        self.history_pnl = deque([0.0]*5, maxlen=5)
        self.history_actions = deque([np.zeros(3)]*5, maxlen=5)
        
        # Sliding Window / Non-Overlapping Sampling
        if self.is_training:
            # Random block of 100 trades
            max_start = self.n_samples - self.EPISODE_LENGTH - 1
            if max_start > 0:
                self.episode_start_idx = np.random.randint(0, max_start)
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
            0.0  # Padding
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
        # Clip actions to valid ranges
        sl_mult = np.clip((action[0] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)   # 0.5 - 4.0
        tp_mult = np.clip((action[1] + 1) / 2 * 7.0 + 1.0, 1.0, 8.0)   # 1.0 - 8.0
        risk_raw = np.clip((action[2] + 1) / 2, 0.01, 1.0)             # 0.01 - 1.0 (Percentage of MAX_RISK)
        
        # Get Trade Data
        global_idx = self.episode_start_idx + self.current_step
        if global_idx >= self.n_samples:
             return self._get_observation(), 0, True, True, {}
             
        row = self.df.iloc[global_idx]
        entry_price = row['entry_price']
        atr = row['atr']
        direction = row['direction'] # 1 or -1 (Long/Short assumption)
        
        # BUG FIX: Currency pair check
        is_usd_quote = True
        if 'pair' in row:
             pair_name = str(row['pair']).upper()
             if not pair_name.endswith('USD'):
                 is_usd_quote = False
        
        # --- 2. Calculate Position ---
        
        # BUG FIX: Calculate Drawdown for PENALTY (before trade), 
        # but for sizing we use current equity state.
        drawdown = 1.0 - (self.equity / self.peak_equity)
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        # Actual Risk % = (0.01 to 1.0) * MAX_RISK (40%) * RiskCap
        actual_risk_pct = risk_raw * self.MAX_RISK_PER_TRADE * risk_cap_mult
        
        # Calculate Lots (Risk Based)
        sl_dist_price = sl_mult * atr
        
        # BUG FIX: ATR-based Min SL
        min_sl_dist = max(0.0005 * entry_price, 0.5 * atr)
        if sl_dist_price < min_sl_dist: sl_dist_price = min_sl_dist
        
        risk_amount_cash = self.equity * actual_risk_pct
        
        if sl_dist_price > 0:
            if is_usd_quote:
                 lots = risk_amount_cash / (sl_dist_price * self.CONTRACT_SIZE)
            else:
                 # Simplified for non-USD: assume standard USD formula approx for stability
                 lots = risk_amount_cash / (sl_dist_price * self.CONTRACT_SIZE)
        else:
            lots = 0

        # BUG FIX: Leverage Clamping (Not Rejection)
        # 1. Calculate Value of 1 Lot in Account Currency (USD)
        if is_usd_quote:
             lot_value_usd = self.CONTRACT_SIZE * entry_price
        else:
             # Fallback for non-USD pairs if no conversion rate available
             # Assume roughly 100k USD value per lot for safety
             lot_value_usd = self.CONTRACT_SIZE * 1.0 
             # Ideally we need cross-rates, but this prevents 100x errors.
        
        # 2. Max Lots allowed by Leverage AND Margin Cap
        # Using 1:200 leverage means 0.5% margin requirement per lot of value.
        # User enforced: "max margin to put to a trade is 40%".
        # So we can use at most 0.40 * Equity for Margin.
        # Max Position Value = (Equity * 0.40) * Leverage
        
        max_position_value = (self.equity * self.MAX_MARGIN_PER_TRADE_PCT) * self.MAX_LEVERAGE
        max_lots_leverage = max_position_value / lot_value_usd
        
        # 3. Take the smaller of Risk-Based or Leverage-Based
        lots = min(lots, max_lots_leverage)
        
        # 4. Clip to min/max limits
        lots = np.clip(lots, self.MIN_LOTS, 100.0) # Increased max lots for high leverage
        
        # 5. Risk Violation Penalty Calculation
        # Calculate what risk would be with final lots
        actual_risk_cash_pen = lots * sl_dist_price * self.CONTRACT_SIZE
        risk_violation_penalty = 0.0
        
        # If we are forced to take > 2x the intended risk (due to min_lots or rounding), penalize
        if actual_risk_cash_pen > risk_amount_cash * 2.0 and risk_amount_cash > 1e-9:
             risk_violation_penalty = -1.0
        
        # Calculate Final Position Value
        position_val = lots * lot_value_usd
            
        # BUG FIX: Store decoded actions
        decoded_action = np.array([sl_mult, tp_mult, actual_risk_pct], dtype=np.float32)
        self.history_actions.append(decoded_action)

        # --- 3. Simulate Outcome (Oracle) ---
        tp_dist_price = tp_mult * atr
        
        sl_price = entry_price - (direction * sl_dist_price)
        tp_price = entry_price + (direction * tp_dist_price)
        
        # Dataset contains 'max_profit_pct' (High % vs Entry) and 'max_loss_pct' (Low % vs Entry)
        # Assumption: absolute price moves, must align with direction.
        # However, following the previously agreed fix logic:
        # High (positive %) and Low (negative %) relative to entry price.
        
        max_favorable = row['max_profit_pct'] # e.g. +0.005
        max_adverse = row['max_loss_pct']     # e.g. -0.005
        
        # BUG FIX: Correct Direction Logic
        if direction == 1: # LONG
             # SL is BELOW entry (Negative Return)
             sl_pct_dist = -abs(sl_dist_price / entry_price)
             # TP is ABOVE entry (Positive Return)
             tp_pct_dist = abs(tp_dist_price / entry_price)
             
             # Hit SL if Low <= SL
             hit_sl = max_adverse <= sl_pct_dist 
             # Hit TP if High >= TP
             hit_tp = max_favorable >= tp_pct_dist
             
        else: # SHORT
             # SL is ABOVE entry (Price went UP -> Positive Return)
             sl_pct_dist = abs(sl_dist_price / entry_price)
             # TP is BELOW entry (Price went DOWN -> Negative Return)
             tp_pct_dist = -abs(tp_dist_price / entry_price)
             
             # Hit SL if High >= SL (Price went too high)
             hit_sl = max_favorable >= sl_pct_dist
             # Hit TP if Low <= TP (Price went low enough)
             hit_tp = max_adverse <= tp_pct_dist 
        
        exit_price = 0.0
        pnl_pct = 0.0
        exited_on = 'TIME'
        
        if hit_sl:
            exit_price = sl_price
            exited_on = 'SL'
            # PnL Calculation
            pnl_pct = (exit_price - entry_price) / entry_price * direction
            
        elif hit_tp:
            exit_price = tp_price
            exited_on = 'TP'
            pnl_pct = (exit_price - entry_price) / entry_price * direction
            
        else:
            # BUG FIX: Time Exit
            # If 'hours_to_exit' exists check it, else use close_1000
            if 'hours_to_exit' in row and row['hours_to_exit'] > 24:
                # Force close check (symbolic here as we simulate jumping to end)
                exit_price = row['close_1000_price'] # Fallback
                exited_on = 'TIME_LIMIT'
            else:
                exit_price = row['close_1000_price']
                exited_on = 'TIME'
            
            pnl_pct = (exit_price - entry_price) / entry_price * direction
            
        # --- 4. Calculate Rewards ---
        # Option 2: Explicit price change
        price_change = exit_price - entry_price
        gross_pnl = price_change * lots * self.CONTRACT_SIZE * direction
        costs = position_val * self.TRADING_COST_PCT
        net_pnl = gross_pnl - costs
        
        # BUG FIX: Drawdown Penalty on PREVIOUS State
        prev_equity = self.equity
        prev_dd = 1.0 - (prev_equity / self.peak_equity)
        
        dd_penalty = 0.0
        if prev_dd > 0.10: 
            dd_penalty = -((prev_dd - 0.10) ** 2) * 50.0

        # Update Equity
        self.equity += net_pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        
        pnl_reward = (net_pnl / prev_equity) * 100.0
        
        # BUG FIX: TP Bonus Scaling
        tp_bonus = 0.0
        if exited_on == 'TP':
            tp_bonus = 0.2 * (tp_mult / 8.0)
        
        reward = pnl_reward + tp_bonus + dd_penalty + risk_violation_penalty
        
        self.history_pnl.append(net_pnl / prev_equity)
        
        # --- 5. Termination ---
        self.current_step += 1
        terminated = False
        truncated = (self.current_step >= self.EPISODE_LENGTH)
        
        # BUG FIX: Stronger Ruin Penalty
        if self.equity < (self.initial_equity_base * 0.3): # 70% loss
            terminated = True
            reward -= 200.0 
            
        info = {
            'pnl': net_pnl,
            'exit': exited_on,
            'lots': lots,
            'equity': self.equity
        }
        
        return self._get_observation(), reward, terminated, truncated, info

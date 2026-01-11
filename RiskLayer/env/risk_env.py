import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
try:
    from .feature_engine import RiskFeatureEngine
except (ImportError, ValueError):
    from RiskLayer.src.feature_engine import RiskFeatureEngine

class RiskTradingEnv(gym.Env):
    """
    Risk Management Environment (TradeGuard).
    Acts as a 'Sniper':
    1. Receives market state (High dimensional features).
    2. Decides to SKIP or TAKE trade (Long/Short).
    3. If TAKE, decides SL and TP placement.
    
    Action Space (Continuous, size 3):
    - [0] Execution Confidence: 
        * < -threshold: SHORT
        * > +threshold: LONG
        * Between: SKIP
    - [1] SL Multiplier (normalized)
    - [2] TP Multiplier (normalized)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir='data', is_training=True, data=None, max_rows=None):
        super(RiskTradingEnv, self).__init__()
        
        self.data_dir = data_dir
        self.is_training = is_training
        self.max_rows = max_rows
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Spreads (Assumes standard market conditions)
        # EURUSD: 1 pip, GBPUSD: 1.5 pips, USDJPY: 1 pip, USDCHF: 1.5 pips, XAUUSD: 20 cents
        self.spreads = {
            'EURUSD': 0.0001,
            'GBPUSD': 0.00015,
            'USDJPY': 0.01,
            'USDCHF': 0.00015,
            'XAUUSD': 0.20
        }
        
        # Configuration
        self.EXECUTION_THRESHOLD = 0.2  # Trigger Threshold
        self.MIN_RR = 1.5               # Minimum Risk:Reward
        self.ATR_SL_MIN = 1.0
        self.ATR_SL_MAX = 5.0
        self.ATR_TP_MIN = 1.0
        self.ATR_TP_MAX = 15.0
        
        # Logging Setup
        self.log_dir = Path("RiskLayer/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_id = str(uuid.uuid4())[:8]
        self.reward_log_file = self.log_dir / f"rewards_env_{self.env_id}.csv"
        
        with open(self.reward_log_file, "w") as f:
            f.write("timestamp,step,asset,avg_reward,win_rate,total_pnl,trades_taken,trades_skipped\n")
            
        # Reward Tracking (Rolling 5000 steps)
        self.period_reward = 0.0
        self.period_trades = 0
        self.period_wins = 0
        self.period_pnl = 0.0
        self.period_skipped = 0
        
        # Load and Process Data
        self.feature_engine = RiskFeatureEngine()
        if data is not None:
            self.data = data
        else:
            self.data = self._load_data()
            
        logging.info("Preprocessing data with RiskFeatureEngine...")
        self.processed_data = self.feature_engine.preprocess_data(self.data)
        logging.info(f"Data processed. Shape: {self.processed_data.shape}")
        
        # Build cache for fast access
        self._cache_data_arrays()
        
        # Action Space: [Execution_Trig, SL_Mult, TP_Mult]
        # [0] Execution: >0.2 LONG, <-0.2 SHORT, -0.2 to 0.2 SKIP
        # [1] SL: Normalized [-1, 1] -> [min, max]
        # [2] TP: Normalized [-1, 1] -> [min, max]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Observation Space
        # We need to determine the size based on the feature engine output
        # For a single asset, we extract specific columns.
        # The feature engine generates ~80 features per asset.
        # We will feed the model:
        # 1. Target Asset Features (~80)
        # 2. Global/Context Features (~20)
        # Total approx 100-110.
        self._determine_obs_shape()
        
        # State
        self.current_step = 0
        self.current_asset = self.assets[0]
        self.max_steps = len(self.processed_data) - 1
        self.equity = 10000.0
        
        # Tracking
        self.total_reward = 0
        self.trades_taken = 0
        self.trades_skipped = 0
        self.winning_trades = 0

    def _load_data(self):
        """Load labeled market data with Alpha signals."""
        data = {}
        data_dir_path = Path("data") # Use shared data directory
        for asset in self.assets:
            file_path = data_dir_path / f"{asset}_alpha_labeled.parquet"
            
            df = None
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    # Rename alpha_confidence to alpha_conf to match FeatureEngine
                    if 'alpha_confidence' in df.columns:
                        df.rename(columns={'alpha_confidence': 'alpha_conf'}, inplace=True)
                        
                    # Slice if requested
                    if self.max_rows is not None and len(df) > self.max_rows:
                        df = df.iloc[-self.max_rows:].copy()
                        
                    logging.info(f"Loaded {asset} from {file_path} ({len(df)} rows)")
                except Exception as e:
                    logging.warning(f"Failed to read {file_path}: {e}")
            
            if df is None:
                logging.warning(f"No labeled data found for {asset}, using dummy.")
                dates = pd.date_range(start='2025-01-01', periods=1000, freq='5min')
                df = pd.DataFrame(index=dates)
                base = 1.0 if 'JPY' not in asset else 150.0
                df['open'] = base
                df['high'] = base * 1.001
                df['low'] = base * 0.999
                df['close'] = base
                df['volume'] = 1000.0
                df['alpha_signal'] = 0.0
                df['alpha_conf'] = 0.0
            
            data[asset] = df
        return data

    def _cache_data_arrays(self):
        """Cache DataFrame columns as numpy arrays for performance."""
        self.feature_arrays = {}
        self.price_arrays = {}
        self.atr_arrays = {}
        self.signal_arrays = {}
        
        # Convert only numeric columns to float32 to prevent string conversion errors
        numeric_df = self.processed_data.select_dtypes(include=[np.number])
        self.master_matrix = numeric_df.astype(np.float32).values
        self.column_map = {col: i for i, col in enumerate(numeric_df.columns)}
        
        # Cache specific price arrays for simulation
        for asset in self.assets:
            self.price_arrays[asset] = {
                'high': self.data[asset]['high'].values.astype(np.float32),
                'low': self.data[asset]['low'].values.astype(np.float32),
                'close': self.data[asset]['close'].values.astype(np.float32),
                'open': self.data[asset]['open'].values.astype(np.float32)
            }
            # ATR is in processed data
            if f"{asset}_atr_14" in self.column_map:
                col_idx = self.column_map[f"{asset}_atr_14"]
                self.atr_arrays[asset] = self.master_matrix[:, col_idx]
            else:
                # Fallback if preprocessing failed to add ATR
                self.atr_arrays[asset] = np.zeros(len(self.processed_data), dtype=np.float32)
                
            # Cache Alpha Signals
            if f"{asset}_alpha_signal" in self.column_map:
                 col_idx = self.column_map[f"{asset}_alpha_signal"]
                 self.signal_arrays[asset] = self.master_matrix[:, col_idx]
            else:
                 # Should not happen if data loaded correctly, but fallback
                 self.signal_arrays[asset] = np.zeros(len(self.processed_data), dtype=np.float32)

    def _determine_obs_shape(self):
        """Calculate observation size based on one asset's view."""
        # We define a standard observation vector:
        # [Asset Specific Features] + [Global Features]
        
        # Identify features for the first asset to count them
        asset = self.assets[0]
        # Only count numeric columns that actually go into the observation
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        asset_cols = [c for c in numeric_cols if c.startswith(f"{asset}_")]
        
        self.obs_dim = len(asset_cols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        logging.info(f"Observation Dimension: {self.obs_dim}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.is_training:
            # Round-Robin asset selection for guaranteed coverage across all pairs
            if not hasattr(self, '_asset_idx'):
                # Initialize with a random start so parallel envs don't all start on the same asset
                self._asset_idx = np.random.randint(0, len(self.assets))
            else:
                self._asset_idx = (self._asset_idx + 1) % len(self.assets)
                
            self.current_asset = self.assets[self._asset_idx]
            
            # Random start, but ensure we land on a signal
            self.current_step = np.random.randint(100, self.max_steps - 2000)
            self._find_next_signal()
        else:
            # Backtest mode: usually linear, handled by caller or options
            self.current_step = 100
            if options and 'asset' in options:
                self.current_asset = options['asset']
            self._find_next_signal()
        
        self.trades_taken = 0
        self.trades_skipped = 0
        
        return self._get_observation(), {}

    def _find_next_signal(self):
        """Advances current_step to the next index where alpha_signal != 0."""
        signals = self.signal_arrays[self.current_asset]
        
        # Search from current_step onwards
        # We limit search to avoid infinite loops if no signals left
        search_limit = self.max_steps - 500
        
        while self.current_step < search_limit:
            if abs(signals[self.current_step]) > 0.01: # Check for non-zero signal
                return True
            self.current_step += 1
            
        # If we run out of data, reset to a random point (in training) or end (in eval)
        if self.is_training:
            self.current_step = np.random.randint(100, self.max_steps - 2000)
            return self._find_next_signal()
        else:
            return False

    def _get_observation(self):
        """Extract features for current asset at current step."""
        # Find columns for current asset
        # This is a bit slow if done every step by string matching.
        # Optimization: Pre-calculate column indices for each asset.
        if not hasattr(self, 'asset_col_indices'):
            self.asset_col_indices = {}
            for asset in self.assets:
                # ONLY include columns that are in our numeric column_map
                asset_cols = [c for c in self.processed_data.columns if c.startswith(f"{asset}_") and c in self.column_map]
                indices = [self.column_map[c] for c in asset_cols]
                self.asset_col_indices[asset] = indices
        
        indices = self.asset_col_indices[self.current_asset]
        obs = self.master_matrix[self.current_step, indices]
        
        # Handle NaNs
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)
            
        return obs

    def step(self, action):
        # 1. Parse Action
        execution_trig = action[0]
        sl_norm = action[1] # [-1, 1]
        tp_norm = action[2] # [-1, 1]
        
        # Map SL/TP to multipliers
        sl_mult = self._map_range(sl_norm, -1, 1, self.ATR_SL_MIN, self.ATR_SL_MAX)
        tp_mult = self._map_range(tp_norm, -1, 1, self.ATR_TP_MIN, self.ATR_TP_MAX)
        
        # 2. Determine Direction from Alpha Signal
        # The direction is FIXED by the environment's current state signal.
        current_signal = self.signal_arrays[self.current_asset][self.current_step]
        direction = 1 if current_signal > 0 else -1
        
        # 3. Determine Execution
        decision = 'SKIP'
        
        if execution_trig > 0: # User Threshold can be 0 or small positive
             decision = 'OPEN'
        else:
             decision = 'SKIP'

        # 4. Calculate Reward
        reward = 0.0
        info = {
            'action': decision,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'sl': sl_mult,
            'tp': tp_mult,
            'pnl': 0.0,
            'outcome': 'SKIP'
        }
        
        # Simulate Outcome (We simulate regardless to calculate Regret)
        # CRITICAL FIX: If SKIP, use STANDARDIZED parameters to prevent "Fake Dodge" exploit.
        # If we used agent's SL, it would set SL=min to force a loss and claim "Good Dodge".
        if decision == 'SKIP':
            sim_sl_mult = 2.0 # Standard wide stop
            sim_tp_mult = 4.0 # Standard target
        else:
            sim_sl_mult = sl_mult
            sim_tp_mult = tp_mult
            
        r_multiple, bars, outcome_type = self._simulate_trade(direction, sim_sl_mult, sim_tp_mult)
        
        if decision == 'SKIP':
            self.trades_skipped += 1
            info['outcome'] = f"SKIP_({outcome_type})"
            info['pnl'] = 0.0 # No real PnL
            
            # --- SKIP LOGIC: Regret vs Relief ---
            # We judge the SKIP based on what a "Standard Strategy" would have done.
            if r_multiple > 0:
                # Standard trade would have WON. You missed it.
                # Regret Penalty
                reward = -0.5 * r_multiple
            else:
                # Standard trade would have LOST. You were right to skip.
                # Relief Reward
                reward = 10.0 # Huge reward for dodging a bullet
                
        else:
            # OPEN TRADE
            self.trades_taken += 1
            info['pnl'] = r_multiple
            info['outcome'] = outcome_type
            
            # --- OPEN LOGIC: Chase Wins ---
            if r_multiple > 0:
                # Win: Direct PnL Reward
                reward = r_multiple
            else:
                # Loss: Money Lost + Missed Dodge Opportunity
                # r_multiple is negative (e.g. -1.0)
                # Penalty = -1.0 - 10.0 = -11.0
                reward = r_multiple - 10.0
                
            # 2. Structure Penalty (Risk:Reward)
            rr_ratio = tp_mult / sl_mult
            if rr_ratio < self.MIN_RR:
                reward -= 0.5
        
        # --- Tracking Update ---
        self.period_reward += reward
        self.total_reward += reward
        
        if decision == 'OPEN':
            self.period_trades += 1
            if r_multiple > 0:
                self.period_wins += 1
                self.winning_trades += 1
            self.period_pnl += r_multiple
        else:
            self.period_skipped += 1
            
        # --- Logging (Every 5000 steps) ---
        if self.current_step % 5000 == 0:
            avg_reward = self.period_reward / 5000
            win_rate = (self.period_wins / self.period_trades) if self.period_trades > 0 else 0.0
            
            log_line = f"{datetime.now()},{self.current_step},{self.current_asset},{avg_reward:.4f},{win_rate:.4f},{self.period_pnl:.4f},{self.period_trades},{self.period_skipped}\n"
            
            try:
                with open(self.reward_log_file, "a") as f:
                    f.write(log_line)
            except Exception as e:
                logging.error(f"Failed to write to log: {e}")
                
            # Reset period trackers
            self.period_reward = 0.0
            self.period_trades = 0
            self.period_wins = 0
            self.period_pnl = 0.0
            self.period_skipped = 0

        # 5. Advance State to NEXT SIGNAL
        self.current_step += 1
        has_next = self._find_next_signal()
        
        terminated = False
        truncated = False
        
        if not has_next or self.current_step >= self.max_steps - 500:
            truncated = True
            
        # Get next observation
        next_obs = self._get_observation()
        
        return next_obs, reward, terminated, truncated, info

    def _map_range(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _simulate_trade(self, direction, sl_mult, tp_mult):
        """
        Simulate trade outcome using realistic Bid/Ask spreads.
        Assumes cached prices (High, Low, Close) are BID prices.
        
        Logic:
        - LONG: Enter at ASK (Bid + Spread). Exit at BID.
          SL Trigger: Low (Bid) <= SL.
          TP Trigger: High (Bid) >= TP.
        - SHORT: Enter at BID. Exit at ASK (Bid + Spread).
          SL Trigger: High (Ask) >= SL -> (High_Bid + Spread) >= SL.
          TP Trigger: Low (Ask) <= TP -> (Low_Bid + Spread) <= TP.
        """
        asset = self.current_asset
        idx = self.current_step
        spread = self.spreads.get(asset, 0.0)
        
        # Price at current step (Signal Candle)
        # We assume trade is entered at the CLOSE of the signal candle (or Open of next).
        # Using CLOSE of current step for simplicity and consistency with previous logic.
        bid_current = self.price_arrays[asset]['close'][idx]
        ask_current = bid_current + spread
        
        atr = self.atr_arrays[asset][idx]
        if atr <= 0: atr = bid_current * 0.0001
        
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        
        # Determine Entry and Triggers
        if direction == 1: # Long
            entry_price = ask_current # Buy at Ask
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else: # Short
            entry_price = bid_current # Sell at Bid
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist
            
        # Look ahead
        horizon = 500
        start = idx + 1
        end = min(start + horizon, len(self.price_arrays[asset]['close']))
        
        # Future Price Arrays (Bid)
        lows_bid = self.price_arrays[asset]['low'][start:end]
        highs_bid = self.price_arrays[asset]['high'][start:end]
        
        # Check hits
        if direction == 1: # Long Checks (against Bid)
            # SL hit if Bid goes below SL
            sl_hits = lows_bid <= sl_price
            # TP hit if Bid goes above TP
            tp_hits = highs_bid >= tp_price
        else: # Short Checks (against Ask)
            # Short SL hit if Ask (High_Bid + Spread) goes above SL
            sl_hits = (highs_bid + spread) >= sl_price
            # Short TP hit if Ask (Low_Bid + Spread) goes below TP
            tp_hits = (lows_bid + spread) <= tp_price
            
        # Find first occurrence
        first_sl = np.argmax(sl_hits) if sl_hits.any() else horizon + 1
        first_tp = np.argmax(tp_hits) if tp_hits.any() else horizon + 1
        
        # Result
        outcome = ''
        exit_price = 0.0
        
        if first_sl == horizon + 1 and first_tp == horizon + 1:
            # Timed out
            outcome = 'TIMEOUT'
            bars = horizon
            # Close position at current market rates
            close_bid = self.price_arrays[asset]['close'][end-1]
            if direction == 1:
                exit_price = close_bid # Sell Long at Bid
            else:
                exit_price = close_bid + spread # Cover Short at Ask
                
        elif first_sl < first_tp:
            # SL Hit
            outcome = 'SL'
            bars = first_sl + 1
            exit_price = sl_price # Assumed filled at SL
        else:
            # TP Hit
            outcome = 'TP'
            bars = first_tp + 1
            exit_price = tp_price # Assumed filled at TP
            
        # Calculate R-multiple
        # Real PnL
        if direction == 1:
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price
            
        # Risk was sl_dist (Price Distance)
        # R = PnL / Risk
        r_multiple = pnl / sl_dist if sl_dist > 0 else 0
        
        return r_multiple, bars, outcome

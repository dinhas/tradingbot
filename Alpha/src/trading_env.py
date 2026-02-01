import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from pathlib import Path
try:
    from .feature_engine import FeatureEngine
except (ImportError, ValueError):
    from feature_engine import FeatureEngine

class TradingEnv(gym.Env):
    """
    Simplified Trading environment for RL agent.
    Focuses on single-pair direction (Buy/Sell/Flat).
    
    Reward System:
        - Peeked P&L: Primary signal via PEEK & LABEL (solves credit assignment)
        - Drawdown Penalty: Progressive risk control
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir='data', is_training=True, data=None, stage=1):
        super(TradingEnv, self).__init__()
        
        self.data_dir = data_dir
        self.is_training = is_training
        self.stage = stage
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Configuration Constants
        self.MIN_POSITION_SIZE = 0.1
        self.MIN_ATR_MULTIPLIER = 0.0001
        self.REWARD_LOG_INTERVAL = 5000
        
        # Load Data
        if data is not None:
            self.data = data
        else:
            self.data = self._load_data()
            
        self.feature_engine = FeatureEngine()
        self.raw_data, self.processed_data = self.feature_engine.preprocess_data(self.data)
        
        # OPTIMIZATION: Build static observation matrix
        self._build_optimization_matrix()
        self._cache_data_arrays()
        
        # Simple Action Space: 1 output for Direction (Buy, Sell, Flat)
        self.action_dim = 1
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        # Define Observation Space (40 features: 25 asset-specific + 15 global)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        
        # State Variables
        self.current_step = 0
        self.current_asset = self.assets[0] # Default, will be randomized in reset
        self.max_steps = len(self.processed_data) - 1
        
        # PRD Risk Constants
        self.MAX_POS_SIZE_PCT = 0.50
        self.MAX_TOTAL_EXPOSURE = 0.60
        self.DRAWDOWN_LIMIT = 0.25
        
    def _build_optimization_matrix(self):
        """
        Constructs a master numpy matrix (Steps x 140) containing all STATIC market data.
        Dynamic features (portfolio state) are left as 0 and filled at runtime.
        """
        n_steps = len(self.processed_data)
        # We keep 140 internally to store all assets, but will extract 40 for observation
        self.master_obs_matrix = np.zeros((n_steps, 140), dtype=np.float32)
        
        # Internal map for ALL possible features
        self.internal_feature_map = {}
        idx = 0
        for asset in self.assets:
            for feat in ["close", "return_1", "return_12", "atr_14", "atr_ratio", "bb_position", 
                        "ema_9", "ema_21", "price_vs_ema9", "ema9_vs_ema21", "rsi_14", "macd_hist", "volume_ratio"]:
                self.internal_feature_map[f"{asset}_{feat}"] = idx
                idx += 1
            # Skip position state (7) - dynamic
            idx += 7
            for feat in ["corr_basket", "rel_strength", "corr_xauusd", "corr_eurusd", "rank"]:
                self.internal_feature_map[f"{asset}_{feat}"] = idx
                idx += 1
        
        # Global features (indices 125-139)
        global_feats = [
            "equity", "margin_usage_pct", "drawdown", "num_open_positions",
            "risk_on_score", "asset_dispersion", "market_volatility",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "session_asian", "session_london", "session_ny", "session_overlap"
        ]
        for feat in global_feats:
            self.internal_feature_map[feat] = idx
            idx += 1

        # Cache dynamic indices for fast updates
        self.dynamic_indices = {feat: self.internal_feature_map[feat] for feat in global_feats[:4]}
        
        self.asset_dynamic_indices = {}
        for asset in self.assets:
            # Re-calculating indices for dynamic position state
            base_idx = self.assets.index(asset) * 25
            self.asset_dynamic_indices[asset] = {
                'has_position': base_idx + 13,
                'position_size': base_idx + 14,
                'unrealized_pnl': base_idx + 15,
                'position_age': base_idx + 16,
                'entry_price': base_idx + 17,
                'current_sl': base_idx + 18,
                'current_tp': base_idx + 19
            }
        
        # Fill static features from DataFrame columns
        for col in self.processed_data.columns:
            if col in self.internal_feature_map:
                idx = self.internal_feature_map[col]
                self.master_obs_matrix[:, idx] = self.processed_data[col].values
            elif any(col.endswith(f"_{feat}") for feat in ["risk_on_score", "asset_dispersion", "market_volatility", "hour_sin", "hour_cos", "day_sin", "day_cos", "session_asian", "session_london", "session_ny", "session_overlap"]):
                # Handle global columns that might not have asset prefix but are in internal_feature_map
                feat_name = col.split('_', 1)[-1] if '_' in col else col
                if feat_name in self.internal_feature_map:
                    self.master_obs_matrix[:, self.internal_feature_map[feat_name]] = self.processed_data[col].values
        
        # Ensure session features are filled (they are named exactly in the dataframe usually)
        for feat in ["hour_sin", "hour_cos", "day_sin", "day_cos", "session_asian", "session_london", "session_ny", "session_overlap", "risk_on_score", "asset_dispersion", "market_volatility"]:
            if feat in self.processed_data.columns:
                self.master_obs_matrix[:, self.internal_feature_map[feat]] = self.processed_data[feat].values

    def _get_observation(self):
        """
        Optimized observation retrieval. Extracts 40 features for current asset.
        """
        # 1. Update master matrix with dynamic features (Internal 140-dim representation)
        full_obs = self.master_obs_matrix[self.current_step].copy()
        
        # Update Global Dynamic
        total_exposure = sum(pos['size'] for pos in self.positions.values() if pos is not None)
        full_obs[self.dynamic_indices['equity']] = self.equity
        full_obs[self.dynamic_indices['margin_usage_pct']] = total_exposure / self.equity if self.equity > 0 else 0
        full_obs[self.dynamic_indices['drawdown']] = 1.0 - (self.equity / self.peak_equity)
        full_obs[self.dynamic_indices['num_open_positions']] = sum(1 for p in self.positions.values() if p is not None)
        
        # Update Per-Asset Dynamic
        current_prices = self._get_current_prices()
        for asset in self.assets:
            pos = self.positions[asset]
            indices = self.asset_dynamic_indices[asset]
            if pos:
                price_change = (current_prices[asset] - pos['entry_price']) * pos['direction']
                price_change_pct = price_change / pos['entry_price'] if pos['entry_price'] != 0 else 0
                unrealized_pnl = price_change_pct * (pos['size'] * self.leverage)
                
                full_obs[indices['has_position']] = 1.0
                full_obs[indices['position_size']] = pos['size'] / self.equity
                full_obs[indices['unrealized_pnl']] = unrealized_pnl
                full_obs[indices['position_age']] = self.current_step - pos['entry_step']
                full_obs[indices['entry_price']] = pos['entry_price']
                full_obs[indices['current_sl']] = pos['sl']
                full_obs[indices['current_tp']] = pos['tp']

        # 2. Extract the 40 features for current_asset
        # [25 asset features] + [15 global features]
        asset_start_idx = self.assets.index(self.current_asset) * 25
        asset_features = full_obs[asset_start_idx : asset_start_idx + 25]
        global_features = full_obs[125:140]
        
        return np.concatenate([asset_features, global_features])

    def _cache_data_arrays(self):
        """Cache DataFrame columns as numpy arrays for performance."""
        self.close_arrays = {}
        self.low_arrays = {}
        self.high_arrays = {}
        self.atr_arrays = {}
        
        for asset in self.assets:
            self.close_arrays[asset] = self.raw_data[f"{asset}_close"].values.astype(np.float32)
            self.low_arrays[asset] = self.raw_data[f"{asset}_low"].values.astype(np.float32)
            self.high_arrays[asset] = self.raw_data[f"{asset}_high"].values.astype(np.float32)
            self.atr_arrays[asset] = self.raw_data[f"{asset}_atr_14"].values.astype(np.float32)

    def _load_data(self):
        """Load market data for all assets."""
        data = {}
        data_dir_path = Path(self.data_dir)
        for asset in self.assets:
            file_path = data_dir_path / f"{asset}_5m.parquet"
            file_path_2025 = data_dir_path / f"{asset}_5m_2025.parquet"
            
            
            df = None
            try:
                df = pd.read_parquet(file_path)
                logging.info(f"Loaded {asset} from {file_path}")
            except FileNotFoundError:
                try:
                    df = pd.read_parquet(file_path_2025)
                    logging.info(f"Loaded {asset} from {file_path_2025}")
                except FileNotFoundError:
                    logging.error(f"Data file not found: {file_path} or {file_path_2025}")
                    logging.warning(f"Using dummy data for {asset} - BACKTEST WILL NOT BE ACCURATE!")
                    
                    # FIX: Use realistic default prices for different assets
                    default_prices = {
                        'EURUSD': 1.1000,
                        'GBPUSD': 1.3000,
                        'USDJPY': 150.00,
                        'USDCHF': 0.9000,
                        'XAUUSD': 2000.00
                    }
                    base_price = default_prices.get(asset, 1.0000)
                    
                    # FIX: Use raw column names, _align_data will add prefixes
                    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
                    df = pd.DataFrame(index=dates)
                    df["open"] = base_price
                    df["high"] = base_price * 1.001
                    df["low"] = base_price * 0.999
                    df["close"] = base_price
                    df["volume"] = 100
                    df["atr_14"] = self.MIN_ATR_MULTIPLIER * base_price  # Default small ATR
                    
            
            data[asset] = df
        return data

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Support forcing a specific asset via options (useful for backtesting/shuffling)
        if options and 'asset' in options:
            self.current_asset = options['asset']
        elif self.is_training:
            # Randomly select asset for this episode
            self.current_asset = np.random.choice(self.assets)
        # In backtesting, if no option provided, keep the current_asset (set via set_asset or init)
        
        if self.is_training:
            # Training: Randomize for diversity
            self.equity = np.random.uniform(5000.0, 15000.0)
            self.leverage = 100
            self.current_step = np.random.randint(500, self.max_steps - 288)
        else:
            # Backtesting: Fixed equity, randomize start point
            self.equity = 10000.0
            self.leverage = 100
            # Backtesting: Start from step 500 for indicator warmup
            self.current_step = 500
            
        self.start_equity = self.equity
        self.peak_equity = self.equity
        self.positions = {asset: None for asset in self.assets}
        self.portfolio_history = []
        
        # Reset reward tracker
        self.peeked_pnl_step = 0.0
        self.max_step_reward = -float('inf')  # TRACKING: Best single step reward in episode
        
        # Reset trade tracking
        self.completed_trades = []
        self.all_trades = []
        
        return self._get_observation(), {}
        
    def set_asset(self, asset):
        """Set the current asset for the environment."""
        if asset not in self.assets:
            raise ValueError(f"Asset {asset} not found in environment assets.")
        self.current_asset = asset

    def _validate_observation(self, obs):
        """Ensure observation shape matches space definition."""
        if obs.shape != self.observation_space.shape:
             raise ValueError(f"Observation shape mismatch: expected {self.observation_space.shape}, got {obs.shape}")
        return obs

    def step(self, action):
        """Execute one environment step."""
        # Reset step tracker
        self.peeked_pnl_step = 0.0
        self.completed_trades = []
        
        # Parse and execute trades (ONLY for current asset)
        parsed_action = self._parse_action(action)
        self._execute_trades({self.current_asset: parsed_action})
        
        # Margin call check
        if self.equity <= 0:
            self.equity = 0.01
            # FIX: Clear positions on margin call to reflect liquidation
            self.positions = {asset: None for asset in self.assets}
            return (
                self._validate_observation(self._get_observation()),
                -1.0,  # Strong terminal penalty
                True, False,
                {'trades': [], 'equity': 0.01, 'termination_reason': 'margin_call'}
            )
        
        # Advance time
        self.current_step += 1
        
        # Update positions (SL/TP checks)
        self._update_positions()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Termination checks
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        # Update peak equity BEFORE calculating drawdown
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = 1.0 - (self.equity / self.peak_equity)
        
        # Drawdown termination: Only apply during training
        # During backtesting, we want to see full performance over entire dataset
        if drawdown > self.DRAWDOWN_LIMIT and self.is_training:
            terminated = True
            reward -= 0.5  # Terminal drawdown penalty
        
        info = {
            'trades': self.completed_trades,
            'equity': self.equity,
            'drawdown': drawdown,
            'timestamp': self._get_current_timestamp(),
            'asset': self.current_asset
        }
        
        return self._validate_observation(self._get_observation()), reward, terminated, truncated, info

    def _parse_action(self, action):
        """Parse raw action array into trading decision (Direction only)."""
        # FIX: Validate action shape
        if len(action) != self.action_dim:
            raise ValueError(f"Action array has {len(action)} elements, expected {self.action_dim}")
            
        direction_raw = action[0]
                
        return {
            'direction': 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0),
            'size': 0.5,     # Fixed: 25% of equity (0.5 * MAX_POS_SIZE_PCT)
            'sl_mult': 2.0,  # Fixed: 2.0x ATR
            'tp_mult': 4.0   # Fixed: 4.0x ATR
        }

    def _execute_trades(self, actions):
        """Execute trading decisions for all assets."""
        current_prices = self._get_current_prices()
        atrs = self._get_current_atrs()
        
        for asset, act in actions.items():
            direction = act['direction']
            current_pos = self.positions[asset]
            price = current_prices[asset]
            atr = atrs[asset]
            
            if current_pos is None:
                # No position: open if direction != 0
                if direction != 0:
                    self._open_position(asset, direction, act, price, atr)
            
            elif current_pos['direction'] == direction:
                # Same direction: hold
                pass
            
            elif direction != 0 and current_pos['direction'] != direction:
                # Opposite direction: close and reverse
                self._close_position(asset, price)
                self._open_position(asset, direction, act, price, atr)
                
            elif direction == 0 and current_pos is not None:
                # Flat signal: close position
                self._close_position(asset, price)

    def _check_global_exposure(self, new_position_size):
        """Check if adding position would exceed 60% exposure limit."""
        current_exposure = sum(
            pos['size'] for pos in self.positions.values() if pos is not None
        )
        total_allocated = current_exposure + new_position_size
        return total_allocated <= (self.equity * self.MAX_TOTAL_EXPOSURE)

    def _open_position(self, asset, direction, act, price, atr):
        """Open a new position with PEEK & LABEL reward assignment."""
        # Risk Validation: Position size
        size_pct = act['size'] * self.MAX_POS_SIZE_PCT
        position_size = size_pct * self.equity
        
        # Minimum position check
        if position_size < self.MIN_POSITION_SIZE:
            return
        
        # Maximum position check
        position_size = min(position_size, self.equity * 0.5)
        
        # Global exposure check
        if not self._check_global_exposure(position_size):
            return
            
        # Calculate SL/TP levels (FIX: Handle zero ATR edge case)
        atr = max(atr, price * self.MIN_ATR_MULTIPLIER)  # Minimum 0.01% of price
        sl_dist = act['sl_mult'] * atr
        tp_dist = act['tp_mult'] * atr
        sl = price - (direction * sl_dist)
        tp = price + (direction * tp_dist)
        
        # Create position
        self.positions[asset] = {
            'direction': direction,
            'entry_price': price,
            'size': position_size,
            'sl': sl,
            'tp': tp,
            'entry_step': self.current_step,
            'sl_dist': sl_dist,
            'tp_dist': tp_dist
        }
        
        # PEEK & LABEL: Simulate outcome and assign reward NOW
        simulated_pnl, fast_win = self._simulate_trade_outcome(asset)
        self.peeked_pnl_step += simulated_pnl
        
        # FAST WIN BONUS: Double reward if TP hit within 30 mins (6 candles)
        if fast_win and simulated_pnl > 0:
            self.peeked_pnl_step += simulated_pnl
        
        # Transaction costs (FIX: Use notional position size, not leveraged)
        # 0.00002 = 0.2 pips spread (entry cost only, exit cost applied on close)
        cost = position_size * 0.00002
        self.equity -= cost

    def _close_position(self, asset, price):
        """Close position and record trade."""
        pos = self.positions[asset]
        if pos is None:
            return
        
        equity_before = self.equity
        
        # Calculate P&L
        price_change = (price - pos['entry_price']) * pos['direction']
        price_change_pct = price_change / pos['entry_price'] if pos['entry_price'] != 0 else 0
        position_value = pos['size'] * self.leverage
        pnl = price_change_pct * position_value
        
        # Update equity
        self.equity += pnl
        
        # Exit transaction cost (FIX: Use notional position size, not leveraged)
        cost = pos['size'] * 0.00002
        self.equity -= cost
        
        # Prevent negative equity
        self.equity = max(self.equity, 0.01)
        
        # Record trade for backtesting
        hold_time = (self.current_step - pos['entry_step']) * 5  # 5 min per step
        trade_record = {
            'timestamp': self._get_current_timestamp(),
            'asset': asset,
            'action': 'BUY' if pos['direction'] == 1 else 'SELL',
            'size': pos['size'],
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'sl': pos['sl'],
            'tp': pos['tp'],
            'pnl': pnl,
            'net_pnl': pnl - cost,
            'fees': cost,
            'equity_before': equity_before,
            'equity_after': self.equity,
            'hold_time': hold_time,
            'rr_ratio': pos['tp_dist'] / pos['sl_dist'] if pos['sl_dist'] > 0 else 0
        }
        
        self.completed_trades.append(trade_record)
        self.all_trades.append(trade_record)
        self.positions[asset] = None

    def _update_positions(self):
        """Check SL/TP for all open positions."""
        current_prices = self._get_current_prices()
        
        # FIX: Safe iteration using list() to avoid runtime issues if dict changes
        for asset, pos in list(self.positions.items()):
            if pos is None:
                continue
            
            price = current_prices[asset]
            
            # Check SL/TP (use SL/TP price for exit, not gap price)
            if pos['direction'] == 1:  # Long
                if price <= pos['sl']:
                    self._close_position(asset, pos['sl'])
                elif price >= pos['tp']:
                    self._close_position(asset, pos['tp'])
            else:  # Short
                if price >= pos['sl']:
                    self._close_position(asset, pos['sl'])
                elif price <= pos['tp']:
                    self._close_position(asset, pos['tp'])

    def _simulate_trade_outcome(self, asset):
        """
        PEEK & LABEL: Look ahead to see if trade hits SL or TP.
        OPTIMIZED: Uses cached numpy arrays instead of pandas slicing.
        Returns: (pnl, fast_win)
        """
        if self.positions[asset] is None:
            return 0.0, False
        
        pos = self.positions[asset]
        direction = pos['direction']
        sl = pos['sl']
        tp = pos['tp']
        
        # Look forward up to 1000 steps
        start_idx = self.current_step + 1
        end_idx = min(start_idx + 1000, len(self.raw_data))
        
        if start_idx >= end_idx:
            return 0.0, False
            
        # OPTIMIZATION: Use pre-cached numpy arrays
        lows = self.low_arrays[asset][start_idx:end_idx]
        highs = self.high_arrays[asset][start_idx:end_idx]
        
        if direction == 1:  # Long
            sl_hit_mask = lows <= sl
            tp_hit_mask = highs >= tp
        else:  # Short
            sl_hit_mask = highs >= sl
            tp_hit_mask = lows <= tp
        
        sl_hit = sl_hit_mask.any()
        tp_hit = tp_hit_mask.any()
        
        fast_win = False
        
        # Determine outcome
        # FIX: When both hit on same candle, assume SL first (conservative)
        if sl_hit and tp_hit:
            first_sl_idx = np.argmax(sl_hit_mask)
            first_tp_idx = np.argmax(tp_hit_mask)
            if first_sl_idx <= first_tp_idx:
                exit_price = sl
            else:
                exit_price = tp
                # Check for fast win (within 6 candles = 30 mins)
                if first_tp_idx < 6:
                    fast_win = True
        elif sl_hit:
            exit_price = sl
        elif tp_hit:
            exit_price = tp
            # Check for fast win
            first_tp_idx = np.argmax(tp_hit_mask)
            if first_tp_idx < 6:
                fast_win = True
        else:
            # Neither hit: use last available price from cached array
            exit_price = self.close_arrays[asset][end_idx - 1]
        
        # Calculate P&L (FIX: Use correct formula matching _close_position)
        price_change = (exit_price - pos['entry_price']) * direction
        price_change_pct = price_change / pos['entry_price'] if pos['entry_price'] != 0 else 0
        position_value = pos['size'] * self.leverage
        pnl = price_change_pct * position_value
        
        return pnl, fast_win

    def _simulate_trade_outcome_with_timing(self, asset):
        """
        PEEK & LABEL: Look ahead to see if trade hits SL or TP.
        Returns detailed dictionary with timing info.
        """
        if self.positions[asset] is None:
            return {'closed': False, 'pnl': 0.0, 'bars_held': 0, 'exit_reason': 'NO_POSITION'}
        
        pos = self.positions[asset]
        direction = pos['direction']
        sl = pos['sl']
        tp = pos['tp']
        
        # Look forward up to 1000 steps
        start_idx = self.current_step + 1
        end_idx = min(start_idx + 1000, len(self.raw_data))
        
        if start_idx >= end_idx:
            return {'closed': False, 'pnl': 0.0, 'bars_held': 0, 'exit_reason': 'END_OF_DATA'}
            
        # OPTIMIZATION: Use pre-cached numpy arrays
        lows = self.low_arrays[asset][start_idx:end_idx]
        highs = self.high_arrays[asset][start_idx:end_idx]
        
        if direction == 1:  # Long
            sl_hit_mask = lows <= sl
            tp_hit_mask = highs >= tp
        else:  # Short
            sl_hit_mask = highs >= sl
            tp_hit_mask = lows <= tp
        
        sl_hit = sl_hit_mask.any()
        tp_hit = tp_hit_mask.any()
        
        exit_reason = 'TIME'
        exit_idx = end_idx - 1
        closed = False
        
        # Determine outcome
        if sl_hit and tp_hit:
            first_sl_idx = np.argmax(sl_hit_mask)
            first_tp_idx = np.argmax(tp_hit_mask)
            if first_sl_idx <= first_tp_idx:
                exit_price = sl
                exit_reason = 'SL'
                exit_idx = start_idx + first_sl_idx
            else:
                exit_price = tp
                exit_reason = 'TP'
                exit_idx = start_idx + first_tp_idx
            closed = True
        elif sl_hit:
            exit_price = sl
            exit_reason = 'SL'
            exit_idx = start_idx + np.argmax(sl_hit_mask)
            closed = True
        elif tp_hit:
            exit_price = tp
            exit_reason = 'TP'
            exit_idx = start_idx + np.argmax(tp_hit_mask)
            closed = True
        else:
            # Neither hit: use last available price
            exit_price = self.close_arrays[asset][end_idx - 1]
            exit_reason = 'OPEN'
            closed = False
        
        # Calculate P&L
        price_change = (exit_price - pos['entry_price']) * direction
        price_change_pct = price_change / pos['entry_price'] if pos['entry_price'] != 0 else 0
        position_value = pos['size'] * self.leverage
        pnl = price_change_pct * position_value
        
        bars_held = exit_idx - self.current_step
        
        return {
            'closed': closed,
            'pnl': pnl,
            'bars_held': bars_held,
            'exit_reason': exit_reason
        }

    def _calculate_reward(self) -> float:
        """
        Reward function with separate modes for training and backtesting.
        
        Training Mode (is_training=True):
            - Uses PEEK & LABEL for credit assignment
            - Progressive drawdown penalty
            
        Backtesting Mode (is_training=False):
            - Uses actual realized P&L from completed trades
            - Reflects real portfolio performance
        """
        reward = 0.0
        
        # =====================================================================
        # BACKTESTING MODE: Use actual realized P&L
        # =====================================================================
        if not self.is_training:
            # Sum up actual P&L from completed trades this step
            step_pnl = sum(trade['net_pnl'] for trade in self.completed_trades)
            
            # Normalize: 1% of starting equity = 0.02 reward
            if step_pnl != 0:
                normalized_pnl = (step_pnl / self.start_equity) * 2.0
                reward += normalized_pnl
            
            return reward
        
        # =====================================================================
        # TRAINING MODE: PEEK & LABEL + Drawdown Penalty
        # =====================================================================
        
        # COMPONENT 1: Peeked P&L (Primary Signal)
        if self.peeked_pnl_step != 0:
            # Normalize: 1% of starting equity = 0.02 reward
            normalized_pnl = (self.peeked_pnl_step / self.start_equity) * 2.0
            
            # Loss Aversion (Prospect Theory): Losses hurt 2.25x more (1.5 * 1.5)
            if normalized_pnl < 0:
                normalized_pnl = np.clip(normalized_pnl, -5.0, 0.0) * 2.25
            else:
                # Relaxed clipping to allow for Fast Win Bonus (up to 5.0)
                normalized_pnl = np.clip(normalized_pnl, 0.0, 5.0)
            reward += normalized_pnl
        
        # COMPONENT 2: Progressive Drawdown Penalty
        drawdown = 1.0 - (self.equity / self.peak_equity)
        
        if drawdown > 0.05:
            severity = min((drawdown - 0.05) / 0.20, 1.0)
            penalty = -0.15 * (severity ** 1.5)
            reward += penalty
        
        # Track best step reward
        if reward > self.max_step_reward:
            self.max_step_reward = reward

        if self.current_step % self.REWARD_LOG_INTERVAL == 0:
            logging.debug(
                f"[Reward] step={self.current_step} "
                f"peeked={self.peeked_pnl_step:.2f} "
                f"drawdown={drawdown:.2%} "
                f"current={reward:.4f} "
                f"best_step={self.max_step_reward:.4f}"
            )
        
        return reward

    def _get_current_prices(self):
        """Get current close prices for all assets using cached arrays."""
        return {asset: self.close_arrays[asset][self.current_step] for asset in self.assets}

    def _get_current_atrs(self):
        """Get current ATR values for all assets using cached arrays."""
        return {asset: self.atr_arrays[asset][self.current_step] for asset in self.assets}
    
    def _get_current_timestamp(self):
        """Get timestamp for current step."""
        try:
            return self.processed_data.index[self.current_step]
        except (IndexError, KeyError):  # FIX: Specific exception handling
            from datetime import datetime, timedelta
            base_time = datetime(2025, 1, 1)
            return base_time + timedelta(minutes=self.current_step * 5)
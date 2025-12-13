import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from .frozen_feature_engine import FeatureEngine

class TradingEnv(gym.Env):
    """
    Trading environment for RL agent.
    
    Curriculum Stages:
        Stage 1: Direction only (5 outputs)
        Stage 2: Direction + Position sizing (10 outputs)
        Stage 3: Direction + Position sizing + SL/TP (20 outputs)
    
    Reward System:
        - Peeked P&L: Primary signal via PEEK & LABEL (solves credit assignment)
        - Drawdown Penalty: Progressive risk control
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir='data', stage=3, is_training=True):
        super(TradingEnv, self).__init__()
        
        self.data_dir = data_dir
        self.stage = stage
        self.is_training = is_training
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Load Data
        self.data = self._load_data()
        self.feature_engine = FeatureEngine()
        self.raw_data, self.processed_data = self.feature_engine.preprocess_data(self.data)
        
        # OPTIMIZATION: Cache data as numpy arrays for fast access
        self._cache_data_arrays()
        
        # Define Action Space based on Stage
        if self.stage == 1:
            self.action_dim = 5   # Direction only
        elif self.stage == 2:
            self.action_dim = 10  # Direction + Size
        else:
            self.action_dim = 20  # Direction + Size + SL/TP
            
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        # Define Observation Space (140 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(140,), dtype=np.float32)
        
        # State Variables
        self.current_step = 0
        self.max_steps = len(self.processed_data) - 1
        self.equity = 10000.0
        self.leverage = 100
        self.positions = {asset: None for asset in self.assets}
        self.portfolio_history = []
        
        # Reward Tracking (simplified)
        self.peeked_pnl_step = 0.0  # PEEK & LABEL: P&L assigned at trade open
        
        # Trade Tracking for Backtesting
        self.completed_trades = []
        self.all_trades = []
        
        # Initialization Safety
        self.start_equity = self.equity
        self.peak_equity = self.equity
        self.max_step_reward = -float('inf')
        
        
        # PRD Risk Constants
        self.MAX_POS_SIZE_PCT = 0.50
        self.MAX_TOTAL_EXPOSURE = 0.60
        self.DRAWDOWN_LIMIT = 0.25
        
        # Configuration Constants
        self.MIN_POSITION_SIZE = 10  # Minimum position size in equity units
        self.MIN_ATR_MULTIPLIER = 0.0001  # Fallback if ATR is zero (0.01%)
        self.REWARD_LOG_INTERVAL = 5000  # Steps between reward logging
        
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
        for asset in self.assets:
            file_path = f"{self.data_dir}/{asset}_5m.parquet"
            file_path_2025 = f"{self.data_dir}/{asset}_5m_2025.parquet"
            
            
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
                    
                    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
                    # FIX: Match column naming convention expected by _get_current_prices/atrs
                    df = pd.DataFrame(index=dates)
                    df[f"{asset}_open"] = base_price
                    df[f"{asset}_high"] = base_price * 1.001
                    df[f"{asset}_low"] = base_price * 0.999
                    df[f"{asset}_close"] = base_price
                    df[f"{asset}_volume"] = 100
                    df[f"{asset}_atr_14"] = self.MIN_ATR_MULTIPLIER * base_price  # Default small ATR
                    
            
            data[asset] = df
        return data

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if self.is_training:
            # Training: Randomize for diversity
            self.equity = np.random.uniform(5000.0, 15000.0)
            self.leverage = 100
            self.current_step = np.random.randint(500, self.max_steps - 288)
        else:
            # Backtesting: Fixed equity, randomize start point
            self.equity = 10000.0
            self.leverage = 100
            # Backtesting: Start from beginning to cover full dataset
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
        
        # Parse and execute trades
        parsed_actions = self._parse_action(action)
        self._execute_trades(parsed_actions)
        
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
            'timestamp': self._get_current_timestamp()
        }
        
        return self._validate_observation(self._get_observation()), reward, terminated, truncated, info

    def _parse_action(self, action):
        """Parse raw action array into per-asset trading decisions."""
        # FIX: Validate action shape
        if len(action) != self.action_dim:
            raise ValueError(f"Action array has {len(action)} elements, expected {self.action_dim}")
            
        parsed = {}
        for i, asset in enumerate(self.assets):
            if self.stage == 1:
                # Stage 1: Direction only
                direction_raw = action[i]
                parsed[asset] = {
                    'direction': 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0),
                    'size': 0.25,      # Fixed size
                    'sl_mult': 1.5,    # Fixed SL
                    'tp_mult': 2.5     # Fixed TP
                }
            elif self.stage == 2:
                # Stage 2: Direction + Position Size
                base_idx = i * 2
                direction_raw = action[base_idx]
                size_raw = np.clip((action[base_idx + 1] + 1) / 2, 0, 1)  # FIX: Bounds validation
                parsed[asset] = {
                    'direction': 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0),
                    'size': size_raw,
                    'sl_mult': 1.5,    # Fixed SL
                    'tp_mult': 2.5     # Fixed TP
                }
            else:
                # Stage 3: Direction + Size + SL/TP
                base_idx = i * 4
                direction_raw = action[base_idx]
                size_raw = np.clip((action[base_idx + 1] + 1) / 2, 0, 1)  # FIX: Bounds validation
                sl_raw = np.clip((action[base_idx + 2] + 1) / 2 * 2.5 + 0.5, 0.5, 3.0)  # 0.5 to 3.0
                tp_raw = np.clip((action[base_idx + 3] + 1) / 2 * 3.5 + 1.5, 1.5, 5.0)  # 1.5 to 5.0
                parsed[asset] = {
                    'direction': 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0),
                    'size': size_raw,
                    'sl_mult': sl_raw,
                    'tp_mult': tp_raw
                }
        return parsed

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
        simulated_pnl = self._simulate_trade_outcome(asset)
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
        """
        if self.positions[asset] is None:
            return 0.0
        
        pos = self.positions[asset]
        direction = pos['direction']
        sl = pos['sl']
        tp = pos['tp']
        
        # Look forward up to 1000 steps
        start_idx = self.current_step + 1
        end_idx = min(start_idx + 1000, len(self.raw_data))
        
        if start_idx >= end_idx:
            return 0.0
            
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
        
        # Determine outcome
        # FIX: When both hit on same candle, assume SL first (conservative)
        if sl_hit and tp_hit:
            first_sl_idx = np.argmax(sl_hit_mask)
            first_tp_idx = np.argmax(tp_hit_mask)
            exit_price = sl if first_sl_idx <= first_tp_idx else tp
        elif sl_hit:
            exit_price = sl
        elif tp_hit:
            exit_price = tp
        else:
            # Neither hit: use last available price from cached array
            exit_price = self.close_arrays[asset][end_idx - 1]
        
        # Calculate P&L (FIX: Use correct formula matching _close_position)
        price_change = (exit_price - pos['entry_price']) * direction
        price_change_pct = price_change / pos['entry_price'] if pos['entry_price'] != 0 else 0
        position_value = pos['size'] * self.leverage
        pnl = price_change_pct * position_value
        
        return pnl

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
            
            # Normalize: 1% of starting equity = 0.1 reward
            if step_pnl != 0:
                normalized_pnl = (step_pnl / self.start_equity) * 10.0
                reward += normalized_pnl
            
            return reward
        
        # =====================================================================
        # TRAINING MODE: PEEK & LABEL + Drawdown Penalty
        # =====================================================================
        
        # COMPONENT 1: Peeked P&L (Primary Signal)
        if self.peeked_pnl_step != 0:
            # Normalize: 1% of starting equity = 0.1 reward
            normalized_pnl = (self.peeked_pnl_step / self.start_equity) * 10.0
            
            # Loss Aversion (Prospect Theory): Losses hurt 1.5x more
            if normalized_pnl < 0:
                normalized_pnl = np.clip(normalized_pnl, -1.0, 0.0) * 1.5
            else:
                normalized_pnl = np.clip(normalized_pnl, 0.0, 1.0)
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

    def _get_observation(self):
        """Build observation vector from market data and portfolio state."""
        current_row = self.processed_data.iloc[self.current_step]
        
        # FIX: Calculate actual margin usage and correct drawdown
        total_exposure = sum(pos['size'] for pos in self.positions.values() if pos is not None)
        
        portfolio_state = {
            'equity': self.equity,
            'margin_usage_pct': total_exposure / self.equity if self.equity > 0 else 0,
            'drawdown': 1 - (self.equity / self.peak_equity),  # FIX: Inverted formula
            'num_open_positions': sum(1 for p in self.positions.values() if p is not None)
        }
        
        current_prices = self._get_current_prices()
        for asset in self.assets:
            pos = self.positions[asset]
            if pos:
                # FIX: Calculate normalized unrealized P&L
                price_change = (current_prices[asset] - pos['entry_price']) * pos['direction']
                price_change_pct = price_change / pos['entry_price'] if pos['entry_price'] != 0 else 0
                unrealized_pnl = price_change_pct * (pos['size'] * self.leverage)
                
                portfolio_state[asset] = {
                    'has_position': 1,
                    'position_size': pos['size'] / self.equity,
                    'unrealized_pnl': unrealized_pnl,
                    'position_age': self.current_step - pos['entry_step'],
                    'entry_price': pos['entry_price'],
                    'current_sl': pos['sl'],
                    'current_tp': pos['tp']
                }
            else:
                portfolio_state[asset] = {
                    'has_position': 0,
                    'position_size': 0,
                    'unrealized_pnl': 0,
                    'position_age': 0,
                    'entry_price': 0,
                    'current_sl': 0,
                    'current_tp': 0
                }
        
        return self.feature_engine.get_observation(current_row, portfolio_state)

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

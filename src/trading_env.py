import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from .feature_engine import FeatureEngine

class TradingEnv(gym.Env):
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
        
        # Define Action Space based on Stage
        if self.stage == 1:
            self.action_dim = 5
        elif self.stage == 2:
            self.action_dim = 10
        else:
            self.action_dim = 20
            
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        
        # Define Observation Space (140 features)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(140,), dtype=np.float32)
        
        # State Variables
        self.current_step = 0
        self.max_steps = len(self.processed_data) - 1
        self.equity = 10000.0
        self.positions = {asset: None for asset in self.assets} # None or dict
        self.portfolio_history = []
        
        # Tracking for Rewards
        self.realized_pnl_step = 0.0
        self.transaction_costs_step = 0.0
        self.new_entry_bonus = 0.0
        
        # Tracking for Backtesting
        self.completed_trades = []  # Trades completed in current step
        self.all_trades = []  # All trades for the episode
        
        # ADDED: Churn/Whipsaw tracking to penalize rapid trading
        self.last_close_step = {asset: -100 for asset in self.assets}  # Step when last closed
        self.trades_opened_this_step = 0  # Count of new trades this step
        
        # PRD Constants
        self.MAX_POS_SIZE_PCT = 0.50
        self.MAX_TOTAL_EXPOSURE = 0.60
        self.DRAWDOWN_LIMIT = 0.25
        
    def _load_data(self):
        data = {}
        for asset in self.assets:
            # Try regular file first, then 2025 backtest file
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
                    dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
                    df = pd.DataFrame({
                        'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 100
                    }, index=dates)
            
            data[asset] = df
        return data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.is_training:
            # Training: Randomize everything for diversity
            self.equity = np.random.choice([100.0, 1000.0, 10000.0])
            self.leverage = np.random.choice([100, 200, 500])
            self.current_step = np.random.randint(100, self.max_steps - 288)
        else:
            # Backtesting: Fixed equity/leverage but randomize starting point
            # This tests the model across different market conditions in 2025
            self.equity = 10000.0
            self.leverage = 100
            # Randomize starting point for backtesting diversity
            # Use at least 500 steps for indicator warmup, leave 288 steps for episode
            if self.max_steps > 788:  # 500 warmup + 288 episode
                self.current_step = np.random.randint(500, self.max_steps - 288)
            else:
                self.current_step = 100  # Fallback for small datasets
            
        self.start_equity = self.equity
        self.peak_equity = self.equity
        self.positions = {asset: None for asset in self.assets}
        self.portfolio_history = []
        
        # Reset Reward Trackers
        self.realized_pnl_step = 0.0
        self.transaction_costs_step = 0.0
        self.new_entry_bonus = 0.0
        
        # Reset Trade Tracking
        self.completed_trades = []
        self.all_trades = []
        
        # ADDED: Reset churn tracking
        self.last_close_step = {asset: -100 for asset in self.assets}
        self.trades_opened_this_step = 0
        
        return self._get_observation(), {}

    def step(self, action):
        # Reset step-specific reward trackers
        self.realized_pnl_step = 0.0
        self.transaction_costs_step = 0.0
        self.new_entry_bonus = 0.0
        self.completed_trades = []  # Clear trades from previous step
        self.trades_opened_this_step = 0  # ADDED: Reset trade counter
        
        # 1. Parse Action
        parsed_actions = self._parse_action(action)
        
        # 2. Execute Trades
        self._execute_trades(parsed_actions)
        
        # 3. Update Market Data
        self.current_step += 1
        
        # 4. Update Positions (Mark to Market)
        self._update_positions()
        
        # 5. Calculate Reward
        reward = self._calculate_reward()
        
        # 6. Check Termination
        terminated = False
        truncated = False
        
        drawdown = (self.equity / self.peak_equity) - 1
        if drawdown < -self.DRAWDOWN_LIMIT:
            terminated = True
            # Reward penalty is handled in _calculate_reward
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # 7. Prepare info dict for backtesting
        info = {
            'trades': self.completed_trades,
            'equity': self.equity,
            'timestamp': self._get_current_timestamp()
        }
        
        return self._get_observation(), reward, terminated, truncated, info

    def _parse_action(self, action):
        parsed = {}
        for i, asset in enumerate(self.assets):
            if self.stage == 1:
                direction_raw = action[i]
                parsed[asset] = {
                    'direction': 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0),
                    'size': 0.25,
                    'sl_mult': 1.5,
                    'tp_mult': 2.5
                }
            elif self.stage == 2:
                base_idx = i * 2
                direction_raw = action[base_idx]
                size_raw = (action[base_idx+1] + 1) / 2
                parsed[asset] = {
                    'direction': 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0),
                    'size': size_raw,
                    'sl_mult': 1.5,
                    'tp_mult': 2.5
                }
            else:
                base_idx = i * 4
                direction_raw = action[base_idx]
                size_raw = (action[base_idx+1] + 1) / 2
                sl_raw = (action[base_idx+2] + 1) / 2 * 2.5 + 0.5
                tp_raw = (action[base_idx+3] + 1) / 2 * 3.5 + 1.5
                parsed[asset] = {
                    'direction': 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0),
                    'size': size_raw,
                    'sl_mult': sl_raw,
                    'tp_mult': tp_raw
                }
        return parsed

    def _execute_trades(self, actions):
        current_prices = self._get_current_prices()
        atrs = self._get_current_atrs()
        
        for asset, act in actions.items():
            direction = act['direction']
            current_pos = self.positions[asset]
            price = current_prices[asset]
            atr = atrs[asset]
            
            if current_pos is None:
                if direction != 0:
                    self._open_position(asset, direction, act, price, atr)
            
            elif current_pos['direction'] == direction:
                pass 
            
            elif direction != 0 and current_pos['direction'] != direction:
                self._close_position(asset, price)
                self._open_position(asset, direction, act, price, atr)
                
            elif direction == 0 and current_pos is not None:
                self._close_position(asset, price)

    def _check_global_exposure(self, new_position_size):
        """
        Checks if adding a new position would exceed the global exposure limit.
        PRD 2.2: Max Total Exposure: 60% of equity
        """
        current_exposure = 0.0
        for pos in self.positions.values():
            if pos is not None:
                current_exposure += pos['size'] # pos['size'] is the margin used? No, let's clarify.
                # In _open_position: 'size': size_pct * self.equity
                # This is the DOLLAR AMOUNT allocated to the trade (Margin Used * Leverage? Or just Margin?)
                # PRD says: "Max Position Size: 50% of equity per asset"
                # "Max Total Exposure: 60% of equity across all assets"
                # Usually Exposure = Position Value = Margin * Leverage.
                # But here "50% of equity" suggests it's about Margin Used or Capital Allocated.
                # Let's assume 'size' stored is the Capital Allocated (Margin).
                
        # Total Capital Allocated + New Capital
        total_allocated = current_exposure + new_position_size
        
        if total_allocated > (self.equity * self.MAX_TOTAL_EXPOSURE):
            return False
        return True

    def _open_position(self, asset, direction, act, price, atr):
        # Risk Validation 1: Max Position Size (Per Asset)
        size_pct = act['size'] * self.MAX_POS_SIZE_PCT
        position_size = size_pct * self.equity # Capital to allocate
        
        # Risk Validation 2: Margin Check
        margin_required = position_size # Since we treat position_size as margin
        if margin_required > self.equity * 0.9: 
            return # Reject
            
        # Risk Validation 3: Global Exposure Check (PRD Fix)
        if not self._check_global_exposure(position_size):
            return # Reject
            
        sl_dist = act['sl_mult'] * atr
        tp_dist = act['tp_mult'] * atr
        
        sl = price - (direction * sl_dist)
        tp = price + (direction * tp_dist)
        
        # Risk-Reward Quality Bonus (PRD 5.2)
        rr_ratio = tp_dist / sl_dist
        if rr_ratio >= 2.0:
            self.new_entry_bonus += 0.5
        elif rr_ratio >= 1.5:
            self.new_entry_bonus += 0.2
        else:
            self.new_entry_bonus -= 0.3
        
        self.positions[asset] = {
            'direction': direction,
            'entry_price': price,
            'size': position_size,
            'sl': sl,
            'tp': tp,
            'entry_step': self.current_step,
            'tp_dist': tp_dist,
            'sl_dist': sl_dist
        }
        
        # ADDED: Track trade opens for churn detection
        self.trades_opened_this_step += 1
        
        # Deduct Transaction Costs (PRD 5.2)
        # Spread + Commission. Simplified as 0.0001 (1 pip) * Position Value
        # Position Value = Margin * Leverage
        pos_value = position_size * self.leverage
        cost = pos_value * 0.0001 
        
        self.equity -= cost
        self.transaction_costs_step += cost

    def _close_position(self, asset, price):
        pos = self.positions[asset]
        if pos is None: return
        
        # Store equity before trade
        equity_before = self.equity
        
        # Calculate PnL - FIXED FORMULA
        # For forex/commodities: P&L = (Exit Price - Entry Price) * Position Size * Direction
        # Position Size here is in units (lots), not dollar value
        # We need to convert position_size (dollars) to units
        
        # Simplified: P&L = price_change_pct * position_value
        price_change = (price - pos['entry_price']) * pos['direction']
        price_change_pct = price_change / pos['entry_price']
        position_value = pos['size'] * self.leverage
        pnl = price_change_pct * position_value
        
        # Alternative simpler calculation (more intuitive):
        # pnl = price_change * (position_value / pos['entry_price'])
        # This gives: dollars_per_pip * number_of_pips
        
        self.equity += pnl
        self.realized_pnl_step += pnl
        
        # Deduct Transaction Costs (Closing also incurs costs)
        # PRD says "Deducted on every trade execution". Usually means entry and exit.
        cost = position_value * 0.0001
        self.equity -= cost
        self.transaction_costs_step += cost
        
        # Record completed trade for backtesting
        hold_time = (self.current_step - pos['entry_step']) * 5  # 5 minutes per step
        
        # Calculate net P&L (after fees)
        net_pnl = pnl - cost
        
        trade_record = {
            'timestamp': self._get_current_timestamp(),
            'asset': asset,
            'action': 'BUY' if pos['direction'] == 1 else 'SELL',
            'size': pos['size'],
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'sl': pos['sl'],
            'tp': pos['tp'],
            'pnl': pnl,  # Gross P&L
            'net_pnl': net_pnl,  # Net P&L after fees
            'fees': cost,
            'equity_before': equity_before,
            'equity_after': self.equity,
            'equity': self.equity,
            'hold_time': hold_time,
            'rr_ratio': pos['tp_dist'] / pos['sl_dist'] if pos['sl_dist'] > 0 else 0,
            'price_change': price_change,
            'price_change_pct': price_change_pct * 100  # As percentage
        }
        
        self.completed_trades.append(trade_record)
        self.all_trades.append(trade_record)
        
        # ADDED: Track when position was closed for churn detection
        self.last_close_step[asset] = self.current_step
        
        self.positions[asset] = None

    def _update_positions(self):
        current_prices = self._get_current_prices()
        
        for asset, pos in self.positions.items():
            if pos is None: continue
            
            price = current_prices[asset]
            
            if pos['direction'] == 1: # Buy
                if price <= pos['sl'] or price >= pos['tp']:
                    self._close_position(asset, price)
            else: # Sell
                if price >= pos['sl'] or price <= pos['tp']:
                    self._close_position(asset, price)

    def _calculate_reward(self):
        """
        PRD Section 5: Reward Function
        
        UPDATED with additional penalties/bonuses:
        - Churn penalty: Penalize rapid open→close→reopen (whipsaw trading)
        - Session hold bonus: Reward patient holding during active sessions
        - Stronger holding penalty: Increased from 0.01 to 0.02
        """
        
        # 1. Realized PnL (Normalized)
        realized = self.realized_pnl_step / self.start_equity
        
        # 2. Unrealized PnL (Discounted)
        unrealized_pnl_total = 0.0
        current_prices = self._get_current_prices()
        for asset, pos in self.positions.items():
            if pos:
                price = current_prices[asset]
                diff = (price - pos['entry_price']) * pos['direction']
                val = pos['size'] * self.leverage
                pnl = (diff / pos['entry_price']) * val
                unrealized_pnl_total += pnl
        
        unrealized = (unrealized_pnl_total * 0.2) / self.start_equity
        
        # 3. Risk-Reward Quality
        rr_quality = self.new_entry_bonus
        
        # 4. Transaction Costs (Normalized)
        costs = self.transaction_costs_step / self.start_equity
        
        # 5. Drawdown Penalty (REDUCED for value stability)
        # NOTE: Previous values (-10, -1, -0.3) were too extreme relative to 
        # other reward components (~0.001 to 0.1), making value estimation impossible.
        # The termination at -25% drawdown is the main safeguard, not the penalty.
        drawdown = (self.equity / self.peak_equity) - 1
        drawdown_penalty = 0.0
        if drawdown < -0.25:
            drawdown_penalty = -2.0    # REDUCED: Was -10.0 (too extreme)
        elif drawdown < -0.15:
            drawdown_penalty = -0.5    # REDUCED: Was -1.0
        elif drawdown < -0.10:
            drawdown_penalty = -0.2    # REDUCED: Was -0.3
            
        # 6. Holding Penalty (INCREASED from 0.01 to 0.02)
        # Stronger penalty to encourage more selective trading
        num_open = sum(1 for p in self.positions.values() if p is not None)
        avg_age = 0
        if num_open > 0:
            total_age = sum(self.current_step - p['entry_step'] for p in self.positions.values() if p is not None)
            avg_age = total_age / num_open
            
        holding_penalty = -0.02 * num_open * (avg_age / 100.0)  # CHANGED: Was -0.01
        
        # 7. Session Penalty/Bonus
        session_penalty = 0.0
        session_bonus = 0.0
        current_row = self.processed_data.iloc[self.current_step]
        
        is_london = current_row['session_london']
        is_ny = current_row['session_ny']
        is_overlap = current_row['session_overlap']
        is_active_session = is_london or is_ny or is_overlap
        
        # Penalty for trading outside sessions
        if self.transaction_costs_step > 0:
            if not is_active_session:
                session_penalty = -0.5
        
        # ADDED: Bonus for patient holding during active sessions
        # Reward the agent for maintaining positions during liquid market hours
        if num_open > 0 and is_active_session:
            session_bonus = 0.01 * num_open  # Small reward for patience
        
        # 8. ADDED: Churn/Whipsaw Penalty
        # Penalize rapid open→close→reopen patterns (within 3 steps = 15 minutes)
        churn_penalty = 0.0
        if self.trades_opened_this_step > 0:
            for asset in self.assets:
                # Check if we closed this asset recently and reopened
                steps_since_close = self.current_step - self.last_close_step[asset]
                if steps_since_close <= 3:  # Within 15 minutes
                    # We opened a new trade right after closing
                    # Check if we opened on this asset this step
                    if self.positions[asset] is not None and self.positions[asset]['entry_step'] == self.current_step:
                        churn_penalty -= 0.3  # Penalty per churned asset
        
        # 9. ADDED: Overtrading Penalty
        # Penalize opening too many trades at once
        if self.trades_opened_this_step > 2:
            overtrading_penalty = -0.2 * (self.trades_opened_this_step - 2)
        else:
            overtrading_penalty = 0.0
                
        # Total Reward
        total_reward = (
            realized 
            + unrealized 
            + rr_quality 
            - costs 
            + drawdown_penalty 
            + holding_penalty 
            + session_penalty
            + session_bonus       # ADDED
            + churn_penalty       # ADDED
            + overtrading_penalty # ADDED
        )
        
        return total_reward

    def _get_observation(self):
        current_row = self.processed_data.iloc[self.current_step]
        
        portfolio_state = {
            'equity': self.equity,
            'margin_usage_pct': 0, # Simplified
            'drawdown': (self.equity / self.peak_equity) - 1,
            'num_open_positions': sum(1 for p in self.positions.values() if p is not None)
        }
        
        current_prices = self._get_current_prices()
        for asset in self.assets:
            pos = self.positions[asset]
            if pos:
                portfolio_state[asset] = {
                    'has_position': 1,
                    'position_size': pos['size'] / self.equity,
                    'unrealized_pnl': (current_prices[asset] - pos['entry_price']) * pos['direction'], 
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
        row = self.raw_data.iloc[self.current_step]
        prices = {}
        for asset in self.assets:
            prices[asset] = row[f"{asset}_close"]
        return prices

    def _get_current_atrs(self):
        row = self.raw_data.iloc[self.current_step]
        atrs = {}
        for asset in self.assets:
            atrs[asset] = row[f"{asset}_atr_14"]
        return atrs
    
    def _get_current_timestamp(self):
        """Get timestamp for current step"""
        try:
            return self.processed_data.index[self.current_step]
        except:
            # Fallback if index is not datetime
            from datetime import datetime, timedelta
            base_time = datetime(2025, 1, 1)
            return base_time + timedelta(minutes=self.current_step * 5)


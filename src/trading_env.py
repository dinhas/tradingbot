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
        self.leverage = 100  # BUG FIX #1: Initialize in __init__ (overwritten in reset)
        self.positions = {asset: None for asset in self.assets} # None or dict
        self.portfolio_history = []
        
        # Tracking for Rewards
        self.realized_pnl_step = 0.0
        self.peeked_pnl_step = 0.0  # PEEK & LABEL: Reward given at trade open
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
            self.equity = np.random.uniform(5000.0, 15000.0)
            self.leverage = 100
            self.current_step = np.random.randint(500, self.max_steps - 288)
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
        self.peeked_pnl_step = 0.0  # PEEK & LABEL
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
        self.peeked_pnl_step = 0.0  # PEEK & LABEL
        self.transaction_costs_step = 0.0
        self.new_entry_bonus = 0.0
        self.completed_trades = []  # Clear trades from previous step
        self.trades_opened_this_step = 0  # ADDED: Reset trade counter
        
        # 1. Parse Action
        parsed_actions = self._parse_action(action)
        
        # 2. Execute Trades
        self._execute_trades(parsed_actions)
        
        # Check for margin call (account blown up)
        if self.equity <= 0:
            self.equity = 0.01
            terminated = True
            reward = -0.5
            info = {
                'trades': self.completed_trades,
                'equity': self.equity,
                'timestamp': self._get_current_timestamp(),
                'termination_reason': 'margin_call'
            }
            return self._get_observation(), reward, terminated, truncated, info
        
        # 3. Update Market Data
        self.current_step += 1
        
        # 4. Update Positions (Mark to Market)
        self._update_positions()
        
        # 5. Calculate Reward
        reward = self._calculate_reward()
        
        # 6. Check Termination
        terminated = False
        truncated = False
        
        # BUG FIX #3: Update peak_equity BEFORE calculating drawdown
        self.peak_equity = max(self.peak_equity, self.equity)
        
        drawdown = (self.equity / self.peak_equity) - 1
        if drawdown < -self.DRAWDOWN_LIMIT:
            terminated = True
            # Reward penalty is handled in _calculate_reward
            
        if self.current_step >= self.max_steps:
            truncated = True
        
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

        # Optimization #2: Add Position Value Validation
        if position_size < 10:  # Minimum $10 position
            return
        if position_size > self.equity * 0.5:  # Sanity check
            position_size = self.equity * 0.5
        
        # Risk Validation 2: Margin Check
        # REMOVED: Redundant check (Issue #4) fixed
        # if margin_required > self.equity * 0.9: 
        #     return # Reject
            
        # Risk Validation 3: Global Exposure Check (PRD Fix)
        if not self._check_global_exposure(position_size):
            return # Reject
            
        sl_dist = act['sl_mult'] * atr
        tp_dist = act['tp_mult'] * atr
        
        sl = price - (direction * sl_dist)
        tp = price + (direction * tp_dist)
        
        # Risk-Reward Quality Bonus (PRD 5.2)
        # REDUCED: Make smaller than profit signal so trading P&L dominates
        rr_ratio = tp_dist / sl_dist
        if rr_ratio >= 2.0:
            self.new_entry_bonus += 0.005  # REDUCED: Was 0.02
        elif rr_ratio >= 1.5:
            self.new_entry_bonus += 0.002  # REDUCED: Was 0.01
        else:
            self.new_entry_bonus -= 0.002  # REDUCED: Was -0.01
        
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
        
        # PEEK & LABEL: Simulate trade outcome and assign reward NOW
        simulated_pnl = self._simulate_trade_outcome(asset)
        self.peeked_pnl_step += simulated_pnl
        
        # Deduct Transaction Costs (PRD 5.2)
        # Spread + Commission. FIXED: 0.00002 (~0.2 pips)
        # Position Value = Margin * Leverage
        pos_value = position_size * self.leverage
        cost = pos_value * 0.00002 
        
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
        # BUG FIX #4: Division by zero protection
        price_change_pct = price_change / pos['entry_price'] if pos['entry_price'] != 0 else 0
        position_value = pos['size'] * self.leverage
        pnl = price_change_pct * position_value
        
        # Alternative simpler calculation (more intuitive):
        # pnl = price_change * (position_value / pos['entry_price'])
        # This gives: dollars_per_pip * number_of_pips
        
        self.equity += pnl
        # PEEK & LABEL: Don't add to realized_pnl_step - reward was given at open time
        # self.realized_pnl_step += pnl  # REMOVED
        
        # Deduct Transaction Costs (Closing also incurs costs)
        # PRD says "Deducted on every trade execution". Usually means entry and exit.
        cost = position_value * 0.00002
        self.equity -= cost
        self.transaction_costs_step += cost
        
        # BUG FIX #6: Prevent equity from going negative
        if self.equity <= 0:
            self.equity = 0.01  # Minimum equity to avoid NaN
        
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
            
            # BUG FIX #5: Use SL/TP prices for exit, not gap price
            # This ensures R:R ratio is honored even with price gaps
            if pos['direction'] == 1:  # Buy
                if price <= pos['sl']:
                    self._close_position(asset, pos['sl'])  # Close at SL price
                elif price >= pos['tp']:
                    self._close_position(asset, pos['tp'])  # Close at TP price
            else:  # Sell
                if price >= pos['sl']:
                    self._close_position(asset, pos['sl'])  # Close at SL price
                elif price <= pos['tp']:
                    self._close_position(asset, pos['tp'])  # Close at TP price

    def _simulate_trade_outcome(self, asset):
        """
        PEEK & LABEL: Look ahead in data to see if the trade hits SL or TP.
        Returns the simulated P&L (in dollars).
        """
        if self.positions[asset] is None: return 0.0
        
        pos = self.positions[asset]
        direction = pos['direction']
        sl = pos['sl']
        tp = pos['tp']
        
        # Look forward up to 1000 steps (or end of data)
        # Note: This is computationally expensive but accurate for training
        start_idx = self.current_step + 1
        end_idx = min(start_idx + 1000, len(self.raw_data))
        
        future_data = self.raw_data.iloc[start_idx:end_idx]
        
        # Vectorized check for speed
        if direction == 1: # Long
            # Get arrays for speed
            # FAST: Convert to numpy only once
            lows = future_data[f"{asset}_low"].values
            highs = future_data[f"{asset}_high"].values
            
            # Find indices where SL or TP are hit
            sl_hit_mask = lows <= sl
            tp_hit_mask = highs >= tp
            
            # Check if any hit occurred
            sl_hit = sl_hit_mask.any()
            tp_hit = tp_hit_mask.any()
            
            if sl_hit and tp_hit:
                # Both hit, see which happened first
                first_sl_idx = np.argmax(sl_hit_mask)
                first_tp_idx = np.argmax(tp_hit_mask)
                
                if first_sl_idx <= first_tp_idx:
                     return (sl - pos['entry_price']) * (pos['size'] * self.leverage / pos['entry_price'])
                else:
                     return (tp - pos['entry_price']) * (pos['size'] * self.leverage / pos['entry_price'])
            
            elif sl_hit:
                 return (sl - pos['entry_price']) * (pos['size'] * self.leverage / pos['entry_price'])
            elif tp_hit:
                 return (tp - pos['entry_price']) * (pos['size'] * self.leverage / pos['entry_price'])
                 
        else: # Short
            lows = future_data[f"{asset}_low"].values
            highs = future_data[f"{asset}_high"].values
            
            sl_hit_mask = highs >= sl
            tp_hit_mask = lows <= tp
            
            sl_hit = sl_hit_mask.any()
            tp_hit = tp_hit_mask.any()
            
            if sl_hit and tp_hit:
                first_sl_idx = np.argmax(sl_hit_mask)
                first_tp_idx = np.argmax(tp_hit_mask)
                
                if first_sl_idx <= first_tp_idx:
                     pnl = (sl - pos['entry_price']) * (pos['size'] * self.leverage / pos['entry_price'])
                     return pnl * -1
                else:
                     pnl = (tp - pos['entry_price']) * (pos['size'] * self.leverage / pos['entry_price'])
                     return pnl * -1

            elif sl_hit:
                 pnl = (sl - pos['entry_price']) * (pos['size'] * self.leverage / pos['entry_price'])
                 return pnl * -1
            elif tp_hit:
                 pnl = (tp - pos['entry_price']) * (pos['size'] * self.leverage / pos['entry_price'])
                 return pnl * -1
                    
        # If neither hit in 1000 steps, return current floating P&L at step 1000
        # or 0 if we reached end of data
        if len(future_data) > 0:
            last_price = future_data.iloc[-1][f"{asset}_close"]
            price_change = (last_price - pos['entry_price']) * direction
            pnl = price_change * (pos['size'] * self.leverage / pos['entry_price'])
            return pnl
            
        return 0.0

    def _calculate_reward(self):
        """
        PRD Section 5: Reward Function
        
        PEEK & LABEL UPDATE:
        - Rewards are assigned at trade OPEN time based on simulated outcome.
        - Unrealized PnL is REMOVED (too noisy).
        - Realized PnL logic moved to trade open (peeked).
        """
        
        # 1. Peeked PnL (The Truth Signal)
        # Normalized by starting equity
        peeked_reward = self.peeked_pnl_step / self.start_equity
        
        # 2. Unrealized PnL - REMOVED (Noise)
        # unrealized = (unrealized_pnl_total * 0.5) / self.start_equity
        
        # 3. Risk-Reward Quality (already scaled down in _open_position)
        rr_quality = self.new_entry_bonus
        
        # 4. Transaction Costs - REMOVED FROM REWARD (already in pnl/equity)
        
        # 5. PROGRESSIVE Drawdown Penalty
        drawdown = (self.equity / self.peak_equity) - 1
        drawdown_penalty = 0.0
        if drawdown < -0.40:
            drawdown_penalty = -0.05
        elif drawdown < -0.30:
            drawdown_penalty = -0.02
        elif drawdown < -0.20:
            drawdown_penalty = -0.01
        elif drawdown < -0.10:
            drawdown_penalty = -0.005
            
        # 6. PROFIT-AWARE Holding Penalty/Bonus
        # STILL USEFUL: Encourage holding winners, cutting losers (active management)
        # Even with peeked rewards, this helps the agent learn to manage the trade *while it's open*
        # (e.g., if it didn't hit TP/SL yet)
        num_open = sum(1 for p in self.positions.values() if p is not None)
        holding_adjustment = 0.0
        current_prices = self._get_current_prices() # Needed for holding calc
        
        if num_open > 0:
            for asset, pos in self.positions.items():
                if pos is not None:
                    price = current_prices[asset]
                    diff = (price - pos['entry_price']) * pos['direction']
                    position_age = (self.current_step - pos['entry_step']) / 100.0
                    
                    if diff > 0:  # Winning position
                        holding_adjustment += 0.001 * position_age 
                    else:  # Losing position
                        holding_adjustment -= 0.0015 * position_age 
        
        # 7. Session Penalty/Bonus
        session_penalty = 0.0
        session_bonus = 0.0
        current_row = self.processed_data.iloc[self.current_step]
        
        is_london = current_row['session_london']
        is_ny = current_row['session_ny']
        is_overlap = current_row['session_overlap']
        is_active_session = is_london or is_ny or is_overlap
        
        if self.transaction_costs_step > 0:
            if not is_active_session:
                session_penalty = -0.001
        
        if num_open > 0 and is_active_session:
            session_bonus = 0.0005 * num_open
        
        # 8. Churn/Whipsaw Penalty
        churn_penalty = 0.0
        if self.trades_opened_this_step > 0:
            for asset in self.assets:
                steps_since_close = self.current_step - self.last_close_step[asset]
                if steps_since_close <= 3:
                     if self.positions[asset] is not None and self.positions[asset]['entry_step'] == self.current_step - 1:
                        churn_penalty -= 0.0015
        
        # 9. Overtrading Penalty
        if self.trades_opened_this_step > 2:
            overtrading_penalty = -0.0005 * (self.trades_opened_this_step - 2)
        else:
            overtrading_penalty = 0.0
                
        # Total Reward
        total_reward = (
            peeked_reward         # Primary signal: future outcome of action
            # + unrealized        # REMOVED
            + rr_quality          # Entry quality
            + drawdown_penalty    # Risk control
            + holding_adjustment  # Active management
            + session_penalty     
            + session_bonus       
            + churn_penalty       
            + overtrading_penalty 
        )
        
        # Logging
        if self.current_step % 10000 == 0:
            logging.info(
                f"Step {self.current_step} Reward Breakdown: "
                f"peeked_pnl={peeked_reward:.6f}, "
                f"rr_quality={rr_quality:.4f}, drawdown={drawdown_penalty:.4f}, "
                f"holding={holding_adjustment:.4f}, session_pen={session_penalty:.4f}, "
                f"TOTAL={total_reward:.6f}"
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


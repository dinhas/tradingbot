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
        
        # PRD Constants
        self.MAX_POS_SIZE_PCT = 0.50
        self.MAX_TOTAL_EXPOSURE = 0.60
        self.DRAWDOWN_LIMIT = 0.25
        
    def _load_data(self):
        data = {}
        for asset in self.assets:
            file_path = f"{self.data_dir}/{asset}_5m.parquet"
            try:
                df = pd.read_parquet(file_path)
                data[asset] = df
            except FileNotFoundError:
                logging.error(f"Data file not found: {file_path}")
                dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
                data[asset] = pd.DataFrame({
                    'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 100
                }, index=dates)
        return data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.is_training:
            self.equity = np.random.choice([100.0, 1000.0, 10000.0])
            self.leverage = np.random.choice([100, 200, 500])
            self.current_step = np.random.randint(100, self.max_steps - 288)
        else:
            self.equity = 10000.0
            self.leverage = 100
            self.current_step = 100
            
        self.start_equity = self.equity
        self.peak_equity = self.equity
        self.positions = {asset: None for asset in self.assets}
        self.portfolio_history = []
        
        # Reset Reward Trackers
        self.realized_pnl_step = 0.0
        self.transaction_costs_step = 0.0
        self.new_entry_bonus = 0.0
        
        return self._get_observation(), {}

    def step(self, action):
        # Reset step-specific reward trackers
        self.realized_pnl_step = 0.0
        self.transaction_costs_step = 0.0
        self.new_entry_bonus = 0.0
        
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
        
        return self._get_observation(), reward, terminated, truncated, {}

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
        
        # Calculate PnL
        diff = (price - pos['entry_price']) * pos['direction']
        position_value = pos['size'] * self.leverage
        pnl = (diff / pos['entry_price']) * position_value
        
        self.equity += pnl
        self.realized_pnl_step += pnl
        
        # Deduct Transaction Costs (Closing also incurs costs?)
        # PRD says "Deducted on every trade execution". Usually means entry and exit.
        cost = position_value * 0.0001
        self.equity -= cost
        self.transaction_costs_step += cost
        
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
        # PRD Section 5: Reward Function
        
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
        
        # 5. Drawdown Penalty
        drawdown = (self.equity / self.peak_equity) - 1
        drawdown_penalty = 0.0
        if drawdown < -0.25:
            drawdown_penalty = -10.0
        elif drawdown < -0.15:
            drawdown_penalty = -1.0
        elif drawdown < -0.10:
            drawdown_penalty = -0.3
            
        # 6. Holding Penalty
        # -0.01 * num_open_positions * (avg_position_age / 100)
        num_open = sum(1 for p in self.positions.values() if p is not None)
        avg_age = 0
        if num_open > 0:
            total_age = sum(self.current_step - p['entry_step'] for p in self.positions.values() if p is not None)
            avg_age = total_age / num_open
            
        holding_penalty = -0.01 * num_open * (avg_age / 100.0)
        
        # 7. Session Penalty
        # -0.5 if trade executed outside sessions
        session_penalty = 0.0
        # Check current hour
        current_row = self.processed_data.iloc[self.current_step]
        # We need to know if a trade was executed THIS step.
        # We can track this with a flag or check if transaction costs > 0 (proxy for execution)
        if self.transaction_costs_step > 0:
            # Check sessions
            is_london = current_row['session_london']
            is_ny = current_row['session_ny']
            is_overlap = current_row['session_overlap']
            
            if not (is_london or is_ny or is_overlap):
                session_penalty = -0.5
                
        # Total Reward
        total_reward = (
            realized 
            + unrealized 
            + rr_quality 
            - costs 
            + drawdown_penalty 
            + holding_penalty 
            + session_penalty
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

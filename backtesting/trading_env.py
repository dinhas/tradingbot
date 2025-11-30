import gymnasium as gym
import numpy as np
import pandas as pd
import json
import logging
from gymnasium import spaces

# --- Configuration ---
INITIAL_BALANCE = 10000.0
TRANSACTION_FEE_CRYPTO = 0.001  # 0.1%
TRANSACTION_FEE_FOREX = 0.0005  # 0.05%
MIN_TRADE_SIZE = 100.0
MIN_CASH_PCT = 0.05
MAX_POS_PCT = 0.40

ASSET_MAPPING = {
    0: 'BTC',
    1: 'ETH',
    2: 'SOL',
    3: 'EUR',
    4: 'GBP',
    5: 'JPY',
    6: 'CASH'
}

class TradingEnv(gym.Env):
    """
    Multi-Asset Trading Environment for RL.
    Action Space: Continuous (9,) - Portfolio Weights + SL/TP Multipliers
    Observation Space: Continuous (97,) - Market Features + Portfolio State
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir="./", volatility_file="volatility_baseline.json"):
        super(TradingEnv, self).__init__()
        
        self.data_dir = data_dir
        self.assets = ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']
        self.n_assets = len(self.assets)
        
        # Load Data
        self.data = self._load_data()
        self.timestamps = self.data.index
        self.n_steps = len(self.data)
        
        # Load Volatility Baselines
        with open(volatility_file, 'r') as f:
            self.volatility_baseline = json.load(f)
            
        # Define Spaces
        # Action: 9 dims -> 7 weights + 1 SL multiplier + 1 TP multiplier
        self.action_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        
        # Observation: 97 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(97,), dtype=np.float32)
        
        # State Variables
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.cash = INITIAL_BALANCE
        self.positions = {asset: 0.0 for asset in self.assets}
        self.holdings = {asset: 0.0 for asset in self.assets}
        self.entry_prices = {asset: 0.0 for asset in self.assets}
        self.previous_weights = np.zeros(self.n_assets + 1)
        self.previous_weights[6] = 1.0
        
        # Position Age Tracking (for holding period rewards)
        self.position_ages = {asset: 0 for asset in self.assets}
        
        # History for Rendering
        self.history = []

    def _load_data(self):
        """Loads and aligns parquet files for all assets."""
        dfs = {}
        
        for asset in self.assets:
            try:
                # Try backtest naming first, then fall back to training naming
                try:
                    df = pd.read_parquet(f"{self.data_dir}backtest_data_{asset}.parquet")
                except FileNotFoundError:
                    df = pd.read_parquet(f"{self.data_dir}data_{asset}_final.parquet")
                dfs[asset] = df
            except FileNotFoundError:
                raise FileNotFoundError(f"Data for {asset} not found.")
        
        # Find common index
        common_index = dfs[self.assets[0]].index
        for asset in self.assets[1:]:
            common_index = common_index.intersection(dfs[asset].index)
            
        if len(common_index) == 0:
            raise ValueError("No overlapping data found across assets.")
            
        full_df = pd.DataFrame(index=common_index)
        
        market_suffixes = [
            'log_ret', 'dist_ema50', 'atr_14_norm', 'bb_width', 'rsi_14_norm', 'macd_norm', 'vol_ratio', 'adx_norm',
            'rsi_4h', 'dist_ema50_4h', 'atr_4h_norm',
            'dist_ema200_1d', 'rsi_1d'
        ]
        
        for asset in self.assets:
            df = dfs[asset].loc[common_index]
            full_df[f"{asset}_close"] = df['close']
            for suffix in market_suffixes:
                full_df[f"{asset}_{suffix}"] = df.get(suffix, 0.0)
                    
        btc_df = dfs['BTC'].loc[common_index]
        full_df['sin_hour'] = btc_df['sin_hour']
        full_df['cos_hour'] = btc_df['cos_hour']
        full_df['day_of_week'] = btc_df['day_of_week']
        
        session_cols = ['is_btc_tradeable', 'is_eth_tradeable', 'is_sol_tradeable', 
                        'is_eur_tradeable', 'is_gbp_tradeable', 'is_jpy_tradeable']
        for col in session_cols:
            full_df[col] = btc_df.get(col, 1.0)
                
        c_corr = full_df['BTC_log_ret'].rolling(window=20).corr(full_df['ETH_log_ret'])
        full_df['crypto_correlation'] = c_corr.fillna(0.0)
        
        crypto_mom = (full_df['BTC_rsi_14_norm'] + full_df['ETH_rsi_14_norm'] + full_df['SOL_rsi_14_norm']) / 3.0
        forex_mom = (full_df['EUR_rsi_14_norm'] + full_df['GBP_rsi_14_norm'] + full_df['JPY_rsi_14_norm']) / 3.0
        full_df['crypto_forex_divergence'] = crypto_mom - forex_mom
        
        full_df.fillna(0.0, inplace=True)
        return full_df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.cash = INITIAL_BALANCE
        self.positions = {asset: 0.0 for asset in self.assets}
        self.holdings = {asset: 0.0 for asset in self.assets}
        self.entry_prices = {asset: 0.0 for asset in self.assets}
        self.previous_weights = np.zeros(self.n_assets + 1)
        self.previous_weights[6] = 1.0
        self.peak_value = INITIAL_BALANCE
        self.position_ages = {asset: 0 for asset in self.assets}
        self.history = []
        
        return self._get_observation(), {}

    def step(self, action):
        # Extract Weights and Risk Params
        raw_weights = action[0:7]
        sl_mult_raw = action[7]
        tp_mult_raw = action[8]
        
        # HARD ENFORCEMENT of SL/TP minimums
        sl_mult_clipped = np.clip(sl_mult_raw, 0.25, 1.0)
        tp_mult_clipped = np.clip(tp_mult_raw, 0.12, 1.0)
        
        self.sl_multiplier = 2.0 + (sl_mult_clipped * 3.0)  # 2.0 to 5.0 ATR
        self.tp_multiplier = 2.0 + (tp_mult_clipped * 8.0)  # 2.0 to 10.0 ATR

        # Apply Softmax with Masking
        mask = self.action_masks()
        exp_action = np.exp(raw_weights)
        masked_exp = exp_action * mask[0:7]
        
        sum_exp = np.sum(masked_exp)
        if sum_exp == 0:
            weights = np.zeros_like(raw_weights)
            weights[6] = 1.0
        else:
            weights = masked_exp / sum_exp
        
        # Enforce Constraints
        if weights[6] < MIN_CASH_PCT:
            weights[6] = MIN_CASH_PCT
            risky_sum = np.sum(weights[0:6])
            if risky_sum > 0:
                weights[0:6] = weights[0:6] / risky_sum * (1.0 - MIN_CASH_PCT)
        
        weights[0:6] = np.minimum(weights[0:6], MAX_POS_PCT)
        weights[6] = 1.0 - np.sum(weights[0:6])
        
        # Get Current Market Data
        current_data = self.data.iloc[self.current_step]
        prices = {asset: current_data[f"{asset}_close"] for asset in self.assets}
        
        # Update Portfolio Value
        new_portfolio_value = self.cash
        for asset in self.assets:
            if self.holdings[asset] > 0:
                new_portfolio_value += self.holdings[asset] * prices[asset]
        
        prev_portfolio_value = self.portfolio_value
        self.portfolio_value = new_portfolio_value
        
        # Automatic SL/TP
        trade_events = {}
        for asset in self.assets:
            if self.holdings[asset] > 0:
                entry_price = self.entry_prices[asset]
                current_price = prices[asset]
                
                atr_norm = current_data.get(f"{asset}_atr_14_norm", 0.01)
                atr = atr_norm * current_price
                
                sl_dist = self.sl_multiplier * atr
                tp_dist = self.tp_multiplier * atr
                
                stop_loss = entry_price - sl_dist
                take_profit = entry_price + tp_dist
                
                if current_price <= stop_loss or current_price >= take_profit:
                    units = self.holdings[asset]
                    proceeds = units * current_price
                    
                    fee_rate = TRANSACTION_FEE_CRYPTO if asset in ['BTC', 'ETH', 'SOL'] else TRANSACTION_FEE_FOREX
                    fee = proceeds * fee_rate
                    
                    pnl = proceeds - (units * entry_price) - fee
                    exit_reason = "SL" if current_price <= stop_loss else "TP"
                    
                    trade_events[asset] = {
                        'reason': exit_reason,
                        'pnl': pnl,
                        'price': current_price
                    }
                    
                    self.cash += (proceeds - fee)
                    self.holdings[asset] = 0.0
                    self.entry_prices[asset] = 0.0
                    self.position_ages[asset] = 0
                    self.portfolio_value -= fee
                    
                    weights[self.assets.index(asset)] = 0.0
                    weights[6] += weights[self.assets.index(asset)]
        
        # Execute Rebalance with Position Age Tracking
        target_values = {asset: self.portfolio_value * weights[i] for i, asset in enumerate(self.assets)}
        total_fees = 0.0
        
        for i, asset in enumerate(self.assets):
            current_pos_value = self.holdings[asset] * prices[asset]
            target_pos_value = target_values[asset]
            diff = target_pos_value - current_pos_value
            
            if abs(diff) > MIN_TRADE_SIZE:
                trade_value = abs(diff)
                fee_rate = TRANSACTION_FEE_CRYPTO if asset in ['BTC', 'ETH', 'SOL'] else TRANSACTION_FEE_FOREX
                fee = trade_value * fee_rate
                total_fees += fee
                
                if diff > 0:  # Buy
                    units_to_buy = diff / prices[asset]
                    current_units = self.holdings[asset]
                    total_units = current_units + units_to_buy
                    if total_units > 0:
                        avg_price = ((current_units * self.entry_prices[asset]) + (units_to_buy * prices[asset])) / total_units
                        self.entry_prices[asset] = avg_price
                        
                    self.holdings[asset] += units_to_buy
                    self.cash -= diff
                    
                    if current_pos_value < 10:
                        self.position_ages[asset] = 0
                else:  # Sell
                    units_to_sell = abs(diff) / prices[asset]
                    self.holdings[asset] -= units_to_sell
                    self.cash += abs(diff)
                    if self.holdings[asset] <= 1e-6:
                        self.entry_prices[asset] = 0.0
                        self.position_ages[asset] = 0
            else:
                # Holding: increment age
                if current_pos_value > 10:
                    self.position_ages[asset] += 1
        
        self.cash -= total_fees
        self.portfolio_value -= total_fees
        
        # Calculate Reward
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
        
        turnover = np.sum(np.abs(weights - self.previous_weights))
        self.previous_weights = weights.copy()
        
        port_vol = sum(weights[i] * self.volatility_baseline.get(asset, 0.01) for i, asset in enumerate(self.assets))
        port_vol += weights[6] * 0.001
        port_vol = max(port_vol, 1e-4)
        
        sharpe = portfolio_return / port_vol
        
        # EXPONENTIAL Turnover Penalty
        if turnover < 0.2:
            turnover_penalty = turnover * 0.3
        elif turnover < 0.5:
            turnover_penalty = turnover * 1.0
        else:
            turnover_penalty = (turnover ** 2) * 2.0
        
        # Holding Period Bonus
        active_positions = sum(1 for h in self.holdings.values() if h > 0)
        if active_positions > 0:
            avg_holding = sum(self.position_ages.values()) / active_positions
            holding_bonus = 0.2 if avg_holding >= 4 else 0.0
        else:
            holding_bonus = 0.0
        
        raw_reward = (0.9 * portfolio_return + 0.1 * sharpe) - turnover_penalty + holding_bonus
        normalized_reward = raw_reward / port_vol
        final_reward = normalized_reward * 0.5
        
        # Check Termination
        terminated = False
        truncated = False
        
        if self.portfolio_value < INITIAL_BALANCE * 0.7:
            terminated = True
            final_reward = -50.0
            
        self.peak_value = getattr(self, 'peak_value', INITIAL_BALANCE)
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        if drawdown > 0.30:
            terminated = True
            final_reward = -30.0
            
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            truncated = True
            
        info = {
            'portfolio_value': self.portfolio_value,
            'fees': total_fees,
            'return': portfolio_return,
            'drawdown': drawdown,
            'sl_multiplier': float(self.sl_multiplier),
            'tp_multiplier': float(self.tp_multiplier),
            'trade_events': trade_events
        }
        
        return self._get_observation(), final_reward, terminated, truncated, info

    def _get_observation(self):
        row = self.data.iloc[self.current_step]
        
        market_features = []
        market_suffixes = [
            'log_ret', 'dist_ema50', 'atr_14_norm', 'bb_width', 'rsi_14_norm', 'macd_norm', 'vol_ratio', 'adx_norm',
            'rsi_4h', 'dist_ema50_4h', 'atr_4h_norm',
            'dist_ema200_1d', 'rsi_1d'
        ]
        
        for asset in self.assets:
            for suffix in market_suffixes:
                market_features.append(row.get(f"{asset}_{suffix}", 0.0))
                
        temporal_features = [
            row.get('sin_hour', 0.0),
            row.get('cos_hour', 0.0),
            row.get('day_of_week', 0.0)
        ]
        
        session_cols = ['is_btc_tradeable', 'is_eth_tradeable', 'is_sol_tradeable', 
                        'is_eur_tradeable', 'is_gbp_tradeable', 'is_jpy_tradeable']
        session_features = [row.get(col, 1.0) for col in session_cols]
        
        current_weights = [self.holdings[a] * row[f"{a}_close"] / self.portfolio_value for a in self.assets]
        current_weights.append(self.cash / self.portfolio_value)
        
        unrealized_pnl = (self.portfolio_value - self.peak_value) / self.peak_value
        portfolio_features = current_weights + [unrealized_pnl]
        
        cross_asset_features = [
            row.get('crypto_correlation', 0.0),
            row.get('crypto_forex_divergence', 0.0)
        ]
        
        obs = np.concatenate([
            market_features, 
            temporal_features, 
            session_features, 
            portfolio_features, 
            cross_asset_features
        ], dtype=np.float32)
        
        if len(obs) != 97:
            obs = np.resize(obs, (97,))
            
        return obs

    def action_masks(self):
        current_time = self.timestamps[self.current_step]
        hour = current_time.hour
        weekday = current_time.weekday()
        
        mask = np.ones(9, dtype=bool)
        is_weekend = weekday >= 5
        
        if is_weekend:
            mask[3] = False
            mask[4] = False
            mask[5] = False
        else:
            if not (7 <= hour < 21): mask[3] = False
            if not (7 <= hour < 21): mask[4] = False
            if not (hour >= 23 or hour < 16): mask[5] = False
            
        return mask

    def _get_action_mask(self):
        return self.action_masks()

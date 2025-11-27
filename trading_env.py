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
    Action Space: Continuous (7,) - Portfolio Weights [BTC, ETH, SOL, EUR, GBP, JPY, CASH]
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
        # Action: 7 weights (sum to 1 handled in step)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        
        # Observation: 97 features
        # 78 Market + 3 Temporal + 6 Session + 8 Portfolio + 2 Cross-Asset
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(97,), dtype=np.float32)
        
        # State Variables
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.cash = INITIAL_BALANCE
        self.positions = {asset: 0.0 for asset in self.assets} # In USD value
        self.holdings = {asset: 0.0 for asset in self.assets}  # In units
        self.entry_prices = {asset: 0.0 for asset in self.assets} # Weighted avg entry price
        
        # History for Rendering
        self.history = []

    def _load_data(self):
        """Loads and aligns parquet files for all assets, calculates cross-asset features."""
        dfs = {}
        
        # Load all asset files
        for asset in self.assets:
            try:
                # Load parquet
                df = pd.read_parquet(f"{self.data_dir}data_{asset}_final.parquet")
                dfs[asset] = df
            except FileNotFoundError:
                raise FileNotFoundError(f"Data for {asset} not found. Run ctradercervice.py first.")
        
        # Find common index (intersection)
        common_index = dfs[self.assets[0]].index
        for asset in self.assets[1:]:
            common_index = common_index.intersection(dfs[asset].index)
            
        if len(common_index) == 0:
            raise ValueError("No overlapping data found across assets.")
            
        # Initialize Full DataFrame
        full_df = pd.DataFrame(index=common_index)
        
        # Define Market Features (13 per asset) based on ctradercervice.py
        market_suffixes = [
            'log_ret', 'dist_ema50', 'atr_14_norm', 'bb_width', 'rsi_14_norm', 'macd_norm', 'vol_ratio', 'adx_norm', # 8 (15m)
            'rsi_4h', 'dist_ema50_4h', 'atr_4h_norm', # 3 (4H)
            'dist_ema200_1d', 'rsi_1d' # 2 (Daily)
        ]
        
        # 1. Merge Asset Data
        for asset in self.assets:
            df = dfs[asset].loc[common_index]
            
            # Add Raw Close (Critical for portfolio value)
            full_df[f"{asset}_close"] = df['close']
            
            # Add Market Features
            for suffix in market_suffixes:
                if suffix in df.columns:
                    full_df[f"{asset}_{suffix}"] = df[suffix]
                else:
                    # Fallback for missing columns (shouldn't happen if data is good)
                    full_df[f"{asset}_{suffix}"] = 0.0
                    
        # 2. Add Temporal Features (Take from BTC, identical across assets)
        btc_df = dfs['BTC'].loc[common_index]
        full_df['sin_hour'] = btc_df['sin_hour']
        full_df['cos_hour'] = btc_df['cos_hour']
        full_df['day_of_week'] = btc_df['day_of_week']
        
        # 3. Add Session Features (Take from BTC, identical across assets)
        session_cols = ['is_btc_tradeable', 'is_eth_tradeable', 'is_sol_tradeable', 
                        'is_eur_tradeable', 'is_gbp_tradeable', 'is_jpy_tradeable']
        for col in session_cols:
            if col in btc_df.columns:
                full_df[col] = btc_df[col]
            else:
                full_df[col] = 1.0 # Default to tradeable if missing
                
        # 4. Calculate Cross-Asset Features (2 features)
        # 4.1 Crypto Correlation (BTC vs ETH returns)
        # Use normalized log_ret (correlation of z-scores is valid)
        c_corr = full_df['BTC_log_ret'].rolling(window=20).corr(full_df['ETH_log_ret'])
        full_df['crypto_correlation'] = c_corr.fillna(0.0)
        
        # 4.2 Crypto vs Forex Divergence
        # avg(crypto_momentum) - avg(forex_momentum)
        # Using RSI as momentum proxy
        crypto_mom = (full_df['BTC_rsi_14_norm'] + full_df['ETH_rsi_14_norm'] + full_df['SOL_rsi_14_norm']) / 3.0
        forex_mom = (full_df['EUR_rsi_14_norm'] + full_df['GBP_rsi_14_norm'] + full_df['JPY_rsi_14_norm']) / 3.0
        full_df['crypto_forex_divergence'] = crypto_mom - forex_mom
        
        # Fill NaNs (from rolling windows)
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
        self.peak_value = INITIAL_BALANCE
        self.history = []
        
        return self._get_observation(), {}

    def step(self, action):
        # 1. Enforce Action Masking (Session Constraints)
        mask = self.action_masks()
        
        # Apply Softmax with Masking
        # We exponentiate the action (logits)
        exp_action = np.exp(action)
        
        # Zero out the probabilities of masked actions
        masked_exp = exp_action * mask
        
        # Normalize to get weights
        sum_exp = np.sum(masked_exp)
        
        if sum_exp == 0:
            # Fallback: If everything is masked (shouldn't happen as Cash is always open), go 100% Cash
            weights = np.zeros_like(action)
            weights[6] = 1.0
        else:
            weights = masked_exp / sum_exp
        
        # --- ENFORCE CONSTRAINTS (RPD v3.0) ---
        # 1. Min Cash 5%
        if weights[6] < MIN_CASH_PCT:
            weights[6] = MIN_CASH_PCT
            # Renormalize others to fit 0.95
            risky_sum = np.sum(weights[0:6])
            if risky_sum > 0:
                weights[0:6] = weights[0:6] / risky_sum * (1.0 - MIN_CASH_PCT)
        
        # 2. Max Position 40%
        # Clip risky assets to 0.40
        weights[0:6] = np.minimum(weights[0:6], MAX_POS_PCT)
        # Recalculate Cash (absorb the difference)
        weights[6] = 1.0 - np.sum(weights[0:6])
        
        # 2. Get Current Market Data
        current_data = self.data.iloc[self.current_step]
        prices = {asset: current_data[f"{asset}_close"] for asset in self.assets}
        
        # 3. Update Portfolio Value (Mark-to-Market)
        new_portfolio_value = self.cash
        for asset in self.assets:
            if self.holdings[asset] > 0:
                new_portfolio_value += self.holdings[asset] * prices[asset]
        
        prev_portfolio_value = self.portfolio_value
        self.portfolio_value = new_portfolio_value
        
        # --- AUTOMATIC STOP-LOSS & TAKE-PROFIT (RPD v3.0) ---
        for asset in self.assets:
            if self.holdings[asset] > 0:
                entry_price = self.entry_prices[asset]
                current_price = prices[asset]
                
                # Recover ATR
                atr_norm = current_data.get(f"{asset}_atr_14_norm", 0.01)
                atr = atr_norm * current_price
                
                stop_loss = entry_price - (2.0 * atr)
                take_profit = entry_price + (3.0 * atr)
                
                if current_price <= stop_loss or current_price >= take_profit:
                    # Force Close Position
                    units = self.holdings[asset]
                    proceeds = units * current_price
                    
                    # Fee
                    fee_rate = TRANSACTION_FEE_CRYPTO if asset in ['BTC', 'ETH', 'SOL'] else TRANSACTION_FEE_FOREX
                    fee = proceeds * fee_rate
                    
                    self.cash += (proceeds - fee)
                    self.holdings[asset] = 0.0
                    self.entry_prices[asset] = 0.0
                    self.portfolio_value -= fee
                    
                    # Override target for this asset to 0 for this step
                    weights[self.assets.index(asset)] = 0.0
                    # Renormalize cash
                    weights[6] += weights[self.assets.index(asset)]
        
        # 4. Execute Rebalance
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
                
                if diff > 0: # Buy
                    units_to_buy = diff / prices[asset]
                    current_units = self.holdings[asset]
                    total_units = current_units + units_to_buy
                    if total_units > 0:
                        avg_price = ((current_units * self.entry_prices[asset]) + (units_to_buy * prices[asset])) / total_units
                        self.entry_prices[asset] = avg_price
                        
                    self.holdings[asset] += units_to_buy
                    self.cash -= diff
                else: # Sell
                    units_to_sell = abs(diff) / prices[asset]
                    self.holdings[asset] -= units_to_sell
                    self.cash += abs(diff)
                    if self.holdings[asset] <= 1e-6:
                        self.entry_prices[asset] = 0.0
        
        # Deduct Fees
        self.cash -= total_fees
        self.portfolio_value -= total_fees
        
        # 5. Calculate Reward
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
        
        # Volatility Baseline
        port_vol = 0.0
        for i, asset in enumerate(self.assets):
            port_vol += weights[i] * self.volatility_baseline.get(asset, 0.01)
        port_vol += weights[6] * 0.001
        
        sharpe = portfolio_return / (port_vol + 1e-9)
        
        raw_reward = 0.9 * portfolio_return + 0.1 * sharpe
        normalized_reward = raw_reward / (port_vol + 1e-9)
        final_reward = normalized_reward * 100
        
        # 6. Check Termination
        terminated = False
        truncated = False
        
        if self.portfolio_value < INITIAL_BALANCE * 0.7:
            terminated = True
            final_reward = -50.0
            
        if not hasattr(self, 'peak_value'):
            self.peak_value = INITIAL_BALANCE
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
            'drawdown': drawdown
        }
        
        return self._get_observation(), final_reward, terminated, truncated, info

    def _get_observation(self):
        """
        Constructs the 97-dimensional observation vector.
        Order:
        1. Market Features (78): 13 features * 6 assets
        2. Temporal Features (3): sin, cos, day
        3. Session Features (6): is_tradeable per asset
        4. Portfolio State (8): 7 weights + 1 unrealized pnl
        5. Cross-Asset Features (2): correlation, divergence
        """
        row = self.data.iloc[self.current_step]
        
        # 1. Market Features (78)
        market_features = []
        market_suffixes = [
            'log_ret', 'dist_ema50', 'atr_14_norm', 'bb_width', 'rsi_14_norm', 'macd_norm', 'vol_ratio', 'adx_norm',
            'rsi_4h', 'dist_ema50_4h', 'atr_4h_norm',
            'dist_ema200_1d', 'rsi_1d'
        ]
        
        for asset in self.assets:
            for suffix in market_suffixes:
                val = row.get(f"{asset}_{suffix}", 0.0)
                market_features.append(val)
                
        # 2. Temporal Features (3)
        temporal_features = [
            row.get('sin_hour', 0.0),
            row.get('cos_hour', 0.0),
            row.get('day_of_week', 0.0)
        ]
        
        # 3. Session Features (6)
        session_cols = ['is_btc_tradeable', 'is_eth_tradeable', 'is_sol_tradeable', 
                        'is_eur_tradeable', 'is_gbp_tradeable', 'is_jpy_tradeable']
        session_features = [row.get(col, 1.0) for col in session_cols]
        
        # 4. Portfolio State (8)
        current_weights = [self.holdings[a] * row[f"{a}_close"] / self.portfolio_value for a in self.assets]
        current_weights.append(self.cash / self.portfolio_value) # Cash weight
        
        unrealized_pnl = (self.portfolio_value - self.peak_value) / self.peak_value
        portfolio_features = current_weights + [unrealized_pnl]
        
        # 5. Cross-Asset Features (2)
        cross_asset_features = [
            row.get('crypto_correlation', 0.0),
            row.get('crypto_forex_divergence', 0.0)
        ]
        
        # Concatenate
        obs = np.concatenate([
            market_features, 
            temporal_features, 
            session_features, 
            portfolio_features, 
            cross_asset_features
        ], dtype=np.float32)
        
        # Safety Check
        if len(obs) != 97:
            # Pad or truncate if necessary (should not happen if logic is correct)
            obs = np.resize(obs, (97,))
            
        return obs

    def action_masks(self):
        """
        Returns a boolean mask for valid actions.
        Required for MaskablePPO.
        """
        # Implement session masking
        current_time = self.timestamps[self.current_step]
        hour = current_time.hour
        weekday = current_time.weekday()
        
        # 0-2: Crypto (Always True)
        # 3: EUR (London/NY: 08-21 UTC)
        # 4: GBP (London/NY: 08-21 UTC)
        # 5: JPY (Tokyo/London: 00-16 UTC)
        # 6: Cash (Always True)
        
        mask = np.ones(7, dtype=bool)
        
        # Forex Constraints
        is_weekend = weekday >= 5
        
        if is_weekend:
            mask[3] = False
            mask[4] = False
            mask[5] = False
        else:
            # Simple hour checks (UTC)
            if not (8 <= hour < 21): mask[3] = False # EUR
            if not (8 <= hour < 21): mask[4] = False # GBP
            if not (0 <= hour < 16): mask[5] = False # JPY
            
        return mask

    def _get_action_mask(self):
        # Deprecated internal method, keeping for compatibility if needed
        return self.action_masks()

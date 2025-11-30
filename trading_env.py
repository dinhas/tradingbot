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
    Action Space: Continuous (9,)
        - 0-6: Portfolio Weights [BTC, ETH, SOL, EUR, GBP, JPY, CASH]
        - 7: Stop Loss Multiplier (1.0 - 5.0)
        - 8: Take Profit Multiplier (1.0 - 10.0)
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
        # Action: 9 dims (7 weights + 1 SL + 1 TP)
        self.action_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        
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
        self.previous_weights = np.zeros(self.n_assets + 1) # Track previous weights for turnover
        self.previous_weights[6] = 1.0 # Start with 100% cash
        
        # Position Age Tracking (for holding period rewards)
        self.position_ages = {asset: 0 for asset in self.assets}  # Track steps held
        
        # Anti-Cheat Tracking
        self.cash_hoarding_steps = 0  # Track consecutive steps with >80% cash
        self.stale_steps = 0  # Track steps with no significant rebalancing
        
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
        self.previous_weights = np.zeros(self.n_assets + 1)
        self.previous_weights[6] = 1.0
        self.tp_hit_count = 0
        self.sl_hit_count = 0
        self.peak_value = INITIAL_BALANCE
        
        # Position Age Tracking
        self.position_ages = {asset: 0 for asset in self.assets}
        
        # Anti-Cheat Tracking
        self.cash_hoarding_steps = 0
        self.stale_steps = 0
        
        self.history = []
        self.trade_history = []  # list of dicts with 'net_profit'
        self.win_rate_window = 50
        
        return self._get_observation(), {}

    def step(self, action):
        # 1. Enforce Action Masking (Session Constraints)
        mask = self.action_masks()
        
        # Extract Weights (0-6) and Risk Params (7-8)
        raw_weights = action[0:7]
        sl_mult_raw = action[7]
        tp_mult_raw = action[8]
        
        # Decode Risk Params with HARD ENFORCEMENT
        # Clip inputs to force minimum values (prevents 0.00x SL exploit)
        sl_mult_clipped = np.clip(sl_mult_raw, 0.25, 1.0)  # Force minimum 25% of range
        tp_mult_clipped = np.clip(tp_mult_raw, 0.12, 1.0)  # Force minimum 12% of range
        
        # SL: 2.0 to 5.0 ATR (minimum doubled from 1.0)
        self.sl_multiplier = 2.0 + (sl_mult_clipped * 3.0)
        # TP: 2.0 to 10.0 ATR (minimum raised to match SL floor)
        self.tp_multiplier = 2.0 + (tp_mult_clipped * 8.0)

        # Apply Softmax with Masking to Weights
        # We exponentiate the action (logits)
        exp_action = np.exp(raw_weights)
        
        # Zero out the probabilities of masked actions (only first 7 are masked)
        masked_exp = exp_action * mask[0:7]
        
        # Normalize to get weights
        sum_exp = np.sum(masked_exp)
        
        if sum_exp == 0:
            # Fallback: If everything is masked (shouldn't happen as Cash is always open), go 100% Cash
            weights = np.zeros_like(raw_weights)
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
        trade_profit_reward = 0.0
        
        for asset in self.assets:
            if self.holdings[asset] > 0:
                entry_price = self.entry_prices[asset]
                current_price = prices[asset]
                
                # Recover ATR
                atr_norm = current_data.get(f"{asset}_atr_14_norm", 0.01)
                atr = atr_norm * current_price
                
                # Use Dynamic Multipliers from Action
                stop_loss = entry_price - (self.sl_multiplier * atr)
                take_profit = entry_price + (self.tp_multiplier * atr)
                
                if current_price <= stop_loss or current_price >= take_profit:
                    # Force Close Position
                    units = self.holdings[asset]
                    proceeds = units * current_price
                    
                    # Fee
                    fee_rate = TRANSACTION_FEE_CRYPTO if asset in ['BTC', 'ETH', 'SOL'] else TRANSACTION_FEE_FOREX
                    fee = proceeds * fee_rate
                    
                    # Compute profit of the trade (in USD)
                    entry_value = units * entry_price
                    gross_profit = proceeds - entry_value
                    net_profit = gross_profit - fee

                    # Scale profit into reward space:
                    # Normalize by initial balance and scale (experiment with multiplier)
                    profit_reward = (net_profit / INITIAL_BALANCE) * 100.0  # e.g. +1 means +1% P&L
                    
                    # Immediate shaping for TP/SL
                    if current_price >= take_profit:
                        profit_reward += 0.5   # small extra shaping for hitting TP
                    elif current_price <= stop_loss:
                        profit_reward -= 0.5   # small penalty for hitting SL
                        
                    trade_profit_reward += profit_reward

                    # Update cash and holdings
                    self.cash += (proceeds - fee)
                    self.holdings[asset] = 0.0
                    self.entry_prices[asset] = 0.0
                    self.portfolio_value -= fee
                    
                    # Track Hits (for logging only now)
                    if current_price <= stop_loss:
                        self.sl_hit_count += 1
                    elif current_price >= take_profit:
                        self.tp_hit_count += 1
                        
                    # Record trade outcome for analysis / win-rate metric
                    self.trade_history.append({
                        'asset': asset,
                        'entry': entry_price,
                        'exit': current_price,
                        'net_profit': net_profit,
                        'timestamp': self.timestamps[self.current_step]
                    })
                    
                    # Override target for this asset to 0 for this step
                    asset_idx = self.assets.index(asset)
                    removed_weight = weights[asset_idx]
                    weights[asset_idx] = 0.0
                    # Renormalize cash
                    weights[6] += removed_weight
        
        # 4. Execute Rebalance with Position Age Tracking
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
                    
                    # Reset age if opening new position
                    if current_pos_value < 10:
                        self.position_ages[asset] = 0
                else: # Sell
                    units_to_sell = abs(diff) / prices[asset]
                    self.holdings[asset] -= units_to_sell
                    self.cash += abs(diff)
                    if self.holdings[asset] <= 1e-6:
                        self.entry_prices[asset] = 0.0
                        self.position_ages[asset] = 0  # Reset age on close
            else:
                # Holding position: increment age
                if current_pos_value > 10:
                    self.position_ages[asset] += 1
        
        # Deduct Fees
        self.cash -= total_fees
        self.portfolio_value -= total_fees
        
        # 5. Calculate Reward - ENTRY QUALITY FOCUSED SYSTEM
        # NO portfolio returns - focus purely on entry quality and risk management
        
        # Calculate Turnover (sum of absolute weight changes)
        turnover = np.sum(np.abs(weights - self.previous_weights))
        self.previous_weights = weights.copy()
        
        # --- ENTRY QUALITY SCORING (Primary Signal: ±10.0 range) ---
        entry_quality = 0.0
        current_data = self.data.iloc[self.current_step]
        
        for i, asset in enumerate(self.assets):
            # Only score active positions (weight > 5%)
            if weights[i] > 0.05:
                # Get technical indicators
                rsi = current_data.get(f"{asset}_rsi_14_norm", 0.5)  # Normalized 0-1
                atr_norm = current_data.get(f"{asset}_atr_14_norm", 0.01)
                dist_ema50 = current_data.get(f"{asset}_dist_ema50", 0.0)
                macd_norm = current_data.get(f"{asset}_macd_norm", 0.0)
                adx = current_data.get(f"{asset}_adx_norm", 0.0)
                
                # 1. RSI Timing Quality (+2.0 for good zones)
                # Oversold (0.3-0.45) or neutral (0.45-0.55) is good for entries
                if 0.30 <= rsi <= 0.55:
                    entry_quality += 2.0
                elif rsi > 0.70:  # Overbought entry is poor
                    entry_quality -= 1.5
                
                # 2. Volatility Quality (+1.5 for calm markets)
                # Lower ATR = safer entry conditions
                if atr_norm < 0.015:
                    entry_quality += 1.5
                elif atr_norm > 0.03:  # High volatility entry is risky
                    entry_quality -= 1.0
                
                # 3. Mean Reversion Quality (+1.5 for near support/resistance)
                # Distance from EMA50: -0.02 to -0.05 = near support (good buy)
                if -0.05 <= dist_ema50 <= -0.02:
                    entry_quality += 1.5
                elif dist_ema50 < -0.10:  # Too far from mean
                    entry_quality -= 1.0
                
                # 4. Trend Strength Quality (+1.0 for clear trends)
                # MACD alignment with ADX confirms trend
                if abs(macd_norm) > 0.3 and adx > 0.5:  # Strong trend
                    entry_quality += 1.0
                elif adx < 0.3:  # Choppy market
                    entry_quality -= 0.5
        
        # --- TURNOVER PENALTY (Prevent Overtrading: -1.0 to -6.0) ---
        # Gentler schedule
        if turnover > 0.5:
            turnover_penalty = turnover * 6.0   # was *40.0
        elif turnover > 0.2:
            turnover_penalty = turnover * 3.0   # was *10.0
        else:
            turnover_penalty = turnover * 1.5   # was *2.0
        
        # --- TP/SL COUNTERS (Reset but not used directly in reward anymore) ---
        # We used trade_profit_reward instead
        self.tp_hit_count = 0
        self.sl_hit_count = 0
        
        # --- RISK-TO-REWARD RATIO QUALITY (Soft reward) ---
        rr_ratio = self.tp_multiplier / (self.sl_multiplier + 1e-9)
        # Soft reward: gaussian-like centered on 2.5
        target_rr = 2.5
        rr_quality = max(-3.0, 5.0 * np.exp(-((rr_ratio - target_rr)**2) / (2 * 0.8**2))) - 1.0
        
        # --- WIN RATE BONUS (Momentum in trade outcomes) ---
        # Keep last N trades and give a small bonus if win rate improves
        last_trades = [1 if t['net_profit']>0 else 0 for t in self.trade_history[-self.win_rate_window:]]
        if len(last_trades) > 0:
            win_rate = sum(last_trades) / len(last_trades)
            winrate_bonus = (win_rate - 0.5) * 2.0  # small bonus if >50% wins
        else:
            winrate_bonus = 0.0

        # --- PORTFOLIO RETURN REWARD (Short-term stability) ---
        # Calculate portfolio return for info dict AND reward
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-9)
        return_reward = portfolio_return * 50.0   # 1% portfolio increase -> +0.5 reward (tune)

        # --- ANTI-CASH-HOARDING (Encourage Deployment: -2.0 max) ---
        cash_weight = weights[6]
        if cash_weight > 0.8:
            self.cash_hoarding_steps += 1
        else:
            self.cash_hoarding_steps = 0
        
        # Penalty: -0.2 per step after 10 consecutive cash-heavy steps (was -0.1)
        cash_hoarding_penalty = max(0, (self.cash_hoarding_steps - 10) * 0.2)
        
        # --- ANTI-STAGNATION (Encourage Action: -2.0 max) ---
        if turnover < 0.05:  # Less than 5% portfolio change
            self.stale_steps += 1
        else:
            self.stale_steps = 0
        
        # Penalty: -0.1 per step after 20 consecutive stale steps (was -0.05)
        staleness_penalty = max(0, (self.stale_steps - 20) * 0.1)
        
        # --- HOLDING PERIOD BONUS (Reward Patience: +2.0 max) ---
        active_positions = sum(1 for h in self.holdings.values() if h > 0)
        if active_positions > 0:
            avg_holding = sum(self.position_ages.values()) / active_positions
            # Progressive bonus for longer holds
            if avg_holding >= 16:  # 4 hours
                holding_bonus = 2.0
            elif avg_holding >= 8:  # 2 hours
                holding_bonus = 1.0
            elif avg_holding >= 4:  # 1 hour
                holding_bonus = 0.5
            else:
                holding_bonus = 0.0
        else:
            holding_bonus = 0.0
        
        # --- FINAL REWARD COMPOSITION ---
        final_reward = (
            entry_quality +           # ±10.0 (PRIMARY SIGNAL)
            rr_quality +              # ±5.0
            trade_profit_reward +     # Realized profit signal (includes TP/SL shaping)
            winrate_bonus +           # ±1.0
            return_reward +           # Portfolio return signal
            holding_bonus -           # +2.0 max
            turnover_penalty -        # -1.5 to -6.0
            cash_hoarding_penalty -   # -2.0 max
            staleness_penalty         # -2.0 max
        )
        
        # Clip final reward
        final_reward = float(np.clip(final_reward, -50.0, 50.0))
        
        # Debug Logging
        if self.current_step % 500 == 0:
            print(f"step={self.current_step} pv={self.portfolio_value:.2f} reward={final_reward:.3f} turnover={turnover:.3f} trade_reward={trade_profit_reward:.3f}")
        
        # 6. Check Termination
        terminated = False
        truncated = False
        
        # Only terminate on catastrophic loss (70% of initial capital)
        # Removed 30% drawdown terminal - let model learn to recover from drawdowns
        if self.portfolio_value < INITIAL_BALANCE * 0.7:
            terminated = True
            final_reward = -50.0
            
        # Track peak for info only (not for termination)
        if not hasattr(self, 'peak_value'):
            self.peak_value = INITIAL_BALANCE
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            truncated = True
        
        # Calculate portfolio return for info dict (not used in reward)
        # portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
            
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
        
        mask = np.ones(9, dtype=bool)
        
        # Forex Constraints
        is_weekend = weekday >= 5
        
        if is_weekend:
            mask[3] = False
            mask[4] = False
            mask[5] = False
        else:
            # Allow trading 1 hour BEFORE session open
            # EUR (London/NY): 08-21 UTC -> Allow 07-21
            if not (7 <= hour < 21): mask[3] = False 
            # GBP (London/NY): 08-21 UTC -> Allow 07-21
            if not (7 <= hour < 21): mask[4] = False 
            # JPY (Tokyo/London): 00-16 UTC -> Allow 23-16 (Handle wrap around)
            # 23 (prev day) to 16
            if not (hour >= 23 or hour < 16): mask[5] = False
            
        # SL/TP actions (7, 8) are always valid
            
        return mask

    def _get_action_mask(self):
        # Deprecated internal method, keeping for compatibility if needed
        return self.action_masks()

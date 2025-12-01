import gymnasium as gym
import numpy as np
import pandas as pd
import json
import logging
from gymnasium import spaces

# --- Configuration ---
INITIAL_BALANCE = 10000.0
LEVERAGE = 200.0  # 1:200 leverage for forex/crypto CFDs (used for display only now)
TRANSACTION_FEE_CRYPTO = 0.001  # 0.1%
TRANSACTION_FEE_FOREX = 0.0005  # 0.05%
MIN_TRADE_SIZE = 100.0
MIN_CASH_PCT = 0.05
MAX_POS_PCT = 0.40

# Reward Scaling Constants (Easy to tune)
RETURN_REWARD_SCALE = 1000.0      # 0.1% return = +1.0 reward
TRADE_PROFIT_SCALE = 0.01         # $100 profit = +1.0 reward
TURNOVER_PENALTY_LIGHT = 0.5      # For <20% turnover
TURNOVER_PENALTY_MEDIUM = 1.5     # For 20-50% turnover
TURNOVER_PENALTY_HEAVY = 3.0      # For >50% turnover
WARMUP_STEPS = 5                  # No turnover penalty for first N steps
DEPLOYMENT_BONUS_SCALE = 0.5      # Max bonus for full deployment
HOLDING_BONUS_MAX = 1.0           # Max bonus for holding positions
RR_QUALITY_SCALE = 1.0            # Risk-reward quality scaling
WINRATE_BONUS_SCALE = 2.0         # Win rate bonus scaling
CASH_HOARD_THRESHOLD = 0.9        # Cash % above which hoarding penalty applies
CASH_HOARD_GRACE_STEPS = 20       # Steps before hoarding penalty kicks in
CASH_HOARD_PENALTY = 0.5          # Penalty per step when hoarding
STALE_THRESHOLD = 0.03            # Turnover below which is considered "stale"
STALE_GRACE_STEPS = 30            # Steps before staleness penalty
STALE_PENALTY = 0.3               # Penalty per step when stale

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
        - 7: Stop Loss Multiplier (2.0 - 5.0 ATR)
        - 8: Take Profit Multiplier (2.0 - 10.0 ATR)
        
    Observation Space: Continuous (97,) - Market Features + Portfolio State
    
    Reward System (v4.0 - Balanced):
        - Primary: Portfolio return (scaled)
        - Secondary: Closed trade P&L
        - Shaping: RR quality, win rate, holding bonus, deployment bonus
        - Penalties: Turnover (with warmup), cash hoarding, staleness
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir="./", volatility_file="volatility_baseline.json", verbose=False):
        super(TradingEnv, self).__init__()
        
        self.data_dir = data_dir
        self.assets = ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']
        self.n_assets = len(self.assets)
        self.verbose = verbose
        
        # Load Data
        self.data = self._load_data()
        self.timestamps = self.data.index
        self.n_steps = len(self.data)
        
        # Load Volatility Baselines
        try:
            with open(volatility_file, 'r') as f:
                self.volatility_baseline = json.load(f)
        except FileNotFoundError:
            print(f"Warning: {volatility_file} not found. Using defaults.")
            self.volatility_baseline = {asset: 0.02 for asset in self.assets}
            
        # Define Spaces
        # Action: 9 dims (7 weights + 1 SL + 1 TP)
        self.action_space = spaces.Box(low=0, high=1, shape=(9,), dtype=np.float32)
        
        # Observation: 97 features
        # 78 Market + 3 Temporal + 6 Session + 8 Portfolio + 2 Cross-Asset
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(97,), dtype=np.float32)
        
        # Initialize state
        self._init_state()
        
        # Print diagnostic info
        if self.verbose:
            self.diagnose_training_data()

    def _init_state(self):
        """Initialize all state variables."""
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.cash = INITIAL_BALANCE
        self.positions = {asset: 0.0 for asset in self.assets}
        self.holdings = {asset: 0.0 for asset in self.assets}
        self.entry_prices = {asset: 0.0 for asset in self.assets}
        self.previous_weights = np.zeros(self.n_assets + 1)
        self.previous_weights[6] = 1.0  # Start with 100% cash
        
        # Position Age Tracking
        self.position_ages = {asset: 0 for asset in self.assets}
        
        # Anti-Cheat Tracking
        self.cash_hoarding_steps = 0
        self.stale_steps = 0
        
        # Performance Tracking
        self.peak_value = INITIAL_BALANCE
        self.tp_hit_count = 0
        self.sl_hit_count = 0
        self.total_fees_paid = 0.0
        
        # History
        self.history = []
        self.trade_history = []
        self.reward_components_history = []
        
        # Risk params (will be set by action)
        self.sl_multiplier = 3.0
        self.tp_multiplier = 6.0

    def _load_data(self):
        """Loads and aligns parquet files for all assets, calculates cross-asset features."""
        dfs = {}
        
        # Load all asset files
        for asset in self.assets:
            try:
                try:
                    df = pd.read_parquet(f"{self.data_dir}data_{asset}_final.parquet")
                except FileNotFoundError:
                    print(f"Warning: data_{asset}_final.parquet not found. Trying backtest_data_{asset}.parquet...")
                    df = pd.read_parquet(f"{self.data_dir}backtest_data_{asset}.parquet")
                
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
        
        # Define Market Features (13 per asset)
        market_suffixes = [
            'log_ret', 'dist_ema50', 'atr_14_norm', 'bb_width', 'rsi_14_norm', 
            'macd_norm', 'vol_ratio', 'adx_norm',  # 8 (15m)
            'rsi_4h', 'dist_ema50_4h', 'atr_4h_norm',  # 3 (4H)
            'dist_ema200_1d', 'rsi_1d'  # 2 (Daily)
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
                    full_df[f"{asset}_{suffix}"] = 0.0
                    
        # 2. Add Temporal Features (Take from BTC)
        btc_df = dfs['BTC'].loc[common_index]
        full_df['sin_hour'] = btc_df.get('sin_hour', 0.0)
        full_df['cos_hour'] = btc_df.get('cos_hour', 0.0)
        full_df['day_of_week'] = btc_df.get('day_of_week', 0.0)
        
        # 3. Add Session Features
        session_cols = ['is_btc_tradeable', 'is_eth_tradeable', 'is_sol_tradeable', 
                        'is_eur_tradeable', 'is_gbp_tradeable', 'is_jpy_tradeable']
        for col in session_cols:
            if col in btc_df.columns:
                full_df[col] = btc_df[col]
            else:
                full_df[col] = 1.0
                
        # 4. Calculate Cross-Asset Features
        c_corr = full_df['BTC_log_ret'].rolling(window=20).corr(full_df['ETH_log_ret'])
        full_df['crypto_correlation'] = c_corr.fillna(0.0)
        
        crypto_mom = (full_df['BTC_rsi_14_norm'] + full_df['ETH_rsi_14_norm'] + full_df['SOL_rsi_14_norm']) / 3.0
        forex_mom = (full_df['EUR_rsi_14_norm'] + full_df['GBP_rsi_14_norm'] + full_df['JPY_rsi_14_norm']) / 3.0
        full_df['crypto_forex_divergence'] = crypto_mom - forex_mom
        
        # Fill NaNs
        full_df.fillna(0.0, inplace=True)
        
        return full_df

    def diagnose_training_data(self):
        """Print diagnostic information about the training data."""
        print("\n" + "="*60)
        print("TRAINING DATA DIAGNOSTICS")
        print("="*60)
        
        for asset in self.assets:
            prices = self.data[f"{asset}_close"]
            start_price = prices.iloc[0]
            end_price = prices.iloc[-1]
            min_price = prices.min()
            max_price = prices.max()
            change_pct = (end_price - start_price) / start_price * 100
            volatility = prices.pct_change().std() * 100
            
            print(f"{asset:4s}: {change_pct:+7.1f}% | "
                  f"Vol: {volatility:.2f}% | "
                  f"Range: {min_price:.2f} - {max_price:.2f}")
        
        print(f"\nTotal steps: {len(self.data)} ({len(self.data)/4:.0f} hours / {len(self.data)/96:.1f} days)")
        print("="*60 + "\n")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()
        return self._get_observation(), {}

    def step(self, action):
        # 1. Enforce Action Masking (Session Constraints)
        mask = self.action_masks()
        
        # Extract Weights (0-6) and Risk Params (7-8)
        raw_weights = action[0:7]
        sl_mult_raw = action[7]
        tp_mult_raw = action[8]
        
        # Decode Risk Params with HARD ENFORCEMENT
        sl_mult_clipped = np.clip(sl_mult_raw, 0.25, 1.0)
        tp_mult_clipped = np.clip(tp_mult_raw, 0.12, 1.0)
        
        # SL: 2.0 to 5.0 ATR
        self.sl_multiplier = 2.0 + (sl_mult_clipped * 3.0)
        # TP: 2.0 to 10.0 ATR
        self.tp_multiplier = 2.0 + (tp_mult_clipped * 8.0)

        # Apply Softmax with Masking to Weights
        exp_action = np.exp(np.clip(raw_weights, -10, 10))  # Clip to prevent overflow
        masked_exp = exp_action * mask[0:7]
        sum_exp = np.sum(masked_exp)
        
        if sum_exp == 0:
            weights = np.zeros_like(raw_weights)
            weights[6] = 1.0
        else:
            weights = masked_exp / sum_exp
        
        # --- ENFORCE CONSTRAINTS ---
        # 1. Min Cash 5%
        if weights[6] < MIN_CASH_PCT:
            weights[6] = MIN_CASH_PCT
            risky_sum = np.sum(weights[0:6])
            if risky_sum > 0:
                weights[0:6] = weights[0:6] / risky_sum * (1.0 - MIN_CASH_PCT)
        
        # 2. Max Position 40%
        weights[0:6] = np.minimum(weights[0:6], MAX_POS_PCT)
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
        
        # --- AUTOMATIC STOP-LOSS & TAKE-PROFIT ---
        trade_profit_reward = 0.0
        trades_closed_this_step = 0
        
        for asset in self.assets:
            if self.holdings[asset] > 0:
                entry_price = self.entry_prices[asset]
                current_price = prices[asset]
                
                # Recover ATR
                atr_norm = current_data.get(f"{asset}_atr_14_norm", 0.01)
                atr = max(atr_norm * current_price, current_price * 0.001)  # Min 0.1% ATR
                
                # Calculate SL/TP levels
                stop_loss = entry_price - (self.sl_multiplier * atr)
                take_profit = entry_price + (self.tp_multiplier * atr)
                
                if current_price <= stop_loss or current_price >= take_profit:
                    # Force Close Position
                    units = self.holdings[asset]
                    proceeds = units * current_price
                    
                    # Fee
                    fee_rate = TRANSACTION_FEE_CRYPTO if asset in ['BTC', 'ETH', 'SOL'] else TRANSACTION_FEE_FOREX
                    fee = proceeds * fee_rate
                    self.total_fees_paid += fee
                    
                    self.cash += (proceeds - fee)
                    self.holdings[asset] = 0.0
                    self.portfolio_value -= fee
                    
                    # Calculate profit (NO leverage distortion in reward)
                    if entry_price > 0:
                        cost_basis = units * entry_price
                        profit = proceeds - cost_basis - fee
                        
                        # Scaled trade reward: $100 profit = +1.0 reward
                        trade_profit_reward += profit * TRADE_PROFIT_SCALE
                        trades_closed_this_step += 1
                        
                        # Record trade
                        self.trade_history.append({
                            'asset': asset,
                            'entry': entry_price,
                            'exit': current_price,
                            'net_profit': profit,
                            'timestamp': self.timestamps[self.current_step],
                            'is_tp': current_price >= take_profit
                        })
                    
                    # Reset entry price
                    self.entry_prices[asset] = 0.0
                    self.position_ages[asset] = 0
                    
                    # Track hits for logging
                    if current_price <= stop_loss:
                        self.sl_hit_count += 1
                    else:
                        self.tp_hit_count += 1
                    
                    # Override target for this asset to 0
                    asset_idx = self.assets.index(asset)
                    removed_weight = weights[asset_idx]
                    weights[asset_idx] = 0.0
                    weights[6] += removed_weight
        
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
                
                if diff > 0:  # Buy
                    units_to_buy = diff / prices[asset]
                    current_units = self.holdings[asset]
                    total_units = current_units + units_to_buy
                    
                    if total_units > 0:
                        if current_units > 0 and self.entry_prices[asset] > 0:
                            avg_price = ((current_units * self.entry_prices[asset]) + 
                                        (units_to_buy * prices[asset])) / total_units
                        else:
                            avg_price = prices[asset]
                        self.entry_prices[asset] = avg_price
                        
                    self.holdings[asset] += units_to_buy
                    self.cash -= diff
                    
                    # Reset age if opening new position
                    if current_pos_value < MIN_TRADE_SIZE:
                        self.position_ages[asset] = 0
                else:  # Sell
                    units_to_sell = abs(diff) / prices[asset]
                    self.holdings[asset] -= units_to_sell
                    self.cash += abs(diff)
                    
                    if self.holdings[asset] <= 1e-6:
                        self.entry_prices[asset] = 0.0
                        self.position_ages[asset] = 0
            else:
                # Holding position: increment age
                if current_pos_value > MIN_TRADE_SIZE:
                    self.position_ages[asset] += 1
        
        # Deduct Fees
        self.cash -= total_fees
        self.portfolio_value -= total_fees
        self.total_fees_paid += total_fees
        
        # ================================================================
        # 5. CALCULATE REWARD - BALANCED SYSTEM v4.0
        # ================================================================
        
        # --- TURNOVER CALCULATION ---
        turnover = np.sum(np.abs(weights - self.previous_weights)) / 2.0
        self.previous_weights = weights.copy()
        
        # --- TURNOVER PENALTY (with warmup) ---
        if self.current_step < WARMUP_STEPS:
            # No penalty during warmup (let agent deploy capital)
            turnover_penalty = 0.0
        else:
            if turnover > 0.5:
                turnover_penalty = turnover * TURNOVER_PENALTY_HEAVY
            elif turnover > 0.2:
                turnover_penalty = turnover * TURNOVER_PENALTY_MEDIUM
            else:
                turnover_penalty = turnover * TURNOVER_PENALTY_LIGHT
        
        # --- PORTFOLIO RETURN (PRIMARY SIGNAL) ---
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-9)
        return_reward = portfolio_return * RETURN_REWARD_SCALE
        
        # --- RISK-REWARD QUALITY ---
        rr_ratio = self.tp_multiplier / (self.sl_multiplier + 1e-9)
        # Linear reward: RR=2 -> 0, RR=3 -> +1, RR=4 -> +2, RR=1 -> -1
        rr_quality = (rr_ratio - 2.0) * RR_QUALITY_SCALE
        rr_quality = float(np.clip(rr_quality, -2.0, 3.0))
        
        # --- WIN RATE BONUS ---
        if len(self.trade_history) >= 5:
            recent_trades = self.trade_history[-50:]  # Last 50 trades
            wins = sum(1 for t in recent_trades if t['net_profit'] > 0)
            win_rate = wins / len(recent_trades)
            winrate_bonus = (win_rate - 0.5) * WINRATE_BONUS_SCALE
        else:
            winrate_bonus = 0.0
            win_rate = 0.0
        
        # --- HOLDING BONUS ---
        active_positions = sum(1 for a in self.assets if self.holdings[a] > 0)
        if active_positions > 0:
            avg_holding = sum(self.position_ages.values()) / active_positions
            # Linear bonus up to max at 16 steps (4 hours at 15min bars)
            holding_bonus = min(HOLDING_BONUS_MAX, avg_holding / 16.0)
        else:
            avg_holding = 0
            holding_bonus = 0.0
        
        # --- DEPLOYMENT BONUS ---
        cash_weight = weights[6]
        deployed_pct = 1.0 - cash_weight
        deployment_bonus = deployed_pct * DEPLOYMENT_BONUS_SCALE
        
        # --- ANTI-CASH-HOARDING PENALTY ---
        if cash_weight > CASH_HOARD_THRESHOLD:
            self.cash_hoarding_steps += 1
        else:
            self.cash_hoarding_steps = 0
        
        if self.cash_hoarding_steps > CASH_HOARD_GRACE_STEPS:
            cash_hoarding_penalty = CASH_HOARD_PENALTY
        else:
            cash_hoarding_penalty = 0.0
        
        # --- ANTI-STALENESS PENALTY ---
        if turnover < STALE_THRESHOLD and self.current_step >= WARMUP_STEPS:
            self.stale_steps += 1
        else:
            self.stale_steps = 0
        
        if self.stale_steps > STALE_GRACE_STEPS:
            staleness_penalty = STALE_PENALTY
        else:
            staleness_penalty = 0.0
        
        # --- FINAL REWARD COMPOSITION ---
        final_reward = (
            return_reward +           # Primary: portfolio change
            trade_profit_reward +     # Closed trade P&L
            rr_quality +              # Risk-reward quality
            winrate_bonus +           # Trade win rate
            holding_bonus +           # Patience bonus
            deployment_bonus -        # Encourage capital use
            turnover_penalty -        # Trading cost
            cash_hoarding_penalty -   # Anti-hoarding
            staleness_penalty         # Anti-stagnation
        )
        
        # Clip to prevent extreme values
        final_reward = float(np.clip(final_reward, -50.0, 50.0))
        
        # Store reward components for debugging
        reward_components = {
            'return_reward': return_reward,
            'trade_profit': trade_profit_reward,
            'rr_quality': rr_quality,
            'winrate_bonus': winrate_bonus,
            'holding_bonus': holding_bonus,
            'deployment_bonus': deployment_bonus,
            'turnover_penalty': -turnover_penalty,
            'cash_hoard_penalty': -cash_hoarding_penalty,
            'stale_penalty': -staleness_penalty,
            'total': final_reward
        }
        self.reward_components_history.append(reward_components)
        
        # Debug Logging
        if self.current_step % 500 == 0:
            print(f"step={self.current_step} pv={self.portfolio_value:.2f} "
                  f"reward={final_reward:.3f} ret_r={return_reward:.3f} "
                  f"turnover={turnover:.3f} to_pen={turnover_penalty:.3f} "
                  f"deployed={deployed_pct:.1%} trades={len(self.trade_history)}")
        
        # 6. Check Termination
        terminated = False
        truncated = False
        
        # Terminate on catastrophic loss (30% of initial capital lost)
        if self.portfolio_value < INITIAL_BALANCE * 0.7:
            terminated = True
            final_reward = -50.0  # Terminal penalty
        
        # Track peak value
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        
        # Advance step
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            truncated = True
        
        # Info dict
        info = {
            'portfolio_value': self.portfolio_value,
            'fees': total_fees,
            'total_fees': self.total_fees_paid,
            'return': portfolio_return,
            'cumulative_return': (self.portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE,
            'drawdown': drawdown,
            'n_trades': len(self.trade_history),
            'tp_hits': self.tp_hit_count,
            'sl_hits': self.sl_hit_count,
            'deployed_pct': deployed_pct,
            'reward_components': reward_components
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
            'log_ret', 'dist_ema50', 'atr_14_norm', 'bb_width', 'rsi_14_norm', 
            'macd_norm', 'vol_ratio', 'adx_norm',
            'rsi_4h', 'dist_ema50_4h', 'atr_4h_norm',
            'dist_ema200_1d', 'rsi_1d'
        ]
        
        for asset in self.assets:
            for suffix in market_suffixes:
                val = row.get(f"{asset}_{suffix}", 0.0)
                # Clip extreme values for stability
                val = np.clip(val, -10.0, 10.0)
                market_features.append(val)
                
        # 2. Temporal Features (3)
        temporal_features = [
            row.get('sin_hour', 0.0),
            row.get('cos_hour', 0.0),
            row.get('day_of_week', 0.0) / 6.0  # Normalize to 0-1
        ]
        
        # 3. Session Features (6)
        session_cols = ['is_btc_tradeable', 'is_eth_tradeable', 'is_sol_tradeable', 
                        'is_eur_tradeable', 'is_gbp_tradeable', 'is_jpy_tradeable']
        session_features = [float(row.get(col, 1.0)) for col in session_cols]
        
        # 4. Portfolio State (8)
        # Calculate current weights safely
        current_weights = []
        for a in self.assets:
            pos_value = self.holdings[a] * row[f"{a}_close"]
            weight = pos_value / (self.portfolio_value + 1e-9)
            current_weights.append(np.clip(weight, 0.0, 1.0))
        
        cash_weight = self.cash / (self.portfolio_value + 1e-9)
        current_weights.append(np.clip(cash_weight, 0.0, 1.0))
        
        # Unrealized PnL (normalized by initial balance)
        unrealized_pnl = (self.portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE
        unrealized_pnl = np.clip(unrealized_pnl, -1.0, 2.0)  # Cap at -100% to +200%
        
        portfolio_features = current_weights + [unrealized_pnl]
        
        # 5. Cross-Asset Features (2)
        cross_asset_features = [
            np.clip(row.get('crypto_correlation', 0.0), -1.0, 1.0),
            np.clip(row.get('crypto_forex_divergence', 0.0), -2.0, 2.0)
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
            obs = np.resize(obs, (97,))
        
        # Replace any NaN/Inf with 0
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
            
        return obs

    def action_masks(self):
        """
        Returns a boolean mask for valid actions.
        """
        current_time = self.timestamps[self.current_step]
        hour = current_time.hour
        weekday = current_time.weekday()
        
        mask = np.ones(9, dtype=bool)
        
        # Forex Constraints
        is_weekend = weekday >= 5
        
        if is_weekend:
            mask[3] = False  # EUR
            mask[4] = False  # GBP
            mask[5] = False  # JPY
        else:
            # EUR/GBP (London/NY): 07-21 UTC
            if not (7 <= hour < 21):
                mask[3] = False
                mask[4] = False
            # JPY (Tokyo/London): 23-16 UTC
            if not (hour >= 23 or hour < 16):
                mask[5] = False
            
        return mask

    def _get_action_mask(self):
        """Deprecated internal method, keeping for compatibility."""
        return self.action_masks()
    
    def get_performance_summary(self):
        """Get a summary of performance metrics."""
        if len(self.trade_history) == 0:
            return {
                'total_return': (self.portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE,
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'total_fees': self.total_fees_paid
            }
        
        profits = [t['net_profit'] for t in self.trade_history]
        wins = sum(1 for p in profits if p > 0)
        
        return {
            'total_return': (self.portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE,
            'n_trades': len(self.trade_history),
            'win_rate': wins / len(self.trade_history),
            'avg_profit': np.mean(profits),
            'total_profit': sum(profits),
            'total_fees': self.total_fees_paid,
            'profit_factor': sum(p for p in profits if p > 0) / (abs(sum(p for p in profits if p < 0)) + 1e-9),
            'max_drawdown': (self.peak_value - min(self.history if self.history else [self.portfolio_value])) / self.peak_value if self.peak_value > 0 else 0
        }
    
    def render(self, mode='human'):
        """Render the environment state."""
        if mode == 'human':
            print(f"\n--- Step {self.current_step} ---")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(f"Cash: ${self.cash:.2f}")
            print(f"Positions:")
            for asset in self.assets:
                if self.holdings[asset] > 0:
                    current_price = self.data.iloc[self.current_step][f"{asset}_close"]
                    pos_value = self.holdings[asset] * current_price
                    pnl = (current_price - self.entry_prices[asset]) / self.entry_prices[asset] * 100 if self.entry_prices[asset] > 0 else 0
                    print(f"  {asset}: {self.holdings[asset]:.6f} units (${pos_value:.2f}, PnL: {pnl:+.2f}%)")
            print(f"Total Trades: {len(self.trade_history)} (TP: {self.tp_hit_count}, SL: {self.sl_hit_count})")
            print(f"Total Fees Paid: ${self.total_fees_paid:.2f}")


# ===== HELPER FUNCTION FOR TESTING =====
def test_environment():
    """Quick test to ensure the environment works."""
    print("Testing TradingEnv...")
    
    try:
        env = TradingEnv(data_dir="./", verbose=True)
        print(f"✓ Environment created successfully")
        print(f"  - Observation space: {env.observation_space.shape}")
        print(f"  - Action space: {env.action_space.shape}")
        print(f"  - Data steps: {env.n_steps}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        # Test a few steps with random actions
        total_reward = 0
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"  Episode ended at step {i}")
                break
        
        print(f"✓ Ran 100 steps successfully")
        print(f"  - Final PV: ${info['portfolio_value']:.2f}")
        print(f"  - Total reward: {total_reward:.2f}")
        print(f"  - Trades: {info['n_trades']}")
        
        # Test performance summary
        summary = env.get_performance_summary()
        print(f"✓ Performance Summary:")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  - {k}: {v:.4f}")
            else:
                print(f"  - {k}: {v}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_environment()
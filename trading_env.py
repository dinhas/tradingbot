"""
TradingEnv v5.0 - Production Ready
==================================
Multi-Asset Trading Environment for Reinforcement Learning

FIXED ISSUES:
- Reward now properly dominated by actual performance
- Shaping rewards are conditional on profitability
- Added drawdown protection
- Removed harmful cash hoarding penalty
- Cumulative return context added to reward

Author: [Your Name]
Last Updated: 2024
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import json
import logging
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

# Portfolio Settings
INITIAL_BALANCE = 10000.0
LEVERAGE = 200.0  # Display only for forex/crypto CFDs

# Transaction Fees
TRANSACTION_FEE_CRYPTO = 0.001   # 0.1%
TRANSACTION_FEE_FOREX = 0.0005   # 0.05%

# Position Constraints
MIN_TRADE_SIZE = 100.0
MIN_CASH_PCT = 0.05
MAX_POS_PCT = 0.40

# Risk Management
DEFAULT_SL_MULTIPLIER = 3.0
DEFAULT_TP_MULTIPLIER = 6.0
SL_MIN = 2.0
SL_MAX = 5.0
TP_MIN = 2.0
TP_MAX = 10.0

# Reward Configuration (v5.0 - Performance Dominated)
class RewardConfig:
    """Centralized reward configuration for easy tuning."""
    
    # === PRIMARY: Return Rewards (MUST DOMINATE) ===
    STEP_RETURN_SCALE = 500.0       # Weight for step-by-step return
    CUMULATIVE_RETURN_SCALE = 100.0  # Weight for cumulative performance
    RETURN_CLIP = 30.0               # Max absolute return reward
    
    # === SECONDARY: Conditional Shaping ===
    # These only apply when cumulative return > PROFIT_THRESHOLD
    PROFIT_THRESHOLD = 0.01          # 1% profit before shaping kicks in
    
    # Risk-Reward Quality (only when profitable)
    RR_QUALITY_SCALE = 0.15          # Max RR quality bonus
    RR_BASELINE = 1.5                # Minimum acceptable RR ratio
    
    # Win Rate (only when enough trades)
    WINRATE_SCALE = 0.3              # Max winrate bonus
    WINRATE_MIN_TRADES = 15          # Min trades before winrate matters
    WINRATE_BASELINE = 0.45          # Target winrate
    
    # Deployment (only when profitable)
    DEPLOYMENT_SCALE = 0.2           # Max deployment bonus
    DEPLOYMENT_TARGET = 0.5          # Target deployment ratio
    
    # === PENALTIES ===
    # Turnover
    WARMUP_STEPS = 50                # No turnover penalty during warmup
    TURNOVER_POWER = 1.5             # Turnover^power for penalty
    TURNOVER_SCALE = 4.0             # Turnover penalty multiplier
    
    # Drawdown Protection
    DRAWDOWN_THRESHOLD = 0.08        # 8% drawdown before penalty
    DRAWDOWN_SCALE = 15.0            # Aggressive drawdown penalty
    
    # Catastrophic Loss
    CATASTROPHIC_THRESHOLD = 0.30    # 30% loss = termination
    CATASTROPHIC_PENALTY = -100.0    # Terminal penalty
    
    # Staleness (inactivity)
    STALENESS_THRESHOLD = 0.00005    # Min absolute return to avoid staleness
    STALENESS_PENALTY = 0.02         # Small nudge to trade
    STALENESS_GRACE_STEPS = 100      # Steps before staleness matters
    
    # Risk Parameter Bounds
    RISK_PARAM_PENALTY_SCALE = 0.1   # Penalty for extreme SL/TP


# Asset Configuration
ASSET_MAPPING = {
    0: 'BTC',
    1: 'ETH', 
    2: 'SOL',
    3: 'EUR',
    4: 'GBP',
    5: 'JPY',
    6: 'CASH'
}

CRYPTO_ASSETS = {'BTC', 'ETH', 'SOL'}
FOREX_ASSETS = {'EUR', 'GBP', 'JPY'}


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ============================================================================
# MAIN ENVIRONMENT CLASS
# ============================================================================

class TradingEnv(gym.Env):
    """
    Multi-Asset Trading Environment for Reinforcement Learning.
    
    Action Space: Continuous (9,)
        - [0-6]: Portfolio Weights [BTC, ETH, SOL, EUR, GBP, JPY, CASH]
        - [7]: Stop Loss Multiplier (2.0 - 5.0 ATR)
        - [8]: Take Profit Multiplier (2.0 - 10.0 ATR)
        
    Observation Space: Continuous (97,)
        - Market Features (78): 13 features × 6 assets
        - Temporal Features (3): sin_hour, cos_hour, day_of_week
        - Session Features (6): tradeable flags per asset
        - Portfolio State (8): 7 weights + unrealized PnL
        - Cross-Asset Features (2): correlation, divergence
    
    Reward System v5.0:
        - Primary: Portfolio return (step + cumulative) - DOMINANT
        - Secondary: Conditional shaping (only when profitable)
        - Penalties: Turnover, drawdown, staleness
    """
    
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(
        self,
        data_dir: str = "./",
        volatility_file: str = "volatility_baseline.json",
        verbose: bool = False,
        reward_config: Optional[RewardConfig] = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            data_dir: Directory containing data files
            volatility_file: JSON file with volatility baselines
            verbose: Enable verbose logging
            reward_config: Custom reward configuration (uses default if None)
        """
        super(TradingEnv, self).__init__()
        
        # Configuration
        self.data_dir = data_dir
        self.assets = list(CRYPTO_ASSETS | FOREX_ASSETS)
        self.assets = ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']  # Fixed order
        self.n_assets = len(self.assets)
        self.verbose = verbose
        self.reward_config = reward_config or RewardConfig()
        
        # Logger
        self.logger = setup_logger(
            'TradingEnv', 
            logging.DEBUG if verbose else logging.WARNING
        )
        
        # Load Data
        self.data = self._load_data()
        self.timestamps = self.data.index
        self.n_steps = len(self.data)
        
        # Load Volatility Baselines
        self.volatility_baseline = self._load_volatility(volatility_file)
        
        # Define Spaces
        self.action_space = spaces.Box(
            low=0, high=1, shape=(9,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(97,), dtype=np.float32
        )
        
        # Initialize State
        self._init_state()
        
        # Diagnostics
        if self.verbose:
            self._print_diagnostics()

    def _load_volatility(self, volatility_file: str) -> Dict[str, float]:
        """Load volatility baselines from JSON file."""
        try:
            with open(volatility_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(
                f"{volatility_file} not found. Using default volatility."
            )
            return {asset: 0.02 for asset in self.assets}

    def _init_state(self) -> None:
        """Initialize/reset all state variables."""
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.cash = INITIAL_BALANCE
        
        # Position Tracking
        self.holdings = {asset: 0.0 for asset in self.assets}
        self.entry_prices = {asset: 0.0 for asset in self.assets}
        self.position_ages = {asset: 0 for asset in self.assets}
        
        # Weight Tracking
        self.previous_weights = np.zeros(self.n_assets + 1)
        self.previous_weights[6] = 1.0  # 100% cash initially
        self.current_weights = self.previous_weights.copy()
        
        # Risk Parameters
        self.sl_multiplier = DEFAULT_SL_MULTIPLIER
        self.tp_multiplier = DEFAULT_TP_MULTIPLIER
        
        # Performance Tracking
        self.peak_value = INITIAL_BALANCE
        self.tp_hit_count = 0
        self.sl_hit_count = 0
        self.total_fees_paid = 0.0
        
        # History
        self.portfolio_history = [INITIAL_BALANCE]
        self.trade_history: List[Dict] = []
        self.reward_history: List[Dict] = []
        
        # Reward tracking
        self.cumulative_reward = 0.0
        self.episode_return = 0.0

    def _load_data(self) -> pd.DataFrame:
        """Load and align data for all assets."""
        dfs = {}
        
        # Load each asset's data
        for asset in self.assets:
            df = self._load_asset_data(asset)
            if df is not None:
                dfs[asset] = df
        
        if not dfs:
            raise FileNotFoundError("No asset data files found!")
        
        # Find common index
        common_index = dfs[self.assets[0]].index
        for asset in self.assets[1:]:
            if asset in dfs:
                common_index = common_index.intersection(dfs[asset].index)
        
        if len(common_index) == 0:
            raise ValueError("No overlapping timestamps across assets.")
        
        self.logger.info(f"Common index: {len(common_index)} steps")
        
        # Build merged DataFrame
        full_df = self._merge_asset_data(dfs, common_index)
        
        return full_df

    def _load_asset_data(self, asset: str) -> Optional[pd.DataFrame]:
        """Load data for a single asset."""
        filenames = [
            f"{self.data_dir}data_{asset}_final.parquet",
            f"{self.data_dir}backtest_data_{asset}.parquet",
            f"{self.data_dir}{asset}_data.parquet"
        ]
        
        for filename in filenames:
            try:
                df = pd.read_parquet(filename)
                self.logger.info(f"Loaded {asset} from {filename}")
                return df
            except FileNotFoundError:
                continue
        
        raise FileNotFoundError(f"No data file found for {asset}")

    def _merge_asset_data(
        self, 
        dfs: Dict[str, pd.DataFrame], 
        common_index: pd.Index
    ) -> pd.DataFrame:
        """Merge all asset data into a single DataFrame."""
        full_df = pd.DataFrame(index=common_index)
        
        # Market feature suffixes (13 per asset)
        market_suffixes = [
            'log_ret', 'dist_ema50', 'atr_14_norm', 'bb_width', 'rsi_14_norm',
            'macd_norm', 'vol_ratio', 'adx_norm',  # 15m features
            'rsi_4h', 'dist_ema50_4h', 'atr_4h_norm',  # 4H features
            'dist_ema200_1d', 'rsi_1d'  # Daily features
        ]
        
        # Add asset data
        for asset in self.assets:
            df = dfs[asset].loc[common_index]
            
            # Price (critical)
            full_df[f"{asset}_close"] = df['close']
            
            # Market features
            for suffix in market_suffixes:
                col_name = f"{asset}_{suffix}"
                full_df[col_name] = df.get(suffix, 0.0)
        
        # Temporal features (from first asset)
        first_df = dfs[self.assets[0]].loc[common_index]
        full_df['sin_hour'] = first_df.get('sin_hour', 0.0)
        full_df['cos_hour'] = first_df.get('cos_hour', 0.0)
        full_df['day_of_week'] = first_df.get('day_of_week', 0.0)
        
        # Session features
        session_cols = [
            'is_btc_tradeable', 'is_eth_tradeable', 'is_sol_tradeable',
            'is_eur_tradeable', 'is_gbp_tradeable', 'is_jpy_tradeable'
        ]
        for col in session_cols:
            full_df[col] = first_df.get(col, 1.0)
        
        # Cross-asset features
        full_df['crypto_correlation'] = (
            full_df['BTC_log_ret']
            .rolling(window=20)
            .corr(full_df['ETH_log_ret'])
            .fillna(0.0)
        )
        
        crypto_mom = (
            full_df['BTC_rsi_14_norm'] + 
            full_df['ETH_rsi_14_norm'] + 
            full_df['SOL_rsi_14_norm']
        ) / 3.0
        
        forex_mom = (
            full_df['EUR_rsi_14_norm'] + 
            full_df['GBP_rsi_14_norm'] + 
            full_df['JPY_rsi_14_norm']
        ) / 3.0
        
        full_df['crypto_forex_divergence'] = crypto_mom - forex_mom
        
        # Fill NaN
        full_df.fillna(0.0, inplace=True)
        
        return full_df

    def _print_diagnostics(self) -> None:
        """Print diagnostic information about the data."""
        print("\n" + "=" * 70)
        print("TRADING ENVIRONMENT DIAGNOSTICS")
        print("=" * 70)
        
        for asset in self.assets:
            prices = self.data[f"{asset}_close"]
            change_pct = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
            volatility = prices.pct_change().std() * 100
            
            print(f"{asset:4s}: {change_pct:+8.2f}% change | "
                  f"Vol: {volatility:.3f}% | "
                  f"Range: {prices.min():.2f} - {prices.max():.2f}")
        
        hours = len(self.data) / 4
        days = hours / 24
        print(f"\nTotal: {len(self.data)} steps ({hours:.0f} hours / {days:.1f} days)")
        print("=" * 70 + "\n")

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self._init_state()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Array of shape (9,) containing weights and risk params
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Store previous state
        prev_portfolio_value = self.portfolio_value
        prev_weights = self.previous_weights.copy()
        
        # 1. Process Action
        weights = self._process_action(action)
        
        # 2. Get Current Prices
        current_data = self.data.iloc[self.current_step]
        prices = self._get_prices(current_data)
        
        # 3. Update Portfolio Value (Mark-to-Market)
        self._update_portfolio_value(prices)
        
        # 4. Check and Execute Stop-Loss/Take-Profit
        sl_tp_trades = self._execute_sl_tp(current_data, prices, weights)
        
        # 5. Execute Rebalancing
        turnover = self._execute_rebalance(weights, prices)
        
        # 6. Calculate Reward
        reward, reward_components = self._calculate_reward(
            prev_portfolio_value=prev_portfolio_value,
            weights=weights,
            turnover=turnover,
            trades_this_step=sl_tp_trades
        )
        
        # 7. Update State
        self.previous_weights = weights.copy()
        self.current_weights = weights.copy()
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.portfolio_history.append(self.portfolio_value)
        self.reward_history.append(reward_components)
        self.cumulative_reward += reward
        
        # 8. Check Termination
        terminated, truncated = self._check_termination()
        
        # Apply terminal penalty if terminated due to losses
        if terminated:
            reward = self.reward_config.CATASTROPHIC_PENALTY
        
        # 9. Advance Step
        self.current_step += 1
        if self.current_step >= self.n_steps - 1:
            truncated = True
        
        # 10. Build Info Dict
        info = self._get_info()
        info['reward_components'] = reward_components
        
        # Debug Logging
        if self.current_step % 500 == 0:
            self._log_step_summary(reward_components)
        
        return self._get_observation(), float(reward), terminated, truncated, info

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process and validate action, returning normalized weights."""
        # Get action mask
        mask = self.action_masks()
        
        # Extract components
        raw_weights = action[0:7]
        sl_raw = action[7]
        tp_raw = action[8]
        
        # Decode risk parameters (clipped and scaled)
        sl_clipped = np.clip(sl_raw, 0.0, 1.0)
        tp_clipped = np.clip(tp_raw, 0.0, 1.0)
        
        self.sl_multiplier = SL_MIN + sl_clipped * (SL_MAX - SL_MIN)
        self.tp_multiplier = TP_MIN + tp_clipped * (TP_MAX - TP_MIN)
        
        # Ensure TP > SL for sensible RR
        if self.tp_multiplier <= self.sl_multiplier:
            self.tp_multiplier = self.sl_multiplier * 1.5
        
        # Apply softmax with masking
        exp_weights = np.exp(np.clip(raw_weights, -10, 10))
        masked_exp = exp_weights * mask[0:7]
        sum_exp = np.sum(masked_exp)
        
        if sum_exp < 1e-8:
            weights = np.zeros(7)
            weights[6] = 1.0
        else:
            weights = masked_exp / sum_exp
        
        # Enforce constraints
        weights = self._enforce_constraints(weights)
        
        return weights

    def _enforce_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Enforce position size and cash constraints."""
        # Min cash constraint
        if weights[6] < MIN_CASH_PCT:
            risky_sum = np.sum(weights[0:6])
            if risky_sum > 0:
                scale = (1.0 - MIN_CASH_PCT) / risky_sum
                weights[0:6] *= scale
            weights[6] = MIN_CASH_PCT
        
        # Max position constraint
        weights[0:6] = np.minimum(weights[0:6], MAX_POS_PCT)
        
        # Renormalize
        weights[6] = 1.0 - np.sum(weights[0:6])
        weights[6] = max(weights[6], MIN_CASH_PCT)
        
        # Final normalization
        total = np.sum(weights)
        if total > 0:
            weights /= total
        
        return weights

    def _get_prices(self, current_data: pd.Series) -> Dict[str, float]:
        """Extract current prices for all assets."""
        return {
            asset: current_data[f"{asset}_close"] 
            for asset in self.assets
        }

    def _update_portfolio_value(self, prices: Dict[str, float]) -> None:
        """Update portfolio value based on current prices."""
        self.portfolio_value = self.cash
        for asset in self.assets:
            if self.holdings[asset] > 0:
                self.portfolio_value += self.holdings[asset] * prices[asset]

    def _execute_sl_tp(
        self, 
        current_data: pd.Series,
        prices: Dict[str, float],
        weights: np.ndarray
    ) -> int:
        """Execute stop-loss and take-profit orders."""
        trades_closed = 0
        
        for i, asset in enumerate(self.assets):
            if self.holdings[asset] <= 0:
                continue
            
            entry_price = self.entry_prices[asset]
            if entry_price <= 0:
                continue
            
            current_price = prices[asset]
            
            # Calculate ATR-based levels
            atr_norm = current_data.get(f"{asset}_atr_14_norm", 0.01)
            atr = max(atr_norm * current_price, current_price * 0.001)
            
            stop_loss = entry_price - (self.sl_multiplier * atr)
            take_profit = entry_price + (self.tp_multiplier * atr)
            
            # Check triggers
            triggered = False
            is_tp = False
            
            if current_price <= stop_loss:
                triggered = True
                is_tp = False
                self.sl_hit_count += 1
            elif current_price >= take_profit:
                triggered = True
                is_tp = True
                self.tp_hit_count += 1
            
            if triggered:
                # Close position
                units = self.holdings[asset]
                proceeds = units * current_price
                
                # Calculate fee
                fee_rate = (
                    TRANSACTION_FEE_CRYPTO 
                    if asset in CRYPTO_ASSETS 
                    else TRANSACTION_FEE_FOREX
                )
                fee = proceeds * fee_rate
                self.total_fees_paid += fee
                
                # Update state
                self.cash += (proceeds - fee)
                self.holdings[asset] = 0.0
                
                # Calculate profit
                cost_basis = units * entry_price
                net_profit = proceeds - cost_basis - fee
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'asset': asset,
                    'entry': entry_price,
                    'exit': current_price,
                    'units': units,
                    'net_profit': net_profit,
                    'is_tp': is_tp,
                    'holding_time': self.position_ages[asset],
                    'timestamp': self.timestamps[self.current_step]
                })
                
                # Reset tracking
                self.entry_prices[asset] = 0.0
                self.position_ages[asset] = 0
                trades_closed += 1
                
                # Update target weight to 0
                weights[i] = 0.0
        
        # Renormalize weights after SL/TP
        if trades_closed > 0:
            weights[6] = 1.0 - np.sum(weights[0:6])
        
        # Update portfolio value after closes
        self.portfolio_value = self.cash
        for asset in self.assets:
            if self.holdings[asset] > 0:
                self.portfolio_value += self.holdings[asset] * prices[asset]
        
        return trades_closed

    def _execute_rebalance(
        self, 
        weights: np.ndarray, 
        prices: Dict[str, float]
    ) -> float:
        """Execute portfolio rebalancing and return turnover."""
        target_values = {
            asset: self.portfolio_value * weights[i]
            for i, asset in enumerate(self.assets)
        }
        
        total_traded = 0.0
        total_fees = 0.0
        
        for i, asset in enumerate(self.assets):
            current_value = self.holdings[asset] * prices[asset]
            target_value = target_values[asset]
            diff = target_value - current_value
            
            if abs(diff) < MIN_TRADE_SIZE:
                # Increment position age if holding
                if current_value > MIN_TRADE_SIZE:
                    self.position_ages[asset] += 1
                continue
            
            trade_value = abs(diff)
            total_traded += trade_value
            
            fee_rate = (
                TRANSACTION_FEE_CRYPTO 
                if asset in CRYPTO_ASSETS 
                else TRANSACTION_FEE_FOREX
            )
            fee = trade_value * fee_rate
            total_fees += fee
            
            if diff > 0:  # Buy
                units_to_buy = diff / prices[asset]
                current_units = self.holdings[asset]
                total_units = current_units + units_to_buy
                
                # Update average entry price
                if total_units > 0:
                    if current_units > 0 and self.entry_prices[asset] > 0:
                        avg_price = (
                            (current_units * self.entry_prices[asset] +
                             units_to_buy * prices[asset]) / total_units
                        )
                    else:
                        avg_price = prices[asset]
                        self.position_ages[asset] = 0  # New position
                    self.entry_prices[asset] = avg_price
                
                self.holdings[asset] += units_to_buy
                self.cash -= diff
                
            else:  # Sell
                units_to_sell = abs(diff) / prices[asset]
                self.holdings[asset] -= units_to_sell
                self.cash += abs(diff)
                
                # Reset if fully closed
                if self.holdings[asset] < 1e-9:
                    self.holdings[asset] = 0.0
                    self.entry_prices[asset] = 0.0
                    self.position_ages[asset] = 0
        
        # Deduct fees
        self.cash -= total_fees
        self.portfolio_value -= total_fees
        self.total_fees_paid += total_fees
        
        # Calculate turnover
        turnover = np.sum(np.abs(weights - self.previous_weights)) / 2.0
        
        return turnover

    def _calculate_reward(
        self,
        prev_portfolio_value: float,
        weights: np.ndarray,
        turnover: float,
        trades_this_step: int
    ) -> Tuple[float, Dict]:
        """
        Calculate reward with proper incentive alignment.
        
        Key Principles:
        1. Return MUST dominate all other signals
        2. Shaping rewards only apply when profitable
        3. Drawdown protection prevents catastrophic losses
        """
        cfg = self.reward_config
        
        # ================================================================
        # 1. PRIMARY: Return-Based Reward (DOMINANT)
        # ================================================================
        
        # Step return
        step_return = (
            (self.portfolio_value - prev_portfolio_value) / 
            max(prev_portfolio_value, 1e-8)
        )
        
        # Cumulative return
        cumulative_return = (
            (self.portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE
        )
        
        # Combined return reward
        # Step return is weighted more for responsiveness
        # Cumulative provides context about overall performance
        return_reward = (
            step_return * cfg.STEP_RETURN_SCALE +
            cumulative_return * cfg.CUMULATIVE_RETURN_SCALE / max(self.current_step + 1, 1)
        )
        
        # Clip to prevent extreme values
        return_reward = np.clip(
            return_reward, 
            -cfg.RETURN_CLIP, 
            cfg.RETURN_CLIP
        )
        
        # ================================================================
        # 2. SECONDARY: Conditional Shaping (Only When Profitable)
        # ================================================================
        
        is_profitable = cumulative_return > cfg.PROFIT_THRESHOLD
        
        # --- Risk-Reward Quality ---
        if is_profitable and self.sl_multiplier > 0:
            rr_ratio = self.tp_multiplier / self.sl_multiplier
            rr_quality = np.clip(
                (rr_ratio - cfg.RR_BASELINE) * 0.1,
                0,
                cfg.RR_QUALITY_SCALE
            )
        else:
            rr_quality = 0.0
        
        # --- Win Rate Reward ---
        total_trades = len(self.trade_history)
        if is_profitable and total_trades >= cfg.WINRATE_MIN_TRADES:
            wins = sum(1 for t in self.trade_history if t['net_profit'] > 0)
            winrate = wins / total_trades
            winrate_reward = np.clip(
                (winrate - cfg.WINRATE_BASELINE) * cfg.WINRATE_SCALE,
                -cfg.WINRATE_SCALE,
                cfg.WINRATE_SCALE
            )
        else:
            winrate_reward = 0.0
        
        # --- Deployment Reward ---
        deployment_ratio = 1.0 - weights[6]
        
        if is_profitable:
            # Reward deployment when making money
            deployment_reward = np.clip(
                (deployment_ratio - cfg.DEPLOYMENT_TARGET) * cfg.DEPLOYMENT_SCALE,
                -cfg.DEPLOYMENT_SCALE,
                cfg.DEPLOYMENT_SCALE
            )
        elif cumulative_return < -0.05:
            # Penalize deployment when losing significantly
            deployment_reward = -deployment_ratio * cfg.DEPLOYMENT_SCALE
        else:
            deployment_reward = 0.0
        
        # ================================================================
        # 3. PENALTIES
        # ================================================================
        
        # --- Turnover Penalty ---
        if self.current_step < cfg.WARMUP_STEPS:
            turnover_penalty = 0.0
        else:
            turnover_penalty = (
                (turnover ** cfg.TURNOVER_POWER) * cfg.TURNOVER_SCALE
            )
        
        # --- Drawdown Penalty ---
        drawdown = (
            (self.peak_value - self.portfolio_value) / 
            max(self.peak_value, 1e-8)
        )
        
        if drawdown > cfg.DRAWDOWN_THRESHOLD:
            drawdown_penalty = (
                (drawdown - cfg.DRAWDOWN_THRESHOLD) * cfg.DRAWDOWN_SCALE
            )
        else:
            drawdown_penalty = 0.0
        
        # --- Staleness Penalty ---
        if (self.current_step > cfg.STALENESS_GRACE_STEPS and 
            abs(step_return) < cfg.STALENESS_THRESHOLD):
            staleness_penalty = cfg.STALENESS_PENALTY
        else:
            staleness_penalty = 0.0
        
        # ================================================================
        # 4. FINAL REWARD CALCULATION
        # ================================================================
        
        final_reward = (
            return_reward +          # Primary: [-30, +30]
            rr_quality +             # Secondary: [0, 0.15]
            winrate_reward +         # Secondary: [-0.3, +0.3]
            deployment_reward -      # Secondary: [-0.2, +0.2]
            turnover_penalty -       # Penalty: [0, ~4]
            drawdown_penalty -       # Penalty: [0, ∞]
            staleness_penalty        # Penalty: [0, 0.02]
        )
        
        # Build components dict for debugging
        components = {
            'step_return': step_return,
            'cumulative_return': cumulative_return,
            'return_reward': return_reward,
            'rr_quality': rr_quality,
            'winrate_reward': winrate_reward,
            'deployment_reward': deployment_reward,
            'turnover': turnover,
            'turnover_penalty': turnover_penalty,
            'drawdown': drawdown,
            'drawdown_penalty': drawdown_penalty,
            'staleness_penalty': staleness_penalty,
            'final_reward': final_reward,
            'is_profitable': is_profitable,
            'trades_this_step': trades_this_step
        }
        
        return final_reward, components

    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
        terminated = False
        truncated = False
        
        # Catastrophic loss
        loss_pct = (INITIAL_BALANCE - self.portfolio_value) / INITIAL_BALANCE
        if loss_pct >= self.reward_config.CATASTROPHIC_THRESHOLD:
            terminated = True
            self.logger.warning(
                f"Episode terminated: {loss_pct*100:.1f}% loss at step {self.current_step}"
            )
        
        return terminated, truncated

    def _get_observation(self) -> np.ndarray:
        """Construct the 97-dimensional observation vector."""
        row = self.data.iloc[self.current_step]
        
        # 1. Market Features (78 = 13 × 6)
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
                val = np.clip(val, -10.0, 10.0)
                market_features.append(val)
        
        # 2. Temporal Features (3)
        temporal_features = [
            row.get('sin_hour', 0.0),
            row.get('cos_hour', 0.0),
            row.get('day_of_week', 0.0) / 6.0
        ]
        
        # 3. Session Features (6)
        session_cols = [
            'is_btc_tradeable', 'is_eth_tradeable', 'is_sol_tradeable',
            'is_eur_tradeable', 'is_gbp_tradeable', 'is_jpy_tradeable'
        ]
        session_features = [float(row.get(col, 1.0)) for col in session_cols]
        
        # 4. Portfolio State (8)
        portfolio_features = []
        for asset in self.assets:
            pos_value = self.holdings[asset] * row[f"{asset}_close"]
            weight = pos_value / max(self.portfolio_value, 1e-9)
            portfolio_features.append(np.clip(weight, 0.0, 1.0))
        
        cash_weight = self.cash / max(self.portfolio_value, 1e-9)
        portfolio_features.append(np.clip(cash_weight, 0.0, 1.0))
        
        unrealized_pnl = (self.portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE
        portfolio_features.append(np.clip(unrealized_pnl, -1.0, 2.0))
        
        # 5. Cross-Asset Features (2)
        cross_features = [
            np.clip(row.get('crypto_correlation', 0.0), -1.0, 1.0),
            np.clip(row.get('crypto_forex_divergence', 0.0), -2.0, 2.0)
        ]
        
        # Concatenate
        obs = np.concatenate([
            market_features,
            temporal_features,
            session_features,
            portfolio_features,
            cross_features
        ], dtype=np.float32)
        
        # Safety check
        assert len(obs) == 97, f"Observation size mismatch: {len(obs)} != 97"
        
        # Handle NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get current environment info."""
        cumulative_return = (
            (self.portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE
        )
        
        drawdown = (
            (self.peak_value - self.portfolio_value) / 
            max(self.peak_value, 1e-8)
        )
        
        deployed_pct = (
            (self.portfolio_value - self.cash) / 
            max(self.portfolio_value, 1e-9)
        ) * 100
        
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'cumulative_return': cumulative_return,
            'drawdown': drawdown,
            'deployed_pct': deployed_pct,
            'n_trades': len(self.trade_history),
            'tp_hits': self.tp_hit_count,
            'sl_hits': self.sl_hit_count,
            'total_fees': self.total_fees_paid,
            'cumulative_reward': self.cumulative_reward
        }

    def _log_step_summary(self, components: Dict) -> None:
        """Log a summary of the current step for debugging."""
        pv = self.portfolio_value
        cum_ret = components['cumulative_return'] * 100
        reward = components['final_reward']
        ret_r = components['return_reward']
        turn = components['turnover']
        dd = components['drawdown'] * 100
        deployed = (1 - self.current_weights[6]) * 100
        
        print(
            f"step={self.current_step:5d} | "
            f"PV=${pv:,.0f} ({cum_ret:+.1f}%) | "
            f"R={reward:+.2f} (ret={ret_r:+.2f}) | "
            f"turn={turn:.2f} dd={dd:.1f}% dep={deployed:.0f}% | "
            f"trades={len(self.trade_history)}"
        )

    def action_masks(self) -> np.ndarray:
        """Return action mask for valid actions based on trading sessions."""
        current_time = self.timestamps[self.current_step]
        hour = current_time.hour
        weekday = current_time.weekday()
        
        mask = np.ones(9, dtype=bool)
        
        # Forex weekend constraint
        is_weekend = weekday >= 5
        
        if is_weekend:
            mask[3] = False  # EUR
            mask[4] = False  # GBP
            mask[5] = False  # JPY
        else:
            # EUR/GBP: London/NY session (07-21 UTC)
            if not (7 <= hour < 21):
                mask[3] = False
                mask[4] = False
            
            # JPY: Tokyo/London session (23-16 UTC)
            if not (hour >= 23 or hour < 16):
                mask[5] = False
        
        return mask

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cumulative_return = (
            (self.portfolio_value - INITIAL_BALANCE) / INITIAL_BALANCE
        )
        
        if not self.trade_history:
            return {
                'total_return': cumulative_return,
                'n_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_profit': 0.0,
                'total_profit': 0.0,
                'total_fees': self.total_fees_paid,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'tp_rate': 0.0
            }
        
        profits = [t['net_profit'] for t in self.trade_history]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        profit_factor = (
            sum(wins) / abs(sum(losses)) 
            if losses else float('inf')
        )
        
        # Calculate Sharpe from portfolio history
        if len(self.portfolio_history) > 1:
            returns = pd.Series(self.portfolio_history).pct_change().dropna()
            sharpe = (
                returns.mean() / returns.std() * np.sqrt(252 * 24 * 4)  # Annualized
                if returns.std() > 0 else 0.0
            )
        else:
            sharpe = 0.0
        
        # Max drawdown
        peak = pd.Series(self.portfolio_history).expanding().max()
        drawdowns = (peak - self.portfolio_history) / peak
        max_dd = drawdowns.max()
        
        return {
            'total_return': cumulative_return,
            'n_trades': len(self.trade_history),
            'win_rate': len(wins) / len(self.trade_history),
            'profit_factor': profit_factor,
            'avg_profit': np.mean(profits),
            'total_profit': sum(profits),
            'total_fees': self.total_fees_paid,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'tp_rate': self.tp_hit_count / max(len(self.trade_history), 1),
            'avg_holding_time': np.mean([t['holding_time'] for t in self.trade_history])
        }

    def render(self, mode: str = 'human') -> Optional[str]:
        """Render the environment state."""
        output = []
        output.append(f"\n{'='*60}")
        output.append(f"Step {self.current_step} / {self.n_steps}")
        output.append(f"{'='*60}")
        output.append(f"Portfolio Value: ${self.portfolio_value:,.2f}")
        output.append(f"Cash: ${self.cash:,.2f}")
        output.append(f"Cumulative Return: {(self.portfolio_value/INITIAL_BALANCE-1)*100:+.2f}%")
        output.append("")
        output.append("Positions:")
        
        for asset in self.assets:
            if self.holdings[asset] > 0:
                price = self.data.iloc[self.current_step][f"{asset}_close"]
                value = self.holdings[asset] * price
                entry = self.entry_prices[asset]
                pnl_pct = (price - entry) / entry * 100 if entry > 0 else 0
                age = self.position_ages[asset]
                
                output.append(
                    f"  {asset}: {self.holdings[asset]:.6f} units "
                    f"(${value:,.2f}, {pnl_pct:+.2f}%, age={age})"
                )
        
        output.append("")
        output.append(f"Total Trades: {len(self.trade_history)}")
        output.append(f"TP Hits: {self.tp_hit_count} | SL Hits: {self.sl_hit_count}")
        output.append(f"Total Fees: ${self.total_fees_paid:.2f}")
        output.append(f"{'='*60}")
        
        text = "\n".join(output)
        
        if mode == 'human':
            print(text)
            return None
        else:
            return text

    def close(self) -> None:
        """Clean up resources."""
        pass


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def validate_reward_system():
    """
    Validate that the reward system is properly aligned.
    
    This test ensures:
    1. Losing money = negative reward
    2. Making money = positive reward  
    3. Shaping rewards don't dominate performance
    """
    print("\n" + "="*70)
    print("REWARD SYSTEM VALIDATION")
    print("="*70)
    
    try:
        env = TradingEnv(data_dir="./", verbose=False)
        
        # Test 1: Random actions should generally lose money and have negative reward
        obs, _ = env.reset()
        total_reward = 0
        steps = 500
        
        for _ in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        pv_change = env.portfolio_value - INITIAL_BALANCE
        
        print(f"\nTest 1: Random Agent ({steps} steps)")
        print(f"  Portfolio Change: ${pv_change:+,.2f}")
        print(f"  Cumulative Reward: {total_reward:+.2f}")
        print(f"  Alignment: {'✓ PASS' if (pv_change < 0) == (total_reward < 0) else '✗ FAIL'}")
        
        # Test 2: Cash-only strategy should have minimal reward
        obs, _ = env.reset()
        cash_only_action = np.zeros(9)
        cash_only_action[6] = 1.0  # 100% cash
        cash_only_action[7] = 0.5
        cash_only_action[8] = 0.5
        
        total_reward = 0
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(cash_only_action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"\nTest 2: Cash-Only Agent ({steps} steps)")
        print(f"  Portfolio Value: ${env.portfolio_value:,.2f}")
        print(f"  Cumulative Reward: {total_reward:+.2f}")
        print(f"  Alignment: {'✓ PASS' if abs(total_reward) < 50 else '✗ FAIL'}")
        
        # Test 3: Check reward component magnitudes
        if env.reward_history:
            last_components = env.reward_history[-1]
            print(f"\nTest 3: Reward Component Analysis")
            print(f"  Return Reward Magnitude: {abs(last_components['return_reward']):.2f}")
            print(f"  Shaping Rewards Magnitude: {abs(last_components['rr_quality'] + last_components['deployment_reward']):.2f}")
            
            return_dominates = abs(last_components['return_reward']) > abs(
                last_components['rr_quality'] + last_components['deployment_reward']
            )
            print(f"  Return Dominates: {'✓ PASS' if return_dominates else '⚠ CHECK'}")
        
        print("\n" + "="*70)
        print("VALIDATION COMPLETE")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"✗ Validation Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """Quick test to ensure the environment works correctly."""
    print("\n" + "="*70)
    print("ENVIRONMENT FUNCTIONALITY TEST")
    print("="*70)
    
    try:
        # Create environment
        env = TradingEnv(data_dir="./", verbose=True)
        print(f"✓ Environment created")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.shape}")
        print(f"  Data steps: {env.n_steps}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  Initial observation shape: {obs.shape}")
        print(f"  Initial portfolio value: ${info['portfolio_value']:,.2f}")
        
        # Test steps
        total_reward = 0
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"  Episode ended at step {i}")
                break
        
        print(f"✓ Ran 100 steps")
        print(f"  Final PV: ${info['portfolio_value']:,.2f}")
        print(f"  Cumulative Return: {info['cumulative_return']*100:+.2f}%")
        print(f"  Total Reward: {total_reward:+.2f}")
        print(f"  Trades: {info['n_trades']}")
        
        # Performance summary
        summary = env.get_performance_summary()
        print(f"✓ Performance Summary:")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
        
        # Test render
        env.render()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"✗ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TradingEnv v5.0 - Production Ready")
    print("="*70)
    
    # Run tests
    test_environment()
    print()
    validate_reward_system()
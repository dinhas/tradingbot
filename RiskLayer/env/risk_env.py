import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import os
import uuid
import yaml
from datetime import datetime
from pathlib import Path

try:
    from .feature_engine import RiskFeatureEngine
except (ImportError, ValueError):
    from RiskLayer.src.feature_engine import RiskFeatureEngine


class RiskTradingEnv(gym.Env):
    """
    Risk Management Environment (TradeGuard).
    Acts as a 'Sniper':
    1. Receives market state (High dimensional features).
    2. Decides to SKIP or TAKE trade (Long/Short).
    3. If TAKE, decides SL and TP placement.

    Action Space (Continuous, size 3):
    - [0] Direction/Risk: 
        * > 0: LONG (Magnitude = Risk %)
        * < 0: SHORT (Magnitude = Risk %)
        * Always executes (no blocking)
    - [1] SL Multiplier (normalized)
    - [2] TP Multiplier (normalized)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data_dir="data", is_training=True, data=None, max_rows=None):
        super(RiskTradingEnv, self).__init__()

        self.data_dir = data_dir
        self.is_training = is_training
        self.max_rows = max_rows
        self.assets = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD"]

        # Spreads (Assumes standard market conditions)
        # EURUSD: 1 pip, GBPUSD: 1.5 pips, USDJPY: 1 pip, USDCHF: 1.5 pips, XAUUSD: 20 cents
        self.spreads = {
            "EURUSD": 0.0001,
            "GBPUSD": 0.00015,
            "USDJPY": 0.01,
            "USDCHF": 0.00015,
            "XAUUSD": 0.20,
        }

        # Asset ID mapping for observation
        self.asset_ids = {asset: float(i) for i, asset in enumerate(self.assets)}

        # Load Configuration
        config_path = Path("RiskLayer/config/ppo_config.yaml")
        config = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)
                logging.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logging.warning(f"Failed to load config from {config_path}: {e}")
        else:
            logging.warning(f"Config not found at {config_path}, using defaults.")

        # Configuration
        self.EXECUTION_THRESHOLD = config.get("execution_threshold", 0.0)  # Unused - no blocking
        self.MIN_RR = config.get("min_rr", 1.5)
        self.ATR_SL_MIN = config.get("atr_sl_min", 0.75)  # Lowered for tighter stops
        self.ATR_SL_MAX = config.get("atr_sl_max", 7.0)
        self.ATR_TP_MIN = config.get("atr_tp_min", 2.0)  # Lowered to allow more TP opportunities
        self.ATR_TP_MAX = config.get("atr_tp_max", 15.0)
        self.MAX_RISK_PER_TRADE = config.get("max_risk_per_trade", 0.40)
        self.SLIPPAGE_MIN_PIPS = config.get("slippage_min_pips", 0.5)
        self.SLIPPAGE_MAX_PIPS = config.get("slippage_max_pips", 1.5)
        self.PERIOD_LOG_SIZE = config.get("period_log_size", 5000)

        # Logging Setup
        self.log_dir = Path("RiskLayer/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env_id = str(uuid.uuid4())[:8]
        self.reward_log_file = self.log_dir / f"rewards_env_{self.env_id}.csv"

        with open(self.reward_log_file, "w") as f:
            f.write(
                "timestamp,step,asset,avg_reward,win_rate,total_pnl,trades_taken,trades_skipped,skip_rate\n"
            )

        # Reward Tracking (Rolling 5000 steps)
        self.period_reward = 0.0
        self.period_trades = 0
        self.period_wins = 0
        self.period_pnl = 0.0
        self.period_skipped = 0

        # Load and Process Data
        self.feature_engine = RiskFeatureEngine()
        if data is not None:
            self.data = data
        else:
            self.data = self._load_data()

        logging.info("Preprocessing data with RiskFeatureEngine...")
        self.processed_data = self.feature_engine.preprocess_data(self.data)
        logging.info(f"Data processed. Shape: {self.processed_data.shape}")

        # Build cache for fast access
        self._cache_data_arrays()
        self._cache_signal_indices()
        self._cache_asset_columns()

        # Action Space: [Execution/Direction, SL_Mult, TP_Mult]
        # [0] Execution: >0.2 LONG, <-0.2 SHORT, -0.2 to 0.2 SKIP
        # [1] SL: Normalized [-1, 1] -> [min, max]
        # [2] TP: Normalized [-1, 1] -> [min, max]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation Space
        # We need to determine the size based on the feature engine output
        # For a single asset, we extract specific columns.
        # The feature engine generates ~80 features per asset.
        # We will feed the model:
        # 1. Target Asset Features (~80)
        # 2. Global/Context Features (~20)
        # Total approx 100-110.
        self._determine_obs_shape()

        # State
        self.current_step = 0
        self.current_asset = self.assets[0]
        self.max_steps = len(self.processed_data) - 1
        self.equity = 10000.0
        self.initial_equity = 10000.0  # Track initial for hypothesis E
        self.peak_equity = 10000.0

        # Tracking
        self.total_reward = 0
        self.trades_taken = 0
        self.trades_skipped = 0
        self.winning_trades = 0
        self.signals_processed = 0 # Robust counter for logging
        self.period_signals = 0    # Signals in current 5000 block
        
        # Performance capping to prevent "Infinite Money" bug in training stats
        self.MAX_EQUITY_MULT = 5.0 # Terminate if 5x profit
        self.MIN_EQUITY_MULT = 0.3 # Terminate if 70% loss

    def _load_data(self):
        """Load labeled market data with Alpha signals, ensuring OHLCV columns exist."""
        data = {}
        data_dir_path = Path("data")  # Use shared data directory
        for asset in self.assets:
            labeled_path = data_dir_path / f"{asset}_alpha_labeled.parquet"
            raw_path = data_dir_path / f"{asset}_5m.parquet"

            df = None
            # 1. Try loading labeled data
            if labeled_path.exists():
                try:
                    df = pd.read_parquet(labeled_path)
                    if "alpha_confidence" in df.columns:
                        df.rename(
                            columns={"alpha_confidence": "alpha_conf"}, inplace=True
                        )

                    # Check if OHLCV exists. If not, try to merge with raw data
                    required_cols = ["open", "high", "low", "close", "volume"]
                    if (
                        not all(col in df.columns for col in required_cols)
                        and raw_path.exists()
                    ):
                        logging.info(
                            f"Merging labeled {asset} with raw data to restore OHLCV..."
                        )
                        raw_df = pd.read_parquet(raw_path)
                        # Join on index (timestamp) and validate alignment
                        df = df[["alpha_signal", "alpha_conf"]].join(
                            raw_df, how="inner"
                        )
                        if len(df) == 0:
                            logging.warning(f"Merge failed for {asset}: No matching timestamps between labeled and raw data.")
                        elif len(df) < 0.9 * len(raw_df):
                            logging.warning(f"Data loss during merge for {asset}: {len(df)} rows remain from {len(raw_df)} raw rows.")

                    logging.info(f"Loaded {asset} from {labeled_path} ({len(df)} rows)")
                except Exception as e:
                    logging.warning(f"Failed to read {labeled_path}: {e}")

            # 2. Fallback to raw data only (no signals)
            if df is None and raw_path.exists():
                try:
                    df = pd.read_parquet(raw_path)
                    df["alpha_signal"] = 0.0
                    df["alpha_conf"] = 0.0
                    logging.info(f"Loaded raw {asset} from {raw_path} (No Signals)")
                except Exception as e:
                    logging.warning(f"Failed to read {raw_path}: {e}")

            # 3. Last resort: Dummy data
            if df is None:
                logging.warning(f"No data found for {asset}, using dummy.")
                dates = pd.date_range(start="2025-01-01", periods=1000, freq="5min")
                df = pd.DataFrame(index=dates)
                base = 1.0 if "JPY" not in asset else 150.0
                df["open"] = base
                df["high"] = base * 1.001
                df["low"] = base * 0.999
                df["close"] = base
                df["volume"] = 1000.0
                df["alpha_signal"] = 0.0
                df["alpha_conf"] = 0.0

            # Slice if requested
            if self.max_rows is not None and len(df) > self.max_rows:
                df = df.iloc[-self.max_rows :].copy()

            data[asset] = df
        return data

    def _cache_data_arrays(self):
        """Cache DataFrame columns as numpy arrays for performance."""
        self.feature_arrays = {}
        self.price_arrays = {}
        self.atr_arrays = {}
        self.signal_arrays = {}

        # Convert only numeric columns to float32 to prevent string conversion errors
        numeric_df = self.processed_data.select_dtypes(include=[np.number])
        self.master_matrix = numeric_df.astype(np.float32).values
        self.column_map = {col: i for i, col in enumerate(numeric_df.columns)}

        # Cache specific price arrays for simulation
        for asset in self.assets:
            self.price_arrays[asset] = {
                "high": self.data[asset]["high"].values.astype(np.float32),
                "low": self.data[asset]["low"].values.astype(np.float32),
                "close": self.data[asset]["close"].values.astype(np.float32),
                "open": self.data[asset]["open"].values.astype(np.float32),
            }
            # ATR is in processed data
            if f"{asset}_atr_14" in self.column_map:
                col_idx = self.column_map[f"{asset}_atr_14"]
                self.atr_arrays[asset] = self.master_matrix[:, col_idx]
            else:
                # Fallback if preprocessing failed to add ATR
                self.atr_arrays[asset] = np.zeros(
                    len(self.processed_data), dtype=np.float32
                )

            # Cache Alpha Signals
            if f"{asset}_alpha_signal" in self.column_map:
                col_idx = self.column_map[f"{asset}_alpha_signal"]
                self.signal_arrays[asset] = self.master_matrix[:, col_idx]
            else:
                # Should not happen if data loaded correctly, but fallback
                self.signal_arrays[asset] = np.zeros(
                    len(self.processed_data), dtype=np.float32
                )

    def _cache_signal_indices(self):
        """Pre-calculate indices of all signals for fast lookups and resets."""
        self.signal_indices = {}
        for asset in self.assets:
            signals = self.signal_arrays[asset]
            # Consider any non-zero signal as valid
            indices = np.where(np.abs(signals) > 0.01)[0]
            self.signal_indices[asset] = indices

    def _cache_asset_columns(self):
        """Pre-calculate column indices for each asset to avoid string matching in observation."""
        self.asset_col_indices = {}
        for asset in self.assets:
            # ONLY include columns that are in our numeric column_map
            asset_cols = [
                c
                for c in self.processed_data.columns
                if c.startswith(f"{asset}_") and c in self.column_map
            ]
            indices = [self.column_map[c] for c in asset_cols]
            self.asset_col_indices[asset] = indices

    def _determine_obs_shape(self):
        """Calculate observation size based on one asset's view."""
        # We define a standard observation vector:
        # [Asset Specific Features] + [Global Features]

        # Identify features for the first asset to count them
        asset = self.assets[0]
        # Only count numeric columns that actually go into the observation
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        asset_cols = [c for c in numeric_cols if c.startswith(f"{asset}_")]

        # +2 for Spread/ATR and Asset ID, +3 for Account State (Equity, Drawdown, RiskCap)
        self.obs_dim = len(asset_cols) + 2 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        logging.info(
            f"Observation Dimension: {self.obs_dim} (Features: {len(asset_cols)} + Spread/ATR + ID + 3 Account Features)"
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Asset Rotation - Works in both training and backtest
        if not hasattr(self, "_asset_idx"):
            self._asset_idx = np.random.randint(0, len(self.assets))
        else:
            self._asset_idx = (self._asset_idx + 1) % len(self.assets)

        found_signal = False
        # Try all assets starting from the rotated index to ensure we don't get stuck
        for i in range(len(self.assets)):
            rotated_idx = (self._asset_idx + i) % len(self.assets)
            self.current_asset = self.assets[rotated_idx]
            
            indices = self.signal_indices[self.current_asset]
            if len(indices) == 0:
                continue
            
            # Horizon is 500 steps, leave room for simulation
            valid_indices = indices[indices < self.max_steps - 550]
            
            if len(valid_indices) > 0:
                if self.is_training:
                    # Randomly pick a signal index
                    self.current_step = int(np.random.choice(valid_indices))
                    # Update master index to the successfully chosen asset
                    self._asset_idx = rotated_idx
                else:
                    # Backtest: usually start from first valid signal or options
                    if options and "step" in options:
                        self.current_step = options["step"]
                    else:
                        self.current_step = int(valid_indices[0])
                
                found_signal = True
                break

        if not found_signal:
            logging.error(f"CRITICAL: No signals found in any asset. Check data loading.")
            # Fallback to avoid complete crash
            self.current_step = 0
            self.current_asset = self.assets[0]

        self.trades_taken = 0
        self.trades_skipped = 0
        self.equity = self.initial_equity
        self.peak_equity = self.initial_equity
        
        # Reset period trackers
        self.period_reward = 0.0
        self.period_trades = 0
        self.period_wins = 0
        self.period_pnl = 0.0
        self.period_skipped = 0
        self.period_signals = 0

        return self._get_observation(), {}

    def _find_next_signal(self):
        """Advances current_step to the next index where alpha_signal != 0 using cached indices."""
        indices = self.signal_indices[self.current_asset]
        
        # Find the first index >= current_step + 1 (since we already processed current_step)
        next_step_target = self.current_step + 1
        idx_search_pos = np.searchsorted(indices, next_step_target)
        
        if idx_search_pos < len(indices):
            candidate_step = int(indices[idx_search_pos])
            # Ensure we haven't hit the end-of-data horizon buffer
            if candidate_step < self.max_steps - 500:
                self.current_step = candidate_step
                return True
        
        return False

    def _get_observation(self):
        """Extract features for current asset at current step."""
        indices = self.asset_col_indices[self.current_asset]
        obs = self.master_matrix[self.current_step, indices]

        # Handle NaNs
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)

        # Append Static Features (Spread/ATR Ratio, Asset ID)
        # Normalize spread by ATR to reduce cross-asset variance
        spread_raw = self.spreads[self.current_asset]
        atr = self.atr_arrays[self.current_asset][self.current_step]
        if atr > 0:
            spread_normalized = spread_raw / atr
        else:
            # Fallback if ATR is invalid
            spread_normalized = 0.0
        asset_id = self.asset_ids[self.current_asset]

        # Account State Features
        peak_safe = max(self.peak_equity, 1e-9)
        drawdown = np.clip(1.0 - (self.equity / peak_safe), 0.0, 1.0)
        equity_norm = np.clip(self.equity / self.initial_equity, 0.0, self.MAX_EQUITY_MULT + 1.0)
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))

        obs = np.concatenate([obs, [spread_normalized, asset_id, equity_norm, drawdown, risk_cap_mult]])

        return obs

    def _update_logging(self, reward):
        """Robust logging that works for both blocked and taken trades."""
        self.period_reward += reward
        self.total_reward += reward
        self.period_signals += 1
        self.signals_processed += 1

        # Log every X signals - Configurable (Issue 16)
        if self.period_signals >= self.PERIOD_LOG_SIZE:
            avg_reward = self.period_reward / self.period_signals
            win_rate = (self.period_wins / self.period_trades) if self.period_trades > 0 else 0.0
            skip_rate = self.period_skipped / self.period_signals
            
            log_line = f"{datetime.now()},{self.signals_processed},{self.current_asset},{avg_reward:.4f},{win_rate:.4f},{self.period_pnl:.4f},{self.period_trades},{self.period_skipped},{skip_rate:.2f}\n"
            
            try:
                with open(self.reward_log_file, "a") as f:
                    f.write(log_line)
            except Exception as e:
                logging.error(f"Failed to write to log: {e}")
            
            # Reset period trackers
            self.period_reward = 0.0
            self.period_trades = 0
            self.period_wins = 0
            self.period_pnl = 0.0
            self.period_skipped = 0
            self.period_signals = 0

    def step(self, action):
        # 1. Parse Action
        conf_raw = action[0]   # Execution Confidence / Direction
        sl_norm = action[1]    # SL Multiplier
        tp_norm = action[2]    # TP Multiplier

        # Pre-execution Account State (Bug 1: Move peak_safe up)
        peak_at_start = max(self.peak_equity, 1e-9)
        prev_equity = self.equity
        prev_dd = 1.0 - (prev_equity / peak_at_start)

        # 2. Determine Direction and Risk from Action
        # REMOVED BLOCKING: Always execute trades, focus on profit maximization
        # > 0 = LONG, < 0 = SHORT, magnitude = risk percentage
        if conf_raw > 0:
            direction = 1
            risk_raw = conf_raw  # Use raw confidence as risk (0 to 1)
        else:
            direction = -1
            risk_raw = abs(conf_raw)  # Use absolute value as risk (0 to 1)

        # Map SL/TP to multipliers
        sl_mult = self._map_range(sl_norm, -1, 1, self.ATR_SL_MIN, self.ATR_SL_MAX)
        tp_mult = self._map_range(tp_norm, -1, 1, self.ATR_TP_MIN, self.ATR_TP_MAX)

        # 3. Trade Execution (ALWAYS EXECUTE - NO BLOCKING)
        # Apply Random Adverse Slippage on Entry
        asset = self.current_asset
        bid_current = self.price_arrays[asset]["close"][self.current_step]
        spread = self.spreads.get(asset, 0.0)
        
        # FIX: Removed slippage entirely per user request to maximize SNR
        if direction == 1:
            entry_price = bid_current + spread  # Entry at ASK
        else:
            entry_price = bid_current           # Entry at BID

        # Actual Risk % (Comment matches implementation now)
        actual_risk_pct = risk_raw * self.MAX_RISK_PER_TRADE
        risk_amount = self.equity * actual_risk_pct

        # Calculate ATR and apply spread protection BEFORE position sizing
        # This ensures position sizing uses the SAME sl_dist as simulation
        atr = self.atr_arrays[asset][self.current_step]
        if atr <= 0:
            atr = max(bid_current * 0.0001, 1e-5)
        
        # Calculate SL distance directly from model's action - NO OVERRIDE
        # Let the model learn the true relationship between its actions and outcomes
        sl_dist = sl_mult * atr
        
        # Safety check: ensure ATR is valid
        if sl_dist <= 0:
            sl_dist = 0.0001 if asset != "USDJPY" else 0.01

        # Simulate Outcome (Passing the slippage-adjusted entry)
        r_multiple, bars, outcome_type, pnl_price = self._simulate_trade(
            direction, sl_mult, tp_mult, entry_price_override=entry_price
        )

        # Calculate position size
        contract_size = 100 if asset == "XAUUSD" else 100000
        is_usd_quote = asset in ["EURUSD", "GBPUSD", "XAUUSD"]
        is_usd_base = asset in ["USDJPY", "USDCHF"]

        if sl_dist > 0:
            if is_usd_quote:
                lots = risk_amount / (sl_dist * contract_size)
            elif is_usd_base:
                lots = (risk_amount * entry_price) / (sl_dist * contract_size)
            else:
                lots = risk_amount / (sl_dist * contract_size)
        else:
            lots = 0.01
            
        lots = np.clip(lots, 0.01, 100.0)

        # Convert price PnL to dollar PnL
        if is_usd_quote:
            pnl = pnl_price * lots * contract_size
        elif is_usd_base:
            # For USD-base, exit price is used for conversion
            exit_price = entry_price + (pnl_price * direction)
            pnl = (pnl_price * lots * contract_size) / exit_price
        else:
            pnl = pnl_price * lots * contract_size

        # Reward Calculation - FOCUS ON PROFIT MAXIMIZATION & LOSS MINIMIZATION
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # PnL Reward (Primary Focus - Maximize Profits)
        pnl_ratio = pnl / max(prev_equity, 1e-6)
        # Amplify profits, penalize losses more
        if pnl > 0:
            pnl_reward = pnl_ratio * 150.0  # Increased reward for profits
        else:
            pnl_reward = pnl_ratio * 200.0  # Stronger penalty for losses
        
        # Drawdown Penalty (Minimize Losses)
        new_dd = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        dd_increase = max(0.0, new_dd - prev_dd)
        dd_penalty = -(dd_increase ** 2) * 30.0  # Reduced from 50.0 to allow more risk-taking
        
        # Profit Maximization Bonuses
        tp_bonus = 0.0
        rr_bonus = 0.0
        
        if outcome_type == 'TP':
            # Reward TP hits more aggressively
            tp_bonus = 1.0 * (tp_mult / 8.0)  # Increased from 0.5
            # Reward high R-multiple trades
            if r_multiple >= 2.0:
                rr_bonus = 2.0  # Bonus for 2R+ trades
            elif r_multiple >= 1.5:
                rr_bonus = 1.0  # Bonus for 1.5R+ trades
        
        # Loss Minimization Penalties
        loss_penalty = 0.0
        if outcome_type == 'SL':
            # Penalize losses more, especially large ones
            if r_multiple <= -2.0:
                loss_penalty = -3.0  # Heavy penalty for large losses
            elif r_multiple <= -1.0:
                loss_penalty = -1.5  # Medium penalty
            
        # Equity Growth Bonus (Encourage account growth)
        equity_growth = (self.equity - self.initial_equity) / self.initial_equity
        growth_bonus = min(equity_growth * 0.1, 5.0)  # Small bonus for overall growth
        
        reward = pnl_reward + tp_bonus + rr_bonus + dd_penalty + loss_penalty + growth_bonus

        # Tracking Update
        self.trades_taken += 1
        self.period_trades += 1
        if r_multiple > 0:
            self.period_wins += 1
            self.winning_trades += 1
        self.period_pnl += r_multiple
        
        self._update_logging(reward)

        info = {
            'action': 'OPEN',
            'direction': "LONG" if direction == 1 else "SHORT",
            'sl': sl_mult,
            'tp': tp_mult,
            'pnl': pnl,
            'outcome': outcome_type,
            'lots': lots,
            'equity': self.equity,
            'is_blocked': False,
            'r_multiple': r_multiple
        }

        # Advance State
        self.current_step += 1
        has_next = self._find_next_signal()
        terminated = False
        
        # Termination conditions
        if self.equity < (self.initial_equity * self.MIN_EQUITY_MULT):
            terminated = True
            reward -= 50.0
        elif self.equity > (self.initial_equity * self.MAX_EQUITY_MULT):
            terminated = True
            reward += 50.0 # Reward for "winning" the simulation
            
        # Increased clipping range to preserve gradient signal from multi-component rewards
        reward = np.clip(reward, -200.0, 200.0)
        truncated = not has_next
        
        return self._get_observation(), reward, terminated, truncated, info

    def _map_range(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _evaluate_block_reward(self, max_favorable, max_adverse):
        """
        Helper to calculate reward for blocked/skipped trades using Oracle.
        Implements Tiered Blocking Rewards (Loss Avoidance) vs Scaled Win-Miss Penalties.
        Now uses spread-aware percentages.
        """
        asset = self.current_asset
        spread_pct = self.spreads.get(asset, 0.0) / self.price_arrays[asset]["close"][self.current_step]
        
        # 1. Tiered Blocking Rewards (Loss Avoidance)
        # We reward blocking trades that would have exceeded our spread + some slippage
        if max_adverse < -(spread_pct * 2.5): 
            return 150.0, "BLOCKED_CATASTROPHIC"
        
        if max_adverse < -(spread_pct * 1.5):
             return 40.0, "BLOCKED_MAJOR_LOSS"
             
        # 2. Scaled Win-Miss Penalties (Opportunity Cost)
        # Missed winners need to be significantly above spread to be a "real" miss
        if max_favorable > (spread_pct * 10.0): # E.g. 10 pips if spread is 1
             return -250.0, "MISSED_HUGE_WIN"
             
        if max_favorable > (spread_pct * 5.0):
             return -80.0, "MISSED_BIG_WIN"
             
        if max_favorable > (spread_pct * 2.0):
             return -20.0, "MISSED_SMALL_WIN"
             
        # 3. Minor Loss / Noise / Spread-Eaten trades
        return 2.0, "BLOCKED_NOISE"

    def _get_oracle_values(self, direction):
        """
        Look ahead to find max potential profit/loss for oracle rewards.
        Correctly accounts for Bid/Ask spread on entry and exit.
        """
        asset = self.current_asset
        idx = self.current_step
        spread = self.spreads.get(asset, 0.0)
        
        bid_current = self.price_arrays[asset]["close"][idx]
        ask_current = bid_current + spread
        
        # Entry logic (spread only, no slippage)
        if direction == 1: # LONG
            entry_price = ask_current
        else: # SHORT
            entry_price = bid_current
        
        horizon = 500
        start = idx + 1
        
        # Explicit boundary check (Bug 5)
        total_len = len(self.price_arrays[asset]["close"])
        if start >= total_len:
             return 0.0, 0.0
             
        end = min(start + horizon, total_len)
        
        if start >= end:
            return 0.0, 0.0
            
        lows_bid = self.price_arrays[asset]["low"][start:end]
        highs_bid = self.price_arrays[asset]["high"][start:end]
        
        if direction == 1: # LONG
            # Exit Long at Bid
            max_fav = (highs_bid.max() - entry_price) / entry_price
            max_adv = (lows_bid.min() - entry_price) / entry_price
        else: # SHORT
            # Exit Short at Ask (Bid + Spread)
            max_fav = (entry_price - (lows_bid.min() + spread)) / entry_price
            max_adv = (entry_price - (highs_bid.max() + spread)) / entry_price
            
        return max(0.0, max_fav), min(0.0, max_adv)

    def _simulate_trade(self, direction, sl_mult, tp_mult, entry_price_override=None):
        """
        Simulate trade outcome using realistic Bid/Ask spreads.
        Ensures PnL and R-multiple are spread-aware.
        """
        asset = self.current_asset
        idx = self.current_step
        spread = self.spreads.get(asset, 0.0)

        bid_current = self.price_arrays[asset]["close"][idx]
        ask_current = bid_current + spread

        if entry_price_override is not None:
            entry_price = entry_price_override
        else:
            entry_price = ask_current if direction == 1 else bid_current

        atr = self.atr_arrays[asset][idx]
        if atr <= 0:
            # Bug 4 Safety: Minimum ATR as at least a healthy floor
            atr = max(bid_current * 0.0001, 1e-5)

        # Calculate SL distance directly from multiplier - NO OVERRIDE
        # Let the model learn optimal SL placement without artificial constraints
        sl_dist = sl_mult * atr
        
        # Safety check: ensure valid distance
        if sl_dist <= 0:
            sl_dist = 0.0001 if asset != "USDJPY" else 0.01
        
        tp_dist = tp_mult * atr

        # Determine Triggers (Calculated from Entry)
        if direction == 1:  # Long
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:  # Short
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # Look ahead
        horizon = 500
        start = idx + 1
        
        # Explicit boundary check (Bug 5)
        total_len = len(self.price_arrays[asset]["close"])
        if start >= total_len:
            return 0.0, 0.0, "TIMEOUT", 0.0
            
        end = min(start + horizon, total_len)

        # Future Price Arrays (Bid)
        lows_bid = self.price_arrays[asset]["low"][start:end]
        highs_bid = self.price_arrays[asset]["high"][start:end]

        # Check hits
        if direction == 1:  # Long Checks (Exit at Bid)
            sl_hits = lows_bid <= sl_price
            tp_hits = highs_bid >= tp_price
        else:  # Short Checks (Exit at Ask)
            # Using bid-only arrays, check if ASK (Bid + Spread) reaches SL/TP level
            # Algebraically equivalent to: highs_bid >= sl_price - spread
            sl_hits = (highs_bid + spread) >= sl_price
            tp_hits = (lows_bid + spread) <= tp_price

        # Find first occurrence
        first_sl = np.argmax(sl_hits) if sl_hits.any() else horizon + 1
        first_tp = np.argmax(tp_hits) if tp_hits.any() else horizon + 1

        # Result Logic
        outcome = ""
        actual_exit_price = 0.0

        if first_sl == horizon + 1 and first_tp == horizon + 1:
            outcome = "TIMEOUT"
            bars = horizon
            close_bid = self.price_arrays[asset]["close"][end - 1]
            # Exit at market rates (already correct: LONG at BID, SHORT at ASK)
            actual_exit_price = close_bid if direction == 1 else (close_bid + spread)

        elif first_sl < first_tp:
            outcome = "SL"
            bars = first_sl + 1
            # FIX: Do NOT add spread here - it's already accounted for in the trigger check
            # Lines 752-753 check (highs_bid + spread) >= sl_price, so sl_price IS the ask-equivalent
            # Adding spread again causes SHORT trades to pay spread TWICE (was -1 pip systematic bias)
            actual_exit_price = sl_price
        else:
            outcome = "TP"
            bars = first_tp + 1
            # FIX: Same as above - spread already in trigger check, don't double-count
            actual_exit_price = tp_price

        # SPREAD-AWARE PnL
        if direction == 1:
            pnl_price = actual_exit_price - entry_price
        else:
            pnl_price = entry_price - actual_exit_price

        # SPREAD-AWARE R-MULTIPLE
        # Risk is the distance from entry to stop loss. 
        # Since entry_price and sl_price both account for spread/slippage, 
        # sl_dist IS the correct risk denominator.
        r_multiple = pnl_price / sl_dist if sl_dist > 0 else 0

        return r_multiple, bars, outcome, pnl_price

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
    - [0] Execution Confidence:
        * < -threshold: SHORT
        * > +threshold: LONG
        * Between: SKIP
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
        self.EXECUTION_THRESHOLD = config.get("execution_threshold", 0.2)
        self.MIN_RR = config.get("min_rr", 1.5)
        self.ATR_SL_MIN = config.get("atr_sl_min", 3.0)
        self.ATR_SL_MAX = config.get("atr_sl_max", 7.0)
        self.ATR_TP_MIN = config.get("atr_tp_min", 1.0)
        self.ATR_TP_MAX = config.get("atr_tp_max", 15.0)

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

        # Action Space: [Execution_Trig, SL_Mult, TP_Mult]
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

        # Tracking
        self.total_reward = 0
        self.trades_taken = 0
        self.trades_skipped = 0
        self.winning_trades = 0

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
                        # Join on index (timestamp)
                        df = df[["alpha_signal", "alpha_conf"]].join(
                            raw_df, how="inner"
                        )

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

    def _determine_obs_shape(self):
        """Calculate observation size based on one asset's view."""
        # We define a standard observation vector:
        # [Asset Specific Features] + [Global Features]

        # Identify features for the first asset to count them
        asset = self.assets[0]
        # Only count numeric columns that actually go into the observation
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns
        asset_cols = [c for c in numeric_cols if c.startswith(f"{asset}_")]

        # +2 for Spread and Asset ID
        self.obs_dim = len(asset_cols) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        logging.info(
            f"Observation Dimension: {self.obs_dim} (Features: {len(asset_cols)} + Spread + ID)"
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.is_training:
            # Try assets until we find one with signals
            found_signal = False
            for _ in range(len(self.assets)):
                # Round-Robin asset selection for guaranteed coverage across all pairs
                if not hasattr(self, "_asset_idx"):
                    # Initialize with a random start so parallel envs don't all start on the same asset
                    self._asset_idx = np.random.randint(0, len(self.assets))
                else:
                    self._asset_idx = (self._asset_idx + 1) % len(self.assets)

                self.current_asset = self.assets[self._asset_idx]

                # Random start, but ensure we land on a signal
                # Use dynamic bounds to handle small datasets (e.g., during dry runs or initial tests)
                low_bound = min(100, max(0, self.max_steps // 10))
                high_bound = max(
                    low_bound + 1, self.max_steps - 600
                )  # Leave ~500 for simulation
                self.current_step = np.random.randint(low_bound, high_bound)
                if self._find_next_signal():
                    found_signal = True
                    break

            if not found_signal:
                logging.error(
                    "CRITICAL: No signals found in ANY asset. Training will fail."
                )
        else:
            # Backtest mode: usually linear, handled by caller or options
            self.current_step = min(100, max(0, self.max_steps // 10))
            if options and "asset" in options:
                self.current_asset = options["asset"]
            if not self._find_next_signal():
                logging.warning(
                    f"No signals found for {self.current_asset} in backtest."
                )

        self.trades_taken = 0
        self.trades_skipped = 0

        return self._get_observation(), {}

    def _find_next_signal(self):
        """Advances current_step to the next index where alpha_signal != 0."""
        # Use a loop instead of recursion to avoid RecursionError
        # Max 5 attempts to find a signal via random resets
        max_attempts = 5 if self.is_training else 1

        for attempt in range(max_attempts):
            signals = self.signal_arrays[self.current_asset]
            # Search limit: leave enough room for a minimal simulation, but don't go negative
            search_limit = max(1, self.max_steps - 50)

            while self.current_step < search_limit:
                if abs(signals[self.current_step]) > 0.01:  # Check for non-zero signal
                    return True
                self.current_step += 1

            # If we run out of data, reset to a random point (in training)
            if self.is_training:
                low_bound = min(100, max(0, self.max_steps // 10))
                high_bound = max(low_bound + 1, self.max_steps - 600)
                self.current_step = np.random.randint(low_bound, high_bound)
                # Next iteration will search from the new random point
            else:
                return False

        logging.warning(
            f"No signals found for {self.current_asset} after {max_attempts} random resets. Switch asset?"
        )
        return False

    def _get_observation(self):
        """Extract features for current asset at current step."""
        # Find columns for current asset
        # This is a bit slow if done every step by string matching.
        # Optimization: Pre-calculate column indices for each asset.
        if not hasattr(self, "asset_col_indices"):
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

        indices = self.asset_col_indices[self.current_asset]
        obs = self.master_matrix[self.current_step, indices]

        # Handle NaNs
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs)

        # Append Static Features (Spread, Asset ID)
        spread = self.spreads[self.current_asset]
        asset_id = self.asset_ids[self.current_asset]

        obs = np.concatenate([obs, [spread, asset_id]])

        return obs

    def step(self, action):
        # 1. Parse Action
        execution_trig = action[0]
        sl_norm = action[1]  # [-1, 1]
        tp_norm = action[2]  # [-1, 1]

        # Map SL/TP to multipliers
        sl_mult = self._map_range(sl_norm, -1, 1, self.ATR_SL_MIN, self.ATR_SL_MAX)
        tp_mult = self._map_range(tp_norm, -1, 1, self.ATR_TP_MIN, self.ATR_TP_MAX)

        # 2. Determine Direction from Alpha Signal
        # The direction is FIXED by the environment's current state signal.
        current_signal = self.signal_arrays[self.current_asset][self.current_step]
        direction = 1 if current_signal > 0 else -1

        # 3. Determine Execution
        # Always execute - no skip decision
        decision = "OPEN"

        # 4. Calculate Reward
        # =====================
        # EQUITY PERCENTAGE REWARD STRUCTURE
        # =====================
        reward = 0.0
        info = {
            "action": decision,
            "direction": "LONG" if direction == 1 else "SHORT",
            "sl": sl_mult,
            "tp": tp_mult,
            "pnl": 0.0,
            "outcome": "OPEN",
        }

        # Simulate Outcome
        r_multiple, bars, outcome_type, pnl_price = self._simulate_trade(
            direction, sl_mult, tp_mult
        )

        # Calculate dollar PnL based on position sizing
        # Assume 1% risk per trade as standard
        asset = self.current_asset
        atr = self.atr_arrays[asset][self.current_step]
        idx = self.current_step
        bid_current = self.price_arrays[asset]["close"][idx]
        spread = self.spreads.get(asset, 0.0)

        sl_dist = sl_mult * atr
        risk_amount = self.equity * 0.01  # Risk 1% per trade

        # Calculate position size based on asset type
        contract_size = 100 if asset == "XAUUSD" else 100000
        is_usd_quote = asset in ["EURUSD", "GBPUSD", "XAUUSD"]
        is_usd_base = asset in ["USDJPY", "USDCHF"]

        # Calculate lots needed to risk 1%
        if sl_dist > 0:
            if is_usd_quote:
                lots = risk_amount / (sl_dist * contract_size)
            elif is_usd_base:
                lots = (risk_amount * bid_current) / (sl_dist * contract_size)
            else:
                lots = risk_amount / (sl_dist * contract_size)
        else:
            lots = 0.01

        # Convert price PnL to dollar PnL
        if is_usd_quote:
            pnl = pnl_price * lots * contract_size
        elif is_usd_base:
            pnl = (pnl_price * lots * contract_size) / bid_current
        else:
            pnl = pnl_price * lots * contract_size

        # =====================
        # OPEN TRADE REWARDS
        # =====================
        self.trades_taken += 1
        info["pnl"] = pnl
        info["outcome"] = outcome_type

        # Main reward: equity change percentage based on current equity
        equity_change_pct = (pnl / self.equity) * 100
        reward = equity_change_pct

        # Update current equity
        self.equity += pnl

        # Risk:Reward Structure Bonus/Penalty
        rr_ratio = tp_mult / sl_mult
        if rr_ratio >= 2.0:
            reward += 0.3  # Bonus for good RR structure
        elif rr_ratio < self.MIN_RR:
            reward -= 0.5  # Penalty for bad RR

            # #region agent log
            # HYPOTHESIS C: Reward amplification of spread losses
            if self.current_step % 5000 == 0 and not self.is_training:
                try:
                    import json
                    import os

                    log_path = r"e:\tradingbot\.cursor\debug.log"
                    with open(log_path, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "id": f"log_{self.current_step}_{self.current_asset}_reward",
                                    "timestamp": int(__import__("time").time() * 1000),
                                    "location": "risk_env.py:step:reward",
                                    "message": "Reward calculation",
                                    "data": {
                                        "asset": self.current_asset,
                                        "decision": decision,
                                        "r_multiple": float(r_multiple),
                                        "reward": float(reward),
                                        "outcome": outcome_type,
                                        "rr_ratio": float(rr_ratio),
                                        "hypothesisId": "C",
                                    },
                                    "sessionId": "debug-session",
                                    "runId": "run1",
                                }
                            )
                            + "\n"
                        )
                except:
                    pass
            # #endregion agent log

        # =====================
        # TRACKING UPDATE
        # =====================
        self.period_reward += reward
        self.total_reward += reward

        self.period_trades += 1
        if r_multiple > 0:
            self.period_wins += 1
            self.winning_trades += 1
        self.period_pnl += r_multiple

        # #region agent log
        # HYPOTHESIS E: No equity tracking with spreads
        if self.current_step % 5000 == 0 and not self.is_training:
            try:
                import json

                log_path = r"e:\tradingbot\.cursor\debug.log"
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": f"log_{self.current_step}_{self.current_asset}_equity",
                                "timestamp": int(__import__("time").time() * 1000),
                                "location": "risk_env.py:step:equity",
                                "message": "Equity tracking check",
                                "data": {
                                    "asset": self.current_asset,
                                    "equity": float(self.equity),
                                    "initial_equity": float(self.initial_equity),
                                    "equity_change_pct": float(
                                        (self.equity - self.initial_equity)
                                        / self.initial_equity
                                        * 100
                                    ),
                                    "r_multiple": float(r_multiple),
                                    "total_reward": float(self.total_reward),
                                    "trades_taken": int(self.trades_taken),
                                    "hypothesisId": "E",
                                },
                                "sessionId": "debug-session",
                                "runId": "run1",
                            }
                        )
                        + "\n"
                    )
            except:
                pass
        # #endregion agent log

        # --- Logging (Every 5000 steps) ---
        if self.current_step % 5000 == 0 and not self.is_training:
            avg_reward = self.period_reward / 5000
            win_rate = (
                (self.period_wins / self.period_trades)
                if self.period_trades > 0
                else 0.0
            )
            skip_rate = self.period_skipped / (
                self.period_trades + self.period_skipped + 1
            )

            log_line = f"{datetime.now()},{self.current_step},{self.current_asset},{avg_reward:.4f},{win_rate:.4f},{self.period_pnl:.4f},{self.period_trades},{self.period_skipped},{skip_rate:.2f}\n"

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

        # 5. Advance State to NEXT SIGNAL
        self.current_step += 1
        has_next = self._find_next_signal()

        terminated = False
        truncated = False

        # Use a dynamic threshold for truncation to avoid immediate termination on small datasets
        truncation_threshold = max(10, self.max_steps - 100)
        if not has_next or self.current_step >= truncation_threshold:
            truncated = True

        # Get next observation
        next_obs = self._get_observation()

        return next_obs, reward, terminated, truncated, info

    def _map_range(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def _simulate_trade(self, direction, sl_mult, tp_mult):
        """
        Simulate trade outcome using realistic Bid/Ask spreads.
        Assumes cached prices (High, Low, Close) are BID prices.

        Logic:
        - LONG: Enter at ASK (Bid + Spread). Exit at BID.
          SL Trigger: Low (Bid) <= SL.
          TP Trigger: High (Bid) >= TP.
        - SHORT: Enter at BID. Exit at ASK (Bid + Spread).
          SL Trigger: High (Ask) >= SL -> (High_Bid + Spread) >= SL.
          TP Trigger: Low (Ask) <= TP -> (Low_Bid + Spread) <= TP.
        """
        # #region agent log
        import json
        import os

        log_path = r"e:\tradingbot\.cursor\debug.log"
        # #endregion agent log

        asset = self.current_asset
        idx = self.current_step
        spread = self.spreads.get(asset, 0.0)

        # Price at current step (Signal Candle)
        # We assume trade is entered at the CLOSE of the signal candle (or Open of next).
        # Using CLOSE of current step for simplicity and consistency with previous logic.
        bid_current = self.price_arrays[asset]["close"][idx]
        ask_current = bid_current + spread

        atr = self.atr_arrays[asset][idx]
        if atr <= 0:
            atr = bid_current * 0.0001

        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        # Determine Entry and Triggers
        if direction == 1:  # Long
            entry_price = ask_current  # Buy at Ask
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else:  # Short
            entry_price = bid_current  # Sell at Bid
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist

        # #region agent log
        try:
            if idx % 5000 == 0 and not self.is_training:
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": f"log_{idx}_{asset}",
                                "timestamp": int(__import__("time").time() * 1000),
                                "location": "risk_env.py:_simulate_trade:entry",
                                "message": "Trade simulation start",
                                "data": {
                                    "asset": asset,
                                    "direction": "LONG" if direction == 1 else "SHORT",
                                    "bid_current": float(bid_current),
                                    "ask_current": float(ask_current),
                                    "spread": float(spread),
                                    "atr": float(atr),
                                    "sl_mult": float(sl_mult),
                                    "tp_mult": float(tp_mult),
                                    "sl_dist": float(sl_dist),
                                    "tp_dist": float(tp_dist),
                                    "entry_price": float(entry_price),
                                    "sl_price": float(sl_price),
                                    "tp_price": float(tp_price),
                                    "hypothesisId": "A",
                                },
                                "sessionId": "debug-session",
                                "runId": "run1",
                            }
                        )
                        + "\n"
                    )
        except:
            pass
        # #endregion agent log

        # Look ahead
        horizon = 500
        start = idx + 1
        end = min(start + horizon, len(self.price_arrays[asset]["close"]))

        # Future Price Arrays (Bid)
        lows_bid = self.price_arrays[asset]["low"][start:end]
        highs_bid = self.price_arrays[asset]["high"][start:end]

        # Check hits
        if direction == 1:  # Long Checks (against Bid)
            # SL hit if Bid goes below SL
            sl_hits = lows_bid <= sl_price
            # TP hit if Bid goes above TP
            tp_hits = highs_bid >= tp_price
        else:  # Short Checks (against Ask)
            # Short SL hit if Ask (High_Bid + Spread) goes above SL
            sl_hits = (highs_bid + spread) >= sl_price
            # Short TP hit if Ask (Low_Bid + Spread) goes below TP
            tp_hits = (lows_bid + spread) <= tp_price

        # Find first occurrence
        first_sl = np.argmax(sl_hits) if sl_hits.any() else horizon + 1
        first_tp = np.argmax(tp_hits) if tp_hits.any() else horizon + 1

        # Result
        outcome = ""
        exit_price = 0.0
        actual_exit_price = 0.0  # Initialize for all cases

        if first_sl == horizon + 1 and first_tp == horizon + 1:
            # Timed out
            outcome = "TIMEOUT"
            bars = horizon
            # Close position at current market rates
            close_bid = self.price_arrays[asset]["close"][end - 1]
            if direction == 1:
                exit_price = close_bid  # Sell Long at Bid
                actual_exit_price = close_bid  # Same as exit_price for TIMEOUT
            else:
                exit_price = close_bid + spread  # Cover Short at Ask
                actual_exit_price = close_bid + spread  # Same as exit_price for TIMEOUT

        elif first_sl < first_tp:
            # SL Hit
            outcome = "SL"
            bars = first_sl + 1
            # #region agent log
            # HYPOTHESIS B: Exit price mismatch - for LONG, should exit at BID, not sl_price
            actual_exit_bid = (
                lows_bid[first_sl] if first_sl < len(lows_bid) else sl_price
            )
            if direction == 1:  # LONG: exit at BID when SL hit
                actual_exit_price = actual_exit_bid
            else:  # SHORT: exit at ASK when SL hit
                actual_exit_price = actual_exit_bid + spread
            # #endregion agent log
            exit_price = sl_price  # Assumed filled at SL
        else:
            # TP Hit
            outcome = "TP"
            bars = first_tp + 1
            # #region agent log
            # HYPOTHESIS D: TP calculation ignores spread
            actual_exit_bid = (
                highs_bid[first_tp] if first_tp < len(highs_bid) else tp_price
            )
            if direction == 1:  # LONG: exit at BID when TP hit
                actual_exit_price = actual_exit_bid
            else:  # SHORT: exit at ASK when TP hit
                actual_exit_price = actual_exit_bid + spread
            # #endregion agent log
            exit_price = tp_price  # Assumed filled at TP

        # Calculate R-multiple
        # Real PnL
        if direction == 1:
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        # #region agent log
        # HYPOTHESIS A: R-multiple doesn't account for spread in risk
        # HYPOTHESIS B: Exit price mismatch causes larger losses
        actual_pnl = (
            actual_exit_price - entry_price
            if direction == 1
            else entry_price - actual_exit_price
        )
        actual_risk_with_spread = (
            sl_dist + spread if direction == 1 else sl_dist + spread
        )
        r_multiple_with_spread_risk = (
            actual_pnl / actual_risk_with_spread if actual_risk_with_spread > 0 else 0
        )
        # #endregion agent log

        # Risk was sl_dist (Price Distance)
        # R = PnL / Risk
        r_multiple = pnl / sl_dist if sl_dist > 0 else 0

        # #region agent log
        try:
            if idx % 5000 == 0 and not self.is_training:
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "id": f"log_{idx}_{asset}_exit",
                                "timestamp": int(__import__("time").time() * 1000),
                                "location": "risk_env.py:_simulate_trade:exit",
                                "message": "Trade simulation result",
                                "data": {
                                    "asset": asset,
                                    "direction": "LONG" if direction == 1 else "SHORT",
                                    "outcome": outcome,
                                    "exit_price_calculated": float(exit_price),
                                    "exit_price_actual": float(actual_exit_price),
                                    "entry_price": float(entry_price),
                                    "spread": float(spread),
                                    "pnl_calculated": float(pnl),
                                    "pnl_actual": float(actual_pnl),
                                    "sl_dist": float(sl_dist),
                                    "actual_risk_with_spread": float(
                                        actual_risk_with_spread
                                    ),
                                    "r_multiple_calculated": float(r_multiple),
                                    "r_multiple_with_spread_risk": float(
                                        r_multiple_with_spread_risk
                                    ),
                                    "r_multiple_diff": float(
                                        r_multiple - r_multiple_with_spread_risk
                                    ),
                                    "hypothesisId": "A,B,D",
                                },
                                "sessionId": "debug-session",
                                "runId": "run1",
                            }
                        )
                        + "\n"
                    )
        except:
            pass
        # #endregion agent log

        return r_multiple, bars, outcome, pnl

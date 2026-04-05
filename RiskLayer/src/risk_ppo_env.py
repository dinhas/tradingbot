import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import os
from collections import deque
try:
    from numba import njit
except ImportError:  # pragma: no cover - allows training smoke tests without numba
    def njit(fn):
        return fn
from frozen_feature_engine import FeatureEngine
from stable_baselines3.common.vec_env import VecEnv

# ──────────────────────────────────────────────
# SPREAD TABLE (in pips, 1 pip = 0.0001 for FX)
# ──────────────────────────────────────────────
SPREAD_PIPS = {
    "EURUSD": 1.2,
    "GBPUSD": 1.5,
    "USDJPY": 1.0,
    "USDCHF": 1.8,
    "XAUUSD": 45.0,   # Updated conservative Gold spread
}

PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "USDCHF": 0.0001,
    "XAUUSD": 0.1,    # 1 pip = $0.10 on Gold
}

EXECUTION_NOISE_ATR = 0.15   # ±0.15x ATR noise on trigger prices

def add_execution_noise(sl_mult, tp_mult):
    """
    Apply conservative execution uncertainty to SL/TP multipliers.
    Returns effective multipliers used for both simulation and reward.
    """
    eff_sl = max(0.3, sl_mult - np.random.uniform(0, EXECUTION_NOISE_ATR))
    eff_tp = tp_mult + np.random.uniform(0, EXECUTION_NOISE_ATR * 0.5)
    return eff_sl, eff_tp

def compute_spread_cost_in_r(asset, atr_price, eff_sl_mult):
    """
    Approximate round-trip spread friction in R units.
    1R is distance from entry to SL in price terms.
    """
    spread_price = SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)
    risk_distance = max(atr_price * max(eff_sl_mult, 0.3), 1e-8)
    return (spread_price / risk_distance)

@njit
def resolve_trade_fast(opens, highs, lows, closes, entry_mid, sl_price, tp_price, direction, spread):
    """
    Realistic trade resolution with Bid/Ask spread, SL/TP side awareness, and gap handling.
    - Long Entry: Ask = Mid + spread/2
    - Long Exit/Triggers: Bid = Price - spread/2
    - Short Entry: Bid = Mid - spread/2
    - Short Exit/Triggers: Ask = Price + spread/2
    - Gap: If Open breaches target, fill at Open.
    - Collision: If SL and TP both hit in same candle, SL hit is assumed (conservative).
    """
    half_spread = spread / 2.0

    if direction == 1:  # Long
        actual_entry = entry_mid + half_spread
    else:               # Short
        actual_entry = entry_mid - half_spread

    sl_dist = abs(actual_entry - sl_price)
    if sl_dist < 1e-10: sl_dist = 1e-10

    for i in range(len(opens)):
        o = opens[i]
        h = highs[i]
        l = lows[i]

        if direction == 1:  # Long
            bid_o = o - half_spread
            bid_h = h - half_spread
            bid_l = l - half_spread

            # 1. Check for gap at open
            if bid_o <= sl_price: return 0, (bid_o - actual_entry) / sl_dist, i + 1
            if bid_o >= tp_price: return 1, (bid_o - actual_entry) / sl_dist, i + 1
            # 2. Check intra-candle (SL takes priority)
            if bid_l <= sl_price: return 0, (sl_price - actual_entry) / sl_dist, i + 1
            if bid_h >= tp_price: return 1, (tp_price - actual_entry) / sl_dist, i + 1
        else:               # Short
            ask_o = o + half_spread
            ask_h = h + half_spread
            ask_l = l + half_spread

            # 1. Check for gap at open
            if ask_o >= sl_price: return 0, (actual_entry - ask_o) / sl_dist, i + 1
            if ask_o <= tp_price: return 1, (actual_entry - ask_o) / sl_dist, i + 1
            # 2. Check intra-candle (SL takes priority)
            if ask_h >= sl_price: return 0, (actual_entry - sl_price) / sl_dist, i + 1
            if ask_l <= tp_price: return 1, (actual_entry - tp_price) / sl_dist, i + 1

    # 3. Timeout at final Close
    if direction == 1:
        pnl = (closes[-1] - half_spread) - actual_entry
    else:
        pnl = actual_entry - (closes[-1] + half_spread)

    return 2, pnl / sl_dist, len(closes) # 2: timeout

class VectorizedRiskEnv(VecEnv):
    """
    Simplified Vectorized Environment for Risk Model Training.
    Focuses on speed by avoiding overhead and using NumPy.
    Extends VecEnv to be compatible with SB3.
    """
    def __init__(self, signals_df, price_data, n_envs=1):
        self.signals = signals_df
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']

        # Cache arrays for fast access
        self.all_features = np.stack(self.signals['features'].values).astype(np.float32)
        self.all_assets = self.signals['asset'].values
        self.all_directions = self.signals['direction'].values.astype(np.int32)

        # Pre-calculate entry data
        self.open_arrays = {a: price_data[a]['open'].values.astype(np.float32) for a in self.assets}
        self.close_arrays = {a: price_data[a]['close'].values.astype(np.float32) for a in self.assets}
        self.high_arrays = {a: price_data[a]['high'].values.astype(np.float32) for a in self.assets}
        self.low_arrays = {a: price_data[a]['low'].values.astype(np.float32) for a in self.assets}
        self.atr_arrays = {a: price_data[a]['atr_14'].values.astype(np.float32) for a in self.assets}

        # Index lookup for speed
        self.time_to_idx = {a: {t: i for i, t in enumerate(price_data[a].index)} for a in self.assets}
        self.signal_step_indices = np.array([self.time_to_idx[row['asset']][row['timestamp']] for _, row in self.signals.iterrows()], dtype=np.int32)

        self.num_signals = len(self.signals)
        self.current_indices = np.random.randint(0, self.num_signals, size=n_envs)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Mocking enough for VecEnv.__init__
        self.num_envs = n_envs
        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = None
        self.metadata = {"render_modes": []}

    def reset(self):
        self.current_indices = np.random.randint(0, self.num_signals, size=self.num_envs)
        return self.all_features[self.current_indices]

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        # actions: [n_envs, 3]
        actions = self.actions
        sl_mults = 1.0 + (actions[:, 0] + 1) / 2 * (3.5 - 1.0)
        tp_mults = 1.2 + (actions[:, 1] + 1) / 2 * (8.0 - 1.2)
        size_pcts = 0.1 + (actions[:, 2] + 1) / 2 * (0.3 - 0.1) # Simplified size curriculum

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        infos = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            sig_idx = self.current_indices[i]
            asset = self.all_assets[sig_idx]
            direction = self.all_directions[sig_idx]
            step_idx = self.signal_step_indices[sig_idx]

            entry_price = self.close_arrays[asset][step_idx]
            atr_price = self.atr_arrays[asset][step_idx]

            sl_mult = sl_mults[i]
            tp_mult = tp_mults[i]

            f_start = step_idx + 1
            f_end = min(f_start + 1000, len(self.close_arrays[asset]))

            eff_sl = max(0.3, sl_mult - np.random.uniform(0, EXECUTION_NOISE_ATR))
            eff_tp = tp_mult + np.random.uniform(0, EXECUTION_NOISE_ATR * 0.5)

            sl_price = entry_price - (direction * eff_sl * atr_price)
            tp_price = entry_price + (direction * eff_tp * atr_price)
            spread_price = SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)

            res_code, res_r, res_len = resolve_trade_fast(
                self.open_arrays[asset][f_start:f_end],
                self.high_arrays[asset][f_start:f_end],
                self.low_arrays[asset][f_start:f_end],
                self.close_arrays[asset][f_start:f_end],
                entry_price, sl_price, tp_price, direction, spread_price
            )

            # reward is already spread-inclusive via res_r
            reward = res_r
            reward *= (size_pcts[i] / 0.1)
            rewards[i] = np.clip(reward, -10.0, 10.0)

            self.current_indices[i] = (self.current_indices[i] + 1) % self.num_signals

        obs = self.all_features[self.current_indices]
        dones = np.zeros(self.num_envs, dtype=bool)

        return obs, rewards, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in range(self.num_envs)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in range(self.num_envs)]

    def seed(self, seed=None):
        pass

class RiskModelLogger:
    """
    Tracks trading metrics per environment instance.
    """
    def __init__(self, window=500):
        self.w = window
        self.reset_window()

    def reset_window(self):
        self.outcomes         = deque(maxlen=self.w)
        self.rewards          = deque(maxlen=self.w)
        self.r_multiples      = deque(maxlen=self.w)
        self.sl_mults         = deque(maxlen=self.w)
        self.tp_mults         = deque(maxlen=self.w)
        self.rr_ratios        = deque(maxlen=self.w)
        self.sl_below_1x      = deque(maxlen=self.w)
        self.sl_below_08x     = deque(maxlen=self.w)
        self.sizes            = deque(maxlen=self.w)
        self.oversize_flags   = deque(maxlen=self.w)
        self.spread_costs_r   = deque(maxlen=self.w)
        self.asset_wins       = {}
        self.timeout_rs       = deque(maxlen=self.w)

    def log_step(self, outcome, reward, reward_info, asset, size_pct):
        self.outcomes.append(outcome)
        self.rewards.append(reward)
        self.sl_mults.append(reward_info["eff_sl"])
        self.tp_mults.append(reward_info["eff_tp"])
        self.rr_ratios.append(reward_info["rr_ratio"])
        self.spread_costs_r.append(reward_info["spread_cost_r"])
        self.sizes.append(size_pct)

        self.sl_below_1x.append(1 if reward_info["eff_sl"] < 1.0 else 0)
        self.sl_below_08x.append(1 if reward_info["eff_sl"] < 0.8 else 0)
        self.oversize_flags.append(1 if size_pct > 0.25 else 0)

        r_earned = reward_info["base_r"]
        self.r_multiples.append(r_earned)
        if outcome == "timeout":
            self.timeout_rs.append(r_earned)

        if asset not in self.asset_wins:
            self.asset_wins[asset] = deque(maxlen=200)
        self.asset_wins[asset].append(1 if outcome == "tp_hit" else 0)

    def get_summary(self):
        outcomes = list(self.outcomes)
        n = len(outcomes)
        if n == 0: return {}

        wins = outcomes.count("tp_hit")
        losses = outcomes.count("sl_hit")
        win_rewards = [r for r, o in zip(self.rewards, outcomes) if o == "tp_hit"]
        loss_rewards = [abs(r) for r, o in zip(self.rewards, outcomes) if o == "sl_hit"]
        profit_factor = sum(win_rewards) / sum(loss_rewards) if sum(loss_rewards) > 0 else 99.0

        metrics = {
            "trade/win_rate": wins / n,
            "trade/timeout_rate": outcomes.count("timeout") / n,
            "trade/profit_factor": profit_factor,
            "trade/avg_reward": np.mean(self.rewards),
            "trade/avg_r_multiple": np.mean(self.r_multiples),
            "placement/avg_sl_mult": np.mean(self.sl_mults),
            "placement/avg_tp_mult": np.mean(self.tp_mults),
            "placement/avg_rr_ratio": np.mean(self.rr_ratios),
            "placement/pct_sl_below_1x_atr": np.mean(self.sl_below_1x),
            "placement/pct_sl_below_08x_atr": np.mean(self.sl_below_08x),
            "sizing/avg_size_pct": np.mean(self.sizes),
            "cost/avg_spread_r": np.mean(self.spread_costs_r),
            "timeout/avg_r_on_timeout": np.mean(self.timeout_rs) if self.timeout_rs else 0,
        }
        # Asset win rates
        for a, v in self.asset_wins.items():
            metrics[f"asset/{a}/win_rate"] = np.mean(list(v))
        
        return metrics

class RiskPPOEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir='data', is_training=True, alpha_model_path=None, dataset_path=None):
        super(RiskPPOEnv, self).__init__()
        
        self.data_dir = data_dir
        self.is_training = is_training
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.logger_util = RiskModelLogger(window=500)
        
        # 1. Load Pre-filtered Dataset
        self.signal_data = None
        if dataset_path and os.path.exists(dataset_path):
            try:
                self.signal_data = pd.read_parquet(dataset_path)
                logging.info(f"Fast Mode: Loaded {len(self.signal_data)} signals.")
                if isinstance(self.signal_data['features'].iloc[0], (list, np.ndarray)):
                    self.signal_data['features'] = self.signal_data['features'].apply(np.array)
            except Exception as e:
                logging.error(f"Failed to load dataset: {e}")

        # 2. Load Raw Data
        self.data = self._load_data()
        
        self.feature_engine = FeatureEngine()
        self.raw_data, self.processed_data = self.feature_engine.preprocess_data(self.data)
        self._cache_data_arrays(self.raw_data)
        self.data_index = self.raw_data.index

        if self.signal_data is None:
            from Alpha.src.model import AlphaSLModel
            import torch
            self.alpha_model = AlphaSLModel(input_dim=40)
            if alpha_model_path is None:
                alpha_model_path = os.path.join(os.path.dirname(__file__), "../../Alpha/models/alpha_model.pth")
            if os.path.exists(alpha_model_path):
                self.alpha_model.load_state_dict(torch.load(alpha_model_path, map_location="cpu"))
            self.alpha_model.eval()
            self.device = torch.device("cpu")
            self.QUAL_THRESHOLD = 0.30
            self.META_THRESHOLD = 0.78
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        self.current_step = 0
        self.total_steps_counter = 0
        self.current_asset_idx = 0
        self.current_signal_idx = 0 
        self.LEVERAGE = 100

    def _load_data(self):
        data = {}
        for asset in self.assets:
            file_path = f"{self.data_dir}/{asset}_5m.parquet"
            if not os.path.exists(file_path):
                fallbacks = [
                    os.path.join(os.path.dirname(__file__), "..", "..", "data", f"{asset}_5m.parquet"),
                    os.path.join(os.path.dirname(__file__), "..", "data", f"{asset}_5m.parquet"),
                    f"/kaggle/working/TradingBot/data/{asset}_5m.parquet"
                ]
                for fb in fallbacks:
                    if os.path.exists(fb):
                        file_path = fb
                        break
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                logging.error(f"Failed to load {asset}: {e}")
                dates = pd.date_range(start='2024-01-01', periods=2000, freq='5min')
                df = pd.DataFrame(index=dates)
                df[f"{asset}_open"] = 1.0; df[f"{asset}_high"] = 1.01; df[f"{asset}_low"] = 0.99; df[f"{asset}_close"] = 1.0; df[f"{asset}_volume"] = 100; df[f"{asset}_atr_14"] = 0.001
            data[asset] = df
        return data

    def _cache_data_arrays(self, df):
        self.open_arrays = {a: df[f"{a}_open"].values.astype(np.float32) for a in self.assets}
        self.close_arrays = {a: df[f"{a}_close"].values.astype(np.float32) for a in self.assets}
        self.low_arrays = {a: df[f"{a}_low"].values.astype(np.float32) for a in self.assets}
        self.high_arrays = {a: df[f"{a}_high"].values.astype(np.float32) for a in self.assets}
        self.atr_arrays = {a: df[f"{a}_atr_14"].values.astype(np.float32) for a in self.assets}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.signal_data is not None:
            self.current_signal_idx = np.random.randint(0, len(self.signal_data))
            self._apply_current_signal()
        else:
            self.current_step = np.random.randint(500, len(self.data_index) - 1001)
            self._find_next_signal()
        
        return self._get_observation(), {}

    def _apply_current_signal(self):
        row = self.signal_data.iloc[self.current_signal_idx]
        self.current_asset_idx = self.assets.index(row['asset'])
        self.current_direction = int(row['direction'])
        self.current_obs = row['features']
        try:
            self.current_step = self.data_index.get_loc(row['timestamp'])
        except:
            self.current_step = np.random.randint(500, len(self.data_index) - 1001)

    def _find_next_signal(self):
        if self.signal_data is not None:
            self.current_signal_idx = (self.current_signal_idx + 1) % len(self.signal_data)
            self._apply_current_signal()
            return

        import torch
        max_search = 5000
        found = False
        while not found and max_search > 0:
            self.current_asset_idx = np.random.randint(0, len(self.assets))
            asset = self.assets[self.current_asset_idx]
            obs = self._get_observation()
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                dir_logits, quality, meta_logits = self.alpha_model(obs_t)
                direction = torch.argmax(dir_logits, dim=1).item() - 1 
                qual_val = torch.sigmoid(quality).item()
                meta_val = torch.sigmoid(meta_logits).item()
            if direction != 0 and qual_val >= self.QUAL_THRESHOLD and meta_val >= self.META_THRESHOLD:
                self.current_direction = direction
                found = True
            else:
                self.current_step += 1
                if self.current_step >= len(self.data_index) - 1001: self.current_step = 500
                max_search -= 1

    def _get_observation(self):
        if self.signal_data is not None:
            return self.current_obs
        asset = self.assets[self.current_asset_idx]
        current_row = self.processed_data.iloc[self.current_step]
        return self.feature_engine.get_observation(current_row, asset=asset)

    def _get_curriculum_size_range(self, current_step, phase2_start=500_000, warmup_steps=100_000):
        if current_step < phase2_start:
            return 0.10, 0.10
        progress = min(1.0, (current_step - phase2_start) / warmup_steps)
        size_min = 0.01
        size_max = 0.10 + progress * (0.30 - 0.10)
        return size_min, size_max

    def step(self, action):
        self.total_steps_counter += 1
        action = np.nan_to_num(action, nan=0.0)
        sl_raw, tp_raw, size_raw = action[0], action[1], action[2]
        
        sl_mult = 1.0 + (sl_raw + 1) / 2 * (3.5 - 1.0)
        tp_mult = 1.2 + (tp_raw + 1) / 2 * (8.0 - 1.2)
        size_min, size_max = self._get_curriculum_size_range(self.total_steps_counter)
        size_pct = size_min + (size_raw + 1) / 2 * (size_max - size_min)
        
        reward, reward_info, outcome = self._simulate_outcome_v3(sl_mult, tp_mult, size_pct, self.current_direction)
        
        # Log to internal logger
        asset = self.assets[self.current_asset_idx]
        self.logger_util.log_step(outcome, reward, reward_info, asset, size_pct)
        
        self.current_step += 1
        if self.current_step >= len(self.data_index) - 1001: self.current_step = 500
        self._find_next_signal()
        obs = self._get_observation()
        
        # Return summary in info for callback logging
        info = self.logger_util.get_summary()
        info['is_success'] = (outcome == "tp_hit")
        
        return obs, float(reward), False, False, info

    def _simulate_outcome_v3(self, sl_mult, tp_mult, size_pct, direction):
        asset = self.assets[self.current_asset_idx]
        entry_price = self.close_arrays[asset][self.current_step]
        atr_price = self.atr_arrays[asset][self.current_step]
        if atr_price <= 0: atr_price = entry_price * 0.001
        
        eff_sl_mult, eff_tp_mult = add_execution_noise(sl_mult, tp_mult)
        sl_price = entry_price - (direction * eff_sl_mult * atr_price)
        tp_price = entry_price + (direction * eff_tp_mult * atr_price)
        
        start_idx = self.current_step + 1
        end_idx = min(start_idx + 1000, len(self.data_index))
        future_opens = self.open_arrays[asset][start_idx:end_idx]
        future_highs = self.high_arrays[asset][start_idx:end_idx]
        future_lows = self.low_arrays[asset][start_idx:end_idx]
        future_closes = self.close_arrays[asset][start_idx:end_idx]
        
        outcome, timeout_r, hold_time = self._resolve_trade_v2(
            future_opens, future_highs, future_lows, future_closes,
            entry_price, sl_price, tp_price, direction, asset
        )
        
        reward, reward_info = self._compute_reward_v3(
            outcome, eff_sl_mult, eff_tp_mult, size_pct, timeout_r, asset, atr_price
        )
        
        return reward, reward_info, outcome

    def _resolve_trade_v2(self, future_opens, future_highs, future_lows, future_closes, entry_price, sl_price, tp_price, direction, asset):
        spread_price = SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)
        
        res_code, res_r, res_len = resolve_trade_fast(
            future_opens, future_highs, future_lows, future_closes,
            entry_price, sl_price, tp_price, direction, spread_price
        )

        outcome = "sl_hit" if res_code == 0 else "tp_hit" if res_code == 1 else "timeout"
        return outcome, res_r, res_len

    def _compute_reward_v3(self, outcome, eff_sl, eff_tp, size_pct, timeout_r, asset, atr_price):
        rr_ratio = eff_tp / eff_sl

        # Use realized trade outcome (spread-inclusive from resolver) to avoid
        # optimistic reward leakage from theoretical RR-only scoring.
        if outcome in ("tp_hit", "sl_hit", "timeout"):
            base_r = timeout_r
        else:
            base_r = 0.0

        spread_cost = compute_spread_cost_in_r(asset, atr_price, eff_sl)
        reward = base_r - spread_cost

        shape = 0.0
        if 1.2 <= rr_ratio <= 3.5: shape += 0.05
        if 1.0 <= eff_sl <= 2.5: shape += 0.05
        
        penalty = 0.0
        if eff_sl < 1.0: penalty -= (0.5 + (1.0 - eff_sl) * 0.7)
        if eff_sl < 1.2: penalty -= 0.15
        if rr_ratio < 1.1: penalty -= 0.3
        if eff_sl > 3.0 and rr_ratio < 1.5: penalty -= 0.2
        if outcome == "sl_hit" and eff_sl < 1.25:
            penalty -= 0.2
        if outcome == "tp_hit" and 1.2 <= eff_sl <= 2.8:
            shape += 0.08
        
        reward = reward + shape + penalty
        size_scalar = np.clip(size_pct / 0.10, 0.1, 5.0)
        final_reward = np.clip(reward * size_scalar, -10.0, 10.0)
        
        info = {
            "base_r": base_r,
            "eff_sl": eff_sl,
            "eff_tp": eff_tp,
            "rr_ratio": rr_ratio,
            "spread_cost_r": spread_cost,
            "final": final_reward
        }
        
        return final_reward, info

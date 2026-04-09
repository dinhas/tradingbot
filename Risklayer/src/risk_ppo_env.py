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

# Unified Feature Engine from Alpha Layer
try:
    import sys
    import os
    # Ensure project root is in path for Alpha imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from Alpha.src.feature_engine import FeatureEngine
except ImportError:
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
    "XAUUSD": 45.0,
}

PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "USDCHF": 0.0001,
    "XAUUSD": 0.1,
}

EXECUTION_NOISE_ATR = 0.15
BREAKEVEN_TRIGGER_R = 0.9
BREAKEVEN_BUFFER_ATR = 0.10
TRAILING_TRIGGER_R = 1.25

# Transaction Fees (4 bps total to match backtest)
LEVERAGE = 100.0

def add_execution_noise(sl_mult, tp_mult):
    """
    Apply conservative execution uncertainty.
    Returns effective multipliers used for both simulation and reward.
    """
    # Noise on SL makes it slightly closer (worse risk)
    eff_sl = max(0.3, sl_mult - np.random.uniform(0, EXECUTION_NOISE_ATR))
    # Noise on TP makes it slightly closer (worse reward)
    eff_tp = max(eff_sl + 0.1, tp_mult - np.random.uniform(0, EXECUTION_NOISE_ATR * 0.5))
    return eff_sl, eff_tp

def compute_fee_cost_in_r(entry_price, atr_price, eff_sl_mult):
    """
    Approximate 4 bps transaction fee in R units.
    """
    risk_dist = max(atr_price * max(eff_sl_mult, 0.3), 1e-8)
    fee_r = (0.00004 * entry_price) / (risk_dist * LEVERAGE)
    return fee_r

@njit
def resolve_trade_with_trailing_fast(
    opens, highs, lows, closes, atrs,
    entry_mid, sl_price, tp_price, direction, spread,
    be_trigger_r, be_buffer_atr, trailing_trigger_r,
):
    half_spread = spread / 2.0
    if direction == 1:
        actual_entry = entry_mid + half_spread
    else:
        actual_entry = entry_mid - half_spread

    initial_risk = abs(actual_entry - sl_price)
    if initial_risk < 1e-10: initial_risk = 1e-10

    current_sl = sl_price
    trailing_active = False
    trailing_distance = 0.0
    best_price = actual_entry

    for i in range(len(opens)):
        o, h, l, atr_i = opens[i], highs[i], lows[i], atrs[i]
        if atr_i <= 0: atr_i = max(actual_entry, 1e-9) * 1e-4

        if direction == 1:
            bid_o, bid_h, bid_l = o - half_spread, h - half_spread, l - half_spread
            if bid_h > best_price: best_price = bid_h
            favorable = best_price - actual_entry

            if not trailing_active:
                if favorable >= be_trigger_r * initial_risk:
                    be_stop = actual_entry + (be_buffer_atr * atr_i)
                    if be_stop > current_sl: current_sl = be_stop
                if favorable >= trailing_trigger_r * initial_risk:
                    trailing_distance = abs(bid_h - current_sl)
                    trailing_active = True

            if trailing_active:
                new_sl = bid_h - trailing_distance
                if new_sl > current_sl: current_sl = new_sl

            if bid_l <= current_sl: return 0, (current_sl - actual_entry) / initial_risk, i + 1
            if bid_o >= tp_price: return 1, (bid_o - actual_entry) / initial_risk, i + 1
            if bid_h >= tp_price: return 1, (tp_price - actual_entry) / initial_risk, i + 1
        else:
            ask_o, ask_h, ask_l = o + half_spread, h + half_spread, l + half_spread
            if ask_l < best_price: best_price = ask_l
            favorable = actual_entry - best_price

            if not trailing_active:
                if favorable >= be_trigger_r * initial_risk:
                    be_stop = actual_entry - (be_buffer_atr * atr_i)
                    if be_stop < current_sl: current_sl = be_stop
                if favorable >= trailing_trigger_r * initial_risk:
                    trailing_distance = abs(ask_l - current_sl)
                    trailing_active = True

            if trailing_active:
                new_sl = ask_l + trailing_distance
                if new_sl < current_sl: current_sl = new_sl

            if ask_h >= current_sl: return 0, (actual_entry - current_sl) / initial_risk, i + 1
            if ask_o <= tp_price: return 1, (actual_entry - ask_o) / initial_risk, i + 1
            if ask_l <= tp_price: return 1, (actual_entry - tp_price) / initial_risk, i + 1

    if direction == 1:
        pnl = (closes[-1] - half_spread) - actual_entry
    else:
        pnl = actual_entry - (closes[-1] + half_spread)

    return 2, pnl / initial_risk, len(closes)

class VectorizedRiskEnv(VecEnv):
    def __init__(self, signals_df, price_data, n_envs=1):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        super().__init__(n_envs, obs_space, act_space)

        self.signals = signals_df
        self.all_features = np.stack(self.signals['features'].values).astype(np.float32)
        self.all_assets = self.signals['asset'].values
        self.all_directions = self.signals['direction'].values.astype(np.int32)

        self.open_arrays = {a: price_data[a]['open'].values.astype(np.float32) for a in self.assets}
        self.close_arrays = {a: price_data[a]['close'].values.astype(np.float32) for a in self.assets}
        self.high_arrays = {a: price_data[a]['high'].values.astype(np.float32) for a in self.assets}
        self.low_arrays = {a: price_data[a]['low'].values.astype(np.float32) for a in self.assets}
        self.atr_arrays = {a: price_data[a]['atr_14'].values.astype(np.float32) for a in self.assets}

        self.time_to_idx = {a: {t: i for i, t in enumerate(price_data[a].index)} for a in self.assets}
        
        indices = []
        for _, row in self.signals.iterrows():
            try:
                indices.append(self.time_to_idx[row['asset']][row['timestamp']])
            except KeyError:
                # Fallback to random or skip if timestamp missing
                indices.append(np.random.randint(0, len(price_data[row['asset']])))
        self.signal_step_indices = np.array(indices, dtype=np.int32)

        self.num_signals = len(self.signals)
        self.current_indices = np.random.randint(0, self.num_signals, size=n_envs)
        self.steps_in_episode = np.zeros(n_envs, dtype=np.int32)
        self.total_global_steps = 0

    def reset(self):
        self.current_indices = np.random.randint(0, self.num_signals, size=self.num_envs)
        self.steps_in_episode.fill(0)
        return self.all_features[self.current_indices]

    def step_async(self, actions):
        self.actions = actions

    def _get_curriculum_size(self, global_step):
        # Match RiskPPOEnv size range: starts at 0.1, grows to 0.3 after phase2
        if global_step < 500_000:
            return 0.10
        warmup = 100_000
        progress = min(1.0, (global_step - 500_000) / warmup)
        return 0.10 + progress * (0.30 - 0.10)

    def step_wait(self):
        actions = self.actions
        self.total_global_steps += self.num_envs
        
        sl_mults = 1.0 + (actions[:, 0] + 1) / 2 * (3.5 - 1.0)
        tp_mults = 1.2 + (actions[:, 1] + 1) / 2 * (8.0 - 1.2)
        
        # Consistent curriculum
        size_max = self._get_curriculum_size(self.total_global_steps)
        size_min = 0.01 if self.total_global_steps >= 500_000 else size_max
        size_pcts = size_min + (actions[:, 2] + 1) / 2 * (size_max - size_min)

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            sig_idx = self.current_indices[i]
            asset = self.all_assets[sig_idx]
            direction = self.all_directions[sig_idx]
            step_idx = self.signal_step_indices[sig_idx]

            entry_price = self.close_arrays[asset][step_idx]
            atr_price = self.atr_arrays[asset][step_idx]
            eff_sl, eff_tp = add_execution_noise(sl_mults[i], tp_mults[i])

            sl_p = entry_price - (direction * eff_sl * atr_price)
            tp_p = entry_price + (direction * eff_tp * atr_price)
            spread_p = SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)

            f_start = step_idx + 1
            f_end = min(f_start + 1000, len(self.close_arrays[asset]))

            res_code, res_r, res_len = resolve_trade_with_trailing_fast(
                self.open_arrays[asset][f_start:f_end],
                self.high_arrays[asset][f_start:f_end],
                self.low_arrays[asset][f_start:f_end],
                self.close_arrays[asset][f_start:f_end],
                self.atr_arrays[asset][f_start:f_end],
                entry_price, sl_p, tp_p, direction, spread_p,
                BREAKEVEN_TRIGGER_R, BREAKEVEN_BUFFER_ATR, TRAILING_TRIGGER_R
            )

            fee_r = compute_fee_cost_in_r(entry_price, atr_price, eff_sl)
            reward = res_r - fee_r # Correct: spread already in res_r
            
            reward *= (size_pcts[i] / 0.1)
            rewards[i] = np.clip(reward, -10.0, 10.0)

            self.steps_in_episode[i] += 1
            if self.steps_in_episode[i] >= 500:
                dones[i] = True
                infos[i]["terminal_observation"] = self.all_features[self.current_indices[i]]
                self.current_indices[i] = np.random.randint(0, self.num_signals)
                self.steps_in_episode[i] = 0
            else:
                self.current_indices[i] = (self.current_indices[i] + 1) % self.num_signals

        return self.all_features[self.current_indices], rewards, dones, infos

    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [getattr(self, attr_name) for _ in range(self.num_envs)]
    def set_attr(self, attr_name, value, indices=None): setattr(self, attr_name, value)
    def env_method(self, method_name, *method_args, indices=None, **method_kwargs): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False for _ in range(self.num_envs)]

class RiskModelLogger:
    def __init__(self, window=500):
        self.w = window
        self.reset_window()

    def reset_window(self):
        self.outcomes, self.rewards, self.r_multiples = deque(maxlen=self.w), deque(maxlen=self.w), deque(maxlen=self.w)
        self.sl_mults, self.tp_mults, self.rr_ratios = deque(maxlen=self.w), deque(maxlen=self.w), deque(maxlen=self.w)
        self.sizes, self.asset_wins = deque(maxlen=self.w), {}

    def log_step(self, outcome, reward, reward_info, asset, size_pct):
        self.outcomes.append(outcome)
        self.rewards.append(reward)
        self.r_multiples.append(reward_info["base_r"])
        self.sl_mults.append(reward_info["eff_sl"])
        self.tp_mults.append(reward_info["eff_tp"])
        self.rr_ratios.append(reward_info["rr_ratio"])
        self.sizes.append(size_pct)
        if asset not in self.asset_wins: self.asset_wins[asset] = deque(maxlen=200)
        self.asset_wins[asset].append(1 if outcome == "tp_hit" else 0)

    def get_summary(self):
        if not self.outcomes: return {}
        outcomes = list(self.outcomes)
        r_list = list(self.r_multiples)
        
        wins_r = [r for r, o in zip(r_list, outcomes) if o == "tp_hit"]
        loss_r = [abs(r) for r, o in zip(r_list, outcomes) if o == "sl_hit"]
        pf = sum(wins_r) / sum(loss_r) if sum(loss_r) > 0 else 99.0

        metrics = {
            "trade/win_rate": outcomes.count("tp_hit") / len(outcomes),
            "trade/profit_factor": pf,
            "trade/avg_reward": np.mean(self.rewards),
            "trade/avg_r_multiple": np.mean(self.r_multiples),
            "placement/avg_sl_mult": np.mean(self.sl_mults),
            "placement/avg_rr_ratio": np.mean(self.rr_ratios),
            "sizing/avg_size_pct": np.mean(self.sizes),
        }
        return metrics

class RiskPPOEnv(gym.Env):
    def __init__(self, data_dir='data', is_training=True, alpha_model_path=None, dataset_path=None):
        super().__init__()
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.data_dir, self.is_training = data_dir, is_training
        self.logger_util = RiskModelLogger(window=500)
        
        self.signal_data = None
        if dataset_path and os.path.exists(dataset_path):
            self.signal_data = pd.read_parquet(dataset_path)
            if isinstance(self.signal_data['features'].iloc[0], (list, np.ndarray)):
                self.signal_data['features'] = self.signal_data['features'].apply(np.array)

        self.data = self._load_data()
        self.feature_engine = FeatureEngine()
        self.raw_data, self.processed_data = self.feature_engine.preprocess_data(self.data)
        self._cache_data_arrays(self.raw_data)
        self.data_index = self.raw_data.index

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.current_step, self.total_steps_counter, self.steps_in_episode = 0, 0, 0

    def _load_data(self):
        data = {}
        for asset in self.assets:
            file_path = f"{self.data_dir}/{asset}_5m.parquet"
            if not os.path.exists(file_path):
                file_path = os.path.join(os.path.dirname(__file__), "../../data", f"{asset}_5m.parquet")
            try: df = pd.read_parquet(file_path)
            except: 
                dates = pd.date_range(start='2024-01-01', periods=2000, freq='5min')
                df = pd.DataFrame(index=dates)
                for c in ['open','high','low','close']: df[f"{asset}_{c}"] = 1.0
                df[f"{asset}_atr_14"] = 0.001
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
        self.steps_in_episode = 0
        if self.signal_data is not None:
            self.current_signal_idx = np.random.randint(0, len(self.signal_data))
            self._apply_current_signal()
        else:
            self.current_step = np.random.randint(500, len(self.data_index) - 1001)
        return self._get_observation(), {}

    def _apply_current_signal(self):
        row = self.signal_data.iloc[self.current_signal_idx]
        self.current_asset_idx = self.assets.index(row['asset'])
        self.current_direction = int(row['direction'])
        self.current_obs = row['features']
        try: self.current_step = self.data_index.get_loc(row['timestamp'])
        except: self.current_step = np.random.randint(500, len(self.data_index) - 1001)

    def _get_observation(self):
        if self.signal_data is not None: return self.current_obs
        asset = self.assets[getattr(self, 'current_asset_idx', 0)]
        return self.feature_engine.get_observation(self.processed_data.iloc[self.current_step], {}, asset)

    def step(self, action):
        self.total_steps_counter += 1
        self.steps_in_episode += 1
        sl_mult = 1.0 + (action[0] + 1) / 2 * (3.5 - 1.0)
        tp_mult = 1.2 + (action[1] + 1) / 2 * (8.0 - 1.2)
        
        # Curriculum
        if self.total_steps_counter < 500_000:
            size_min, size_max = 0.10, 0.10
        else:
            progress = min(1.0, (self.total_steps_counter - 500_000) / 100_000)
            size_min, size_max = 0.01, 0.10 + progress * 0.20
        size_pct = size_min + (action[2] + 1) / 2 * (size_max - size_min)

        reward, info_dict, outcome = self._simulate(sl_mult, tp_mult, size_pct)
        self.logger_util.log_step(outcome, reward, info_dict, self.assets[self.current_asset_idx], size_pct)
        
        # Advance
        if self.signal_data is not None:
            self.current_signal_idx = (self.current_signal_idx + 1) % len(self.signal_data)
            self._apply_current_signal()
        else:
            self.current_step = (self.current_step + 1) % (len(self.data_index) - 1001)
        
        truncated = self.steps_in_episode >= 500
        return self._get_observation(), float(reward), False, truncated, self.logger_util.get_summary()

    def _simulate(self, sl_m, tp_m, size):
        asset = self.assets[self.current_asset_idx]
        entry_p = self.close_arrays[asset][self.current_step]
        atr_p = max(self.atr_arrays[asset][self.current_step], entry_p * 1e-4)
        
        eff_sl, eff_tp = add_execution_noise(sl_m, tp_m)
        sl_p = entry_p - (self.current_direction * eff_sl * atr_p)
        tp_p = entry_p + (self.current_direction * eff_tp * atr_p)
        
        f_s = self.current_step + 1
        f_e = min(f_s + 1000, len(self.data_index))
        
        res_code, res_r, _ = resolve_trade_with_trailing_fast(
            self.open_arrays[asset][f_s:f_e], self.high_arrays[asset][f_s:f_e],
            self.low_arrays[asset][f_s:f_e], self.close_arrays[asset][f_s:f_e],
            self.atr_arrays[asset][f_s:f_e], entry_p, sl_p, tp_p, self.current_direction,
            SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001),
            BREAKEVEN_TRIGGER_R, BREAKEVEN_BUFFER_ATR, TRAILING_TRIGGER_R
        )

        fee_r = compute_fee_cost_in_r(entry_p, atr_p, eff_sl)
        reward = res_r - fee_r
        
        # Shaping
        shape = 0.0
        if 1.2 <= (eff_tp/eff_sl) <= 3.5: shape += 0.05
        if eff_sl < 1.0: shape -= (0.5 + (1.0 - eff_sl) * 0.7)
        
        reward = (reward + shape) * (size / 0.10)
        final_reward = np.clip(reward, -10.0, 10.0)
        
        outcome = "sl_hit" if res_code == 0 else "tp_hit" if res_code == 1 else "timeout"
        info = {"base_r": res_r, "eff_sl": eff_sl, "eff_tp": eff_tp, "rr_ratio": eff_tp/eff_sl}
        return final_reward, info, outcome

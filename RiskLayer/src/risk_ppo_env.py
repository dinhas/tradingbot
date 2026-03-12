"""
Risk Model PPO Environment v3
================================
Fixes from v2:

BUG 1 FIXED: add_execution_noise() was called twice — once in _simulate_outcome_v3
             to compute actual SL/TP prices, then AGAIN in _compute_reward_v3 with
             different random values. Trade resolved on noise-A, reward computed on
             noise-B. Gradient signal was completely decoupled from actual behavior.
             Fix: compute noise ONCE, pass eff_sl/eff_tp through to reward function.

BUG 2 FIXED: Tight SL penalty (-0.4) was ~10% of a high-RR TP reward (+5R).
             Agent learned "take the lottery ticket, penalty is noise."
             Fix: penalty now scales proportionally to the potential reward,
             making tight SL genuinely expensive even on wins.

BUG 3 FIXED: base_r = rr_ratio had no ceiling. Agent could get +7R from a
             7x TP target, massively incentivizing extreme TP placement.
             Fix: TP reward is capped and uses a realistic hit-probability
             discount — high RR targets get a smaller reward per R because
             they're harder to actually hit.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import os
from collections import deque
from numba import njit
from stable_baselines3.common.vec_env import VecEnv

# ──────────────────────────────────────────────
# SPREAD TABLE
# ──────────────────────────────────────────────
SPREAD_PIPS = {
    "EURUSD": 1.8,
    "GBPUSD": 2.5,
    "USDJPY": 1.5,
    "USDCHF": 2.8,
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


# ──────────────────────────────────────────────
# BUG 1 FIX: Single noise computation, returned
# as a tuple and passed through the call chain.
# ──────────────────────────────────────────────
def compute_execution_noise(sl_mult, tp_mult):
    """
    Compute execution noise ONCE per trade.
    Returns (eff_sl, eff_tp) — use these everywhere for this trade.
    Never call this twice for the same trade step.
    """
    eff_sl = max(0.3, sl_mult - np.random.uniform(0, EXECUTION_NOISE_ATR))
    eff_tp = tp_mult + np.random.uniform(0, EXECUTION_NOISE_ATR * 0.5)
    return float(eff_sl), float(eff_tp)


def compute_spread_cost_in_r(asset, atr_price, eff_sl):
    spread_price = SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)
    sl_distance = atr_price * eff_sl
    if sl_distance < 1e-9:
        return 2.0  # catastrophic — SL is zero
    return (spread_price * 1.5) / sl_distance


@njit
def resolve_trade_fast(highs, lows, closes, entry_price, sl_price, tp_price, direction, spread_price):
    sl_distance = abs(entry_price - sl_price)

    if direction == 1:
        actual_entry = entry_price + spread_price
    else:
        actual_entry = entry_price - spread_price

    for i in range(len(highs)):
        high = highs[i]
        low  = lows[i]

        sl_hit = (direction == 1 and low  <= sl_price) or \
                 (direction == -1 and high >= sl_price)
        tp_hit = (direction == 1 and high >= tp_price) or \
                 (direction == -1 and low  <= tp_price)

        # Conservative: if both hit same candle, SL wins
        if sl_hit:
            return 0, -1.0, i + 1
        if tp_hit:
            tp_dist = abs(tp_price - entry_price)
            r_won = tp_dist / sl_distance if sl_distance > 1e-10 else 0.0
            return 1, r_won, i + 1

    # Timeout — force close with spread on exit
    exit_price = closes[-1]
    if direction == 1:
        exit_price -= spread_price
    else:
        exit_price += spread_price

    pnl = (exit_price - actual_entry) * direction
    timeout_r = pnl / sl_distance if sl_distance > 1e-10 else 0.0
    return 2, timeout_r, len(closes)


# ──────────────────────────────────────────────
# BUG 2 + 3 FIX: New reward function
# ──────────────────────────────────────────────
def compute_reward_v3(outcome, sl_mult, tp_mult, size_pct, timeout_r,
                      asset, atr_price, eff_sl, eff_tp):
    """
    Parameters
    ----------
    eff_sl, eff_tp : floats
        Pre-computed noise values from compute_execution_noise().
        These MUST be the same values used to set SL/TP prices.
        Never recompute inside this function.
    """
    rr_ratio = eff_tp / max(eff_sl, 0.01)

    # ── BASE REWARD ────────────────────────────────────────────
    # BUG 3 FIX: Cap TP reward and apply a realistic hit discount.
    # High RR targets are harder to hit in real markets.
    # A 3R target might hit 35% of the time; a 7R target maybe 10%.
    # We discount the reward so the agent doesn't blindly chase huge TPs.
    if outcome == "tp_hit":
        # Hard cap at 3.5R — beyond this, real hit probability drops sharply.
        # This prevents the "lottery ticket" exploit.
        capped_rr = min(rr_ratio, 3.5)
        
        # Discount factor: RR targets above 2.5 get diminishing reward.
        # Models the fact that a 5R TP is not 2.5x better than a 2R TP
        # because it hits far less often.
        if rr_ratio <= 2.5:
            tp_discount = 1.0          # full reward up to 2.5R
        elif rr_ratio <= 4.0:
            tp_discount = 0.80         # 20% discount for 2.5-4R
        else:
            tp_discount = 0.60         # 40% discount for 4R+
        
        base_r = capped_rr * tp_discount

    elif outcome == "sl_hit":
        base_r = -1.0

    elif outcome == "timeout":
        # Timeout capped at 85% of what a clean TP hit would give
        # (slight discount to keep agent preferring actual TP hits)
        base_r = np.clip(timeout_r, -1.5, rr_ratio * 0.85)
    else:
        base_r = 0.0

    # ── SPREAD COST ────────────────────────────────────────────
    spread_cost = compute_spread_cost_in_r(asset, atr_price, eff_sl)
    reward = base_r - spread_cost

    # ── SHAPING BONUSES ────────────────────────────────────────
    shape = 0.0

    # Reward the "signal zone" SL placement (1.0–2.5x ATR)
    # This is where real market structure (swing highs/lows) lives
    if 1.0 <= eff_sl <= 2.5:
        shape += 0.10

    # Reward balanced, achievable RR (1.5–3.0)
    # Peaks at 2.0 to encourage realistic profit targets
    if 1.5 <= rr_ratio <= 3.0:
        shape += 0.15
    elif 3.0 < rr_ratio <= 4.0:
        shape += 0.05   # still ok but declining incentive

    # ── PENALTIES ──────────────────────────────────────────────
    # BUG 2 FIX: Penalties now scale with the potential reward so
    # they can never be drowned out by a lucky TP hit.
    penalty = 0.0

    # Noise-zone SL: below 0.8x ATR
    if eff_sl < 0.8:
        noise_zone_depth = 0.8 - eff_sl  # 0 to 0.5
        # Base penalty + proportion of what you could have won
        # This makes the penalty meaningful even on TP hits
        penalty -= (0.5 + noise_zone_depth * 1.0 + abs(base_r) * 0.30)

    # Inverted RR
    if rr_ratio < 1.0:
        penalty -= 0.5

    # Absurd TP hunting: RR > 5 without a structural basis
    # The TP discount above handles most of this, but add a soft nudge
    if rr_ratio > 5.0:
        penalty -= (rr_ratio - 5.0) * 0.1

    # Very wide SL with poor RR — lazy risk management
    if eff_sl > 3.0 and rr_ratio < 1.5:
        penalty -= 0.2

    reward = reward + shape + penalty

    # ── SIZE SCALING ───────────────────────────────────────────
    # Size affects magnitude but not the sign/learning direction
    size_scalar = np.clip(size_pct / 0.10, 0.5, 2.0)
    final_reward = np.clip(reward * size_scalar, -3.0, 3.0)

    info = {
        "base_r":       base_r,
        "eff_sl":       eff_sl,
        "eff_tp":       eff_tp,
        "rr_ratio":     rr_ratio,
        "spread_cost_r": spread_cost,
        "shape":        shape,
        "penalty":      penalty,
        "final":        final_reward,
    }
    return final_reward, info


# ──────────────────────────────────────────────
# VECTORIZED ENV (updated)
# ──────────────────────────────────────────────
class VectorizedRiskEnv(VecEnv):
    def __init__(self, signals_df, price_data, n_envs=1):
        self.signals = signals_df
        self.assets  = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']

        self.all_features   = np.stack(self.signals['features'].values).astype(np.float32)
        self.all_assets     = self.signals['asset'].values
        self.all_directions = self.signals['direction'].values.astype(np.int32)

        self.close_arrays = {a: price_data[a]['close'].values.astype(np.float32)   for a in self.assets}
        self.high_arrays  = {a: price_data[a]['high'].values.astype(np.float32)    for a in self.assets}
        self.low_arrays   = {a: price_data[a]['low'].values.astype(np.float32)     for a in self.assets}
        self.atr_arrays   = {a: price_data[a]['atr_14'].values.astype(np.float32)  for a in self.assets}

        self.time_to_idx = {
            a: {t: i for i, t in enumerate(price_data[a].index)}
            for a in self.assets
        }
        self.signal_step_indices = np.array([
            self.time_to_idx[row['asset']][row['timestamp']]
            for _, row in self.signals.iterrows()
        ], dtype=np.int32)

        self.num_signals     = len(self.signals)
        self.current_indices = np.random.randint(0, self.num_signals, size=n_envs)

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        action_space      = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.num_envs         = n_envs
        self.observation_space = observation_space
        self.action_space      = action_space
        self.render_mode       = None
        self.metadata          = {"render_modes": []}

    def reset(self):
        self.current_indices = np.random.randint(0, self.num_signals, size=self.num_envs)
        return self.all_features[self.current_indices]

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        actions   = self.actions
        sl_mults  = 0.8 + (actions[:, 0] + 1) / 2 * (3.5 - 0.8)
        tp_mults  = 1.2 + (actions[:, 1] + 1) / 2 * (8.0 - 1.2)
        size_pcts = 0.10 * np.ones(self.num_envs)  # Phase 1: fixed size

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        infos   = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            sig_idx   = self.current_indices[i]
            asset     = self.all_assets[sig_idx]
            direction = self.all_directions[sig_idx]
            step_idx  = self.signal_step_indices[sig_idx]

            entry_price = self.close_arrays[asset][step_idx]
            atr_price   = self.atr_arrays[asset][step_idx]
            if atr_price <= 0:
                atr_price = entry_price * 0.001

            sl_mult = sl_mults[i]
            tp_mult = tp_mults[i]

            # ── BUG 1 FIX: compute noise ONCE ──
            eff_sl, eff_tp = compute_execution_noise(sl_mult, tp_mult)

            sl_price     = entry_price - (direction * eff_sl * atr_price)
            tp_price     = entry_price + (direction * eff_tp * atr_price)
            spread_price = SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)

            f_start = step_idx + 1
            f_end   = min(f_start + 1000, len(self.close_arrays[asset]))

            res_code, res_r, res_len = resolve_trade_fast(
                self.high_arrays[asset][f_start:f_end],
                self.low_arrays[asset][f_start:f_end],
                self.close_arrays[asset][f_start:f_end],
                entry_price, sl_price, tp_price, direction, spread_price
            )

            outcome  = ["sl_hit", "tp_hit", "timeout"][res_code]
            timeout_r = res_r if res_code == 2 else 0.0

            # ── Pass eff_sl/eff_tp — no recomputation ──
            reward, _ = compute_reward_v3(
                outcome, sl_mult, tp_mult, size_pcts[i],
                timeout_r, asset, atr_price,
                eff_sl, eff_tp   # ← same values used for SL/TP prices
            )

            rewards[i] = reward
            self.current_indices[i] = (self.current_indices[i] + 1) % self.num_signals

        obs   = self.all_features[self.current_indices]
        dones = np.zeros(self.num_envs, dtype=bool)
        return obs, rewards, dones, infos

    def close(self): pass
    def get_attr(self, attr_name, indices=None): return [getattr(self, attr_name) for _ in range(self.num_envs)]
    def set_attr(self, attr_name, value, indices=None): setattr(self, attr_name, value)
    def env_method(self, method_name, *args, indices=None, **kwargs): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def seed(self, seed=None): pass


# ──────────────────────────────────────────────
# GYM ENV (updated — same fixes)
# ──────────────────────────────────────────────
class RiskPPOEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir='data', is_training=True,
                 alpha_model_path=None, dataset_path=None):
        super().__init__()

        self.data_dir    = data_dir
        self.is_training = is_training
        self.assets      = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.logger_util = RiskModelLogger(window=500)

        self.signal_data = None
        if dataset_path and os.path.exists(dataset_path):
            try:
                self.signal_data = pd.read_parquet(dataset_path)
                logging.info(f"Fast Mode: Loaded {len(self.signal_data)} signals.")
                if isinstance(self.signal_data['features'].iloc[0], (list, np.ndarray)):
                    self.signal_data['features'] = self.signal_data['features'].apply(np.array)
            except Exception as e:
                logging.error(f"Failed to load dataset: {e}")

        self.data = self._load_data()
        from frozen_feature_engine import FeatureEngine
        self.feature_engine = FeatureEngine()
        self.raw_data, self.processed_data = self.feature_engine.preprocess_data(self.data)
        self._cache_data_arrays(self.raw_data)
        self.data_index = self.raw_data.index

        if self.signal_data is None:
            from Alpha.src.model import AlphaSLModel
            import torch
            self.alpha_model = AlphaSLModel(input_dim=40)
            if alpha_model_path is None:
                alpha_model_path = os.path.join(
                    os.path.dirname(__file__), "../../Alpha/models/alpha_model.pth")
            if os.path.exists(alpha_model_path):
                self.alpha_model.load_state_dict(
                    torch.load(alpha_model_path, map_location="cpu"))
            self.alpha_model.eval()
            self.device = torch.device("cpu")
            self.QUAL_THRESHOLD = 0.30
            self.META_THRESHOLD = 0.78

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.current_step        = 0
        self.total_steps_counter = 0
        self.current_asset_idx   = 0
        self.current_signal_idx  = 0

    # ── (unchanged helpers) ──────────────────────
    def _load_data(self):
        data = {}
        for asset in self.assets:
            file_path = f"{self.data_dir}/{asset}_5m.parquet"
            fallbacks = [
                os.path.join(os.path.dirname(__file__), "..", "..", "data", f"{asset}_5m.parquet"),
                f"/kaggle/working/TradingBot/data/{asset}_5m.parquet",
            ]
            if not os.path.exists(file_path):
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
                df[f"{asset}_close"] = 1.0; df[f"{asset}_high"] = 1.01
                df[f"{asset}_low"] = 0.99;  df[f"{asset}_volume"] = 100
                df[f"{asset}_atr_14"] = 0.001
            data[asset] = df
        return data

    def _cache_data_arrays(self, df):
        self.close_arrays = {a: df[f"{a}_close"].values.astype(np.float32)  for a in self.assets}
        self.low_arrays   = {a: df[f"{a}_low"].values.astype(np.float32)    for a in self.assets}
        self.high_arrays  = {a: df[f"{a}_high"].values.astype(np.float32)   for a in self.assets}
        self.atr_arrays   = {a: df[f"{a}_atr_14"].values.astype(np.float32) for a in self.assets}

    def _get_curriculum_size_range(self, step, phase2_start=500_000, warmup=100_000):
        if step < phase2_start:
            return 0.10, 0.10
        progress = min(1.0, (step - phase2_start) / warmup)
        return 0.01, 0.10 + progress * 0.20

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
        self.current_asset_idx  = self.assets.index(row['asset'])
        self.current_direction  = int(row['direction'])
        self.current_obs        = row['features']
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
            obs = self._get_observation()
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                dir_logits, quality, meta_logits = self.alpha_model(obs_t)
                direction = torch.argmax(dir_logits, dim=1).item() - 1
                qual_val  = torch.sigmoid(quality).item()
                meta_val  = torch.sigmoid(meta_logits).item()
            if direction != 0 and qual_val >= self.QUAL_THRESHOLD and meta_val >= self.META_THRESHOLD:
                self.current_direction = direction
                found = True
            else:
                self.current_step += 1
                if self.current_step >= len(self.data_index) - 1001:
                    self.current_step = 500
                max_search -= 1

    def _get_observation(self):
        if self.signal_data is not None:
            return self.current_obs
        asset       = self.assets[self.current_asset_idx]
        current_row = self.processed_data.iloc[self.current_step]
        return self.feature_engine.get_observation(current_row, asset=asset)

    def step(self, action):
        self.total_steps_counter += 1
        action = np.nan_to_num(action, nan=0.0)

        sl_mult  = 0.8 + (action[0] + 1) / 2 * (3.5 - 0.8)
        tp_mult  = 1.2 + (action[1] + 1) / 2 * (8.0 - 1.2)
        size_min, size_max = self._get_curriculum_size_range(self.total_steps_counter)
        size_pct = size_min + (action[2] + 1) / 2 * (size_max - size_min)

        asset     = self.assets[self.current_asset_idx]
        entry_price = self.close_arrays[asset][self.current_step]
        atr_price   = self.atr_arrays[asset][self.current_step]
        if atr_price <= 0:
            atr_price = entry_price * 0.001

        # ── BUG 1 FIX: compute noise once, use everywhere ──
        eff_sl, eff_tp = compute_execution_noise(sl_mult, tp_mult)

        sl_price     = entry_price - (self.current_direction * eff_sl * atr_price)
        tp_price     = entry_price + (self.current_direction * eff_tp * atr_price)
        spread_price = SPREAD_PIPS.get(asset, 2.0) * PIP_SIZE.get(asset, 0.0001)

        start_idx     = self.current_step + 1
        end_idx       = min(start_idx + 1000, len(self.data_index))

        res_code, res_r, _ = resolve_trade_fast(
            self.high_arrays[asset][start_idx:end_idx],
            self.low_arrays[asset][start_idx:end_idx],
            self.close_arrays[asset][start_idx:end_idx],
            entry_price, sl_price, tp_price,
            self.current_direction, spread_price
        )

        outcome   = ["sl_hit", "tp_hit", "timeout"][res_code]
        timeout_r = res_r if res_code == 2 else 0.0

        reward, reward_info = compute_reward_v3(
            outcome, sl_mult, tp_mult, size_pct,
            timeout_r, asset, atr_price,
            eff_sl, eff_tp   # ← same values, no recompute
        )

        self.logger_util.log_step(outcome, reward, reward_info, asset, size_pct)

        self.current_step += 1
        if self.current_step >= len(self.data_index) - 1001:
            self.current_step = 500
        self._find_next_signal()

        info = self.logger_util.get_summary()
        info['is_success'] = (outcome == "tp_hit")

        return self._get_observation(), float(reward), False, False, info


# ──────────────────────────────────────────────
# LOGGER (unchanged from v2)
# ──────────────────────────────────────────────
class RiskModelLogger:
    def __init__(self, window=500):
        self.w = window
        self.reset_window()

    def reset_window(self):
        self.outcomes       = deque(maxlen=self.w)
        self.rewards        = deque(maxlen=self.w)
        self.r_multiples    = deque(maxlen=self.w)
        self.sl_mults       = deque(maxlen=self.w)
        self.tp_mults       = deque(maxlen=self.w)
        self.rr_ratios      = deque(maxlen=self.w)
        self.sl_below_1x    = deque(maxlen=self.w)
        self.sl_below_08x   = deque(maxlen=self.w)
        self.sizes          = deque(maxlen=self.w)
        self.oversize_flags = deque(maxlen=self.w)
        self.spread_costs_r = deque(maxlen=self.w)
        self.asset_wins     = {}
        self.timeout_rs     = deque(maxlen=self.w)

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
        self.r_multiples.append(reward_info["base_r"])
        if outcome == "timeout":
            self.timeout_rs.append(reward_info["base_r"])
        if asset not in self.asset_wins:
            self.asset_wins[asset] = deque(maxlen=200)
        self.asset_wins[asset].append(1 if outcome == "tp_hit" else 0)

    def get_summary(self):
        outcomes = list(self.outcomes)
        n = len(outcomes)
        if n == 0:
            return {}
        wins   = outcomes.count("tp_hit")
        losses = outcomes.count("sl_hit")
        win_rewards  = [r for r, o in zip(self.rewards, outcomes) if o == "tp_hit"]
        loss_rewards = [abs(r) for r, o in zip(self.rewards, outcomes) if o == "sl_hit"]
        profit_factor = sum(win_rewards) / sum(loss_rewards) if sum(loss_rewards) > 0 else 99.0

        metrics = {
            "trade/win_rate":                  wins / n,
            "trade/timeout_rate":              outcomes.count("timeout") / n,
            "trade/profit_factor":             profit_factor,
            "trade/avg_reward":                np.mean(self.rewards),
            "trade/avg_r_multiple":            np.mean(self.r_multiples),
            "placement/avg_sl_mult":           np.mean(self.sl_mults),
            "placement/avg_tp_mult":           np.mean(self.tp_mults),
            "placement/avg_rr_ratio":          np.mean(self.rr_ratios),
            "placement/pct_sl_below_1x_atr":   np.mean(self.sl_below_1x),
            "placement/pct_sl_below_08x_atr":  np.mean(self.sl_below_08x),
            "sizing/avg_size_pct":             np.mean(self.sizes),
            "cost/avg_spread_r":               np.mean(self.spread_costs_r),
            "timeout/avg_r_on_timeout":        np.mean(self.timeout_rs) if self.timeout_rs else 0,
        }
        for a, v in self.asset_wins.items():
            metrics[f"asset/{a}/win_rate"] = np.mean(list(v))
        return metrics


# ──────────────────────────────────────────────
# SANITY CHECK
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("V3 REWARD SANITY CHECK — comparing tight vs realistic SL/TP")
    print("=" * 65)

    cases = [
        # (outcome, sl, tp, size, t_r, label)
        ("tp_hit", 0.6, 6.0, 0.10, 0, "EXPLOIT: Tight SL (0.6) + Huge TP (6x) — TP hit"),
        ("sl_hit", 0.6, 6.0, 0.10, 0, "EXPLOIT: Tight SL (0.6) + Huge TP (6x) — SL hit"),
        ("tp_hit", 1.5, 3.0, 0.10, 0, "GOOD:    Signal zone SL (1.5) + TP (3x) — TP hit"),
        ("sl_hit", 1.5, 3.0, 0.10, 0, "GOOD:    Signal zone SL (1.5) + TP (3x) — SL hit"),
        ("tp_hit", 1.2, 2.5, 0.10, 0, "GOOD:    Tight-but-valid SL (1.2) + TP (2.5x) — TP hit"),
        ("tp_hit", 1.2, 2.5, 0.20, 0, "GOOD:    Same setup, doubled size"),
        ("tp_hit", 2.0, 2.5, 0.10, 0, "WIDE:    Wide SL (2.0) + moderate TP — TP hit"),
    ]

    np.random.seed(42)
    for outcome, sl, tp, size, t_r, label in cases:
        # Use fixed noise for reproducible test
        eff_sl = max(0.3, sl - 0.07)
        eff_tp = tp + 0.04
        r, info = compute_reward_v3(
            outcome=outcome, sl_mult=sl, tp_mult=tp, size_pct=size,
            timeout_r=t_r, asset="EURUSD", atr_price=0.0010,
            eff_sl=eff_sl, eff_tp=eff_tp
        )
        print(f"\n{label}")
        print(f"  Reward={info['final']:+.3f}  base_r={info['base_r']:+.2f}  "
              f"RR={info['rr_ratio']:.2f}  spread={info['spread_cost_r']:.3f}R  "
              f"penalty={info['penalty']:+.3f}  shape={info['shape']:+.3f}")
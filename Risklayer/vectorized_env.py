import numpy as np
from Risklayer.config import config
from Risklayer.reward_engine import RewardEngine

class VectorizedTradingEnv:
    def __init__(self, num_envs: int, price_data: dict, signal_data: dict):
        self.num_envs = num_envs
        self.price_data = price_data
        self.signal_data = signal_data
        self.reward_engine = RewardEngine()

        # State buffers
        self.equity = np.full(num_envs, config.INITIAL_EQUITY, dtype=np.float32)
        self.max_equity = np.full(num_envs, config.INITIAL_EQUITY, dtype=np.float32)
        self.drawdown = np.zeros(num_envs, dtype=np.float32)

        # Indices of the current signal for each env
        self.signal_indices = np.random.randint(0, len(signal_data['indices']), size=num_envs)

    def reset_envs(self, mask):
        """Resets environments where mask is True."""
        count = np.sum(mask)
        if count == 0: return

        self.signal_indices[mask] = np.random.randint(0, len(self.signal_data['indices']), size=count)

        # Check for bankruptcy
        bankrupt = self.equity[mask] <= 0.02 * config.INITIAL_EQUITY
        if np.any(bankrupt):
            env_indices = np.where(mask)[0]
            reset_bankrupt = env_indices[bankrupt]
            self.equity[reset_bankrupt] = config.INITIAL_EQUITY
            self.max_equity[reset_bankrupt] = config.INITIAL_EQUITY
            self.drawdown[reset_bankrupt] = 0.0

    def get_observations(self):
        obs = np.zeros((self.num_envs, config.STATE_DIM), dtype=np.float32)

        # Static part from signal data
        obs[:, :32] = self.signal_data['obs_static'][self.signal_indices]

        # Dynamic part
        obs[:, 32] = self.equity / config.INITIAL_EQUITY
        obs[:, 33] = self.drawdown
        obs[:, 34] = 0.0 # Margin
        obs[:, 35] = 0.0 # PosState

        return obs

    def step(self, actions):
        """Vectorized step for all environments."""
        # 1. Denormalize Actions
        sl_mults = (actions[:, 0] + 1) * (config.SL_MULTIPLIER_MAX - config.SL_MULTIPLIER_MIN) / 2 + config.SL_MULTIPLIER_MIN
        rr_ratios = (actions[:, 1] + 1) * (config.RR_RATIO_MAX - config.RR_RATIO_MIN) / 2 + config.RR_RATIO_MIN
        risk_pcts = (actions[:, 2] + 1) * (config.RISK_PERCENT_MAX - config.RISK_PERCENT_MIN) / 2 + config.RISK_PERCENT_MIN

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        pnls = np.zeros(self.num_envs, dtype=np.float32)

        # 2. Extract Signal Info
        assets = self.signal_data['assets'][self.signal_indices]
        global_indices = self.signal_data['indices'][self.signal_indices]
        dirs = self.signal_data['dir'][self.signal_indices]

        # 3. Process each trade (Partially vectorized)
        for i in range(self.num_envs):
            asset = assets[i]
            idx = global_indices[i]
            p_data = self.price_data[asset]

            atr = p_data['atr'][idx]
            close = p_data['close'][idx]
            side = 'long' if dirs[i] > 0 else 'short'

            spread = config.SPREADS.get(asset, 0.0)
            entry_price = close + (spread / 2) if side == 'long' else close - (spread / 2)

            sl_dist = atr * sl_mults[i]
            tp_dist = sl_dist * rr_ratios[i]
            sl_price = entry_price - sl_dist if side == 'long' else entry_price + sl_dist
            tp_price = entry_price + tp_dist if side == 'long' else entry_price - tp_dist

            # Simulation
            start = idx + 1
            end = min(start + 2000, len(p_data['high']))
            f_high = p_data['high'][start:end]
            f_low = p_data['low'][start:end]

            if side == 'long':
                sl_hits = np.where(f_low <= sl_price)[0]
                tp_hits = np.where(f_high >= tp_price)[0]
            else:
                sl_hits = np.where(f_high >= sl_price)[0]
                tp_hits = np.where(f_low <= tp_price)[0]

            first_sl = sl_hits[0] if len(sl_hits) > 0 else 999999
            first_tp = tp_hits[0] if len(tp_hits) > 0 else 999999

            if first_sl == 999999 and first_tp == 999999:
                exit_price = p_data['close'][end-1] if end > start else entry_price
            elif first_sl <= first_tp:
                exit_price = sl_price
            else:
                exit_price = tp_price

            # PnL & Reward
            vol = (self.equity[i] * risk_pcts[i]) / (sl_dist * config.CONTRACT_SIZES.get(asset, 100000) + 1e-8)
            pnl = (exit_price - entry_price) * vol * config.CONTRACT_SIZES.get(asset, 100000) if side == 'long' else (entry_price - exit_price) * vol * config.CONTRACT_SIZES.get(asset, 100000)

            self.equity[i] += pnl
            pnls[i] = pnl
            self.max_equity[i] = max(self.max_equity[i], self.equity[i])
            self.drawdown[i] = (self.max_equity[i] - self.equity[i]) / (self.max_equity[i] + 1e-8)

            reward = self.reward_engine.calculate_structural_reward(p_data['peak'][idx], p_data['valley'][idx], tp_dist, sl_dist)
            reward += self.reward_engine.calculate_trade_close_reward(pnl, config.INITIAL_EQUITY, self.drawdown[i], 0.0)

            if self.equity[i] <= 0.02 * config.INITIAL_EQUITY:
                reward += self.reward_engine.get_termination_penalty()
                terminated[i] = True

            rewards[i] = reward

        # Advance signals for all
        old_obs = self.get_observations()
        self.reset_envs(np.ones(self.num_envs, dtype=bool))
        next_obs = self.get_observations()

        return next_obs, rewards, terminated, old_obs, pnls, self.drawdown.copy()

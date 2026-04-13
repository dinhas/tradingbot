import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from .feature_engine import FeatureEngine, NUM_FEATURES

class TradingEnv(gym.Env):
    """
    Trading environment for RL agent in Alpha Layer.
    
    Curriculum Stages:
        Stage 1: Direction only (5 outputs)
        Stage 2: Direction + Position sizing (10 outputs)
        Stage 3: Direction + Position sizing + SL/TP (20 outputs)
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, data_dir='data', stage=3, is_training=True):
        super(TradingEnv, self).__init__()
        
        self.data_dir = data_dir
        self.stage = stage
        self.is_training = is_training
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        
        # Configuration Constants
        self.MAX_POS_SIZE_PCT = 0.50
        self.MAX_TOTAL_EXPOSURE = 0.60
        self.DRAWDOWN_LIMIT = 0.25
        self.MIN_POSITION_SIZE = 10  
        self.MIN_ATR_MULTIPLIER = 0.0001
        self.REWARD_LOG_INTERVAL = 5000
        self.enable_trailing_stop = False
        self.breakeven_trigger_r = 0.9
        self.breakeven_buffer_atr = 0.10
        self.trailing_trigger_r = 1.25
        self.trailing_atr_mult = 1.0

        # Load Data
        self.data = self._load_data()
        self.feature_engine = FeatureEngine()
        self.raw_data, self.processed_data = self.feature_engine.preprocess_data(self.data)
        
        # Cache for performance
        self._cache_data_arrays()
        
        # Master Obs Matrix for Combined Backtest Speed
        self._create_master_obs_matrix()
        
        # Define Action Space
        if self.stage == 1:
            self.action_dim = 5   
        elif self.stage == 2:
            self.action_dim = 10  
        else:
            self.action_dim = 20  
            
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(NUM_FEATURES,), dtype=np.float32)
        
        # State Variables
        self.current_step = 0
        self.max_steps = len(self.processed_data) - 1
        self.equity = 10000.0
        self.leverage = 100
        self.positions = {asset: None for asset in self.assets}
        
        self.completed_trades = []
        self.start_equity = self.equity
        self.peak_equity = self.equity

        # Spread Configuration
        self.SPREAD_PIPS = {
            "EURUSD": 1.2, "GBPUSD": 1.5, "USDJPY": 1.0, "USDCHF": 1.8, "XAUUSD": 45.0,
        }
        self.PIP_SIZE = {
            "EURUSD": 0.0001, "GBPUSD": 0.0001, "USDJPY": 0.01, "USDCHF": 0.0001, "XAUUSD": 0.1,
        }
        
    def _create_master_obs_matrix(self):
        """Creates a single matrix containing all asset observations for batch inference."""
        n_steps = len(self.processed_data)
        self.master_obs_matrix = np.zeros((n_steps, len(self.assets) * NUM_FEATURES), dtype=np.float32)
        
        # Use vectorized extraction for each asset
        for i, asset in enumerate(self.assets):
            obs_matrix = self.feature_engine.get_observation_vectorized(self.processed_data, asset)
            self.master_obs_matrix[:, i*NUM_FEATURES:(i+1)*NUM_FEATURES] = obs_matrix

    def _cache_data_arrays(self):
        self.open_arrays = {a: self.raw_data[f"{a}_open"].values.astype(np.float32) for a in self.assets}
        self.close_arrays = {a: self.raw_data[f"{a}_close"].values.astype(np.float32) for a in self.assets}
        self.low_arrays = {a: self.raw_data[f"{a}_low"].values.astype(np.float32) for a in self.assets}
        self.high_arrays = {a: self.raw_data[f"{a}_high"].values.astype(np.float32) for a in self.assets}
        self.atr_arrays = {a: self.raw_data[f"{a}_atr_14"].values.astype(np.float32) for a in self.assets}

    def _load_data(self):
        data = {}
        for asset in self.assets:
            # Check multiple possible paths
            paths = [
                f"{self.data_dir}/{asset}_5m.parquet",
                f"{self.data_dir}/{asset}_5m_2025.parquet",
                f"data/{asset}_5m.parquet"
            ]
            
            df = None
            for p in paths:
                try:
                    df = pd.read_parquet(p)
                    logging.info(f"Loaded {asset} from {p}")
                    break
                except FileNotFoundError:
                    continue
            
            if df is None:
                logging.error(f"Data for {asset} not found.")
                # Create dummy data to prevent crash
                dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
                df = pd.DataFrame(index=dates)
                for col in ['open', 'high', 'low', 'close']: df[f"{asset}_{col}"] = 1.0
                df[f"{asset}_volume"] = 100
                df[f"{asset}_atr_14"] = 0.001
            
            data[asset] = df
        return data

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.equity = 10000.0
        self.current_step = 500 if not self.is_training else np.random.randint(500, self.max_steps - 288)
        self.positions = {asset: None for asset in self.assets}
        self.completed_trades = []
        return self._get_observation(), {}

    def _get_observation(self):
        current_row = self.processed_data.iloc[self.current_step]
        # In multi-asset, we default to the first asset or a specific one set via helper
        asset = getattr(self, 'current_asset', self.assets[0])
        return self.feature_engine.get_observation(current_row, {}, asset)

    def set_asset(self, asset):
        self.current_asset = asset

    def _get_current_timestamp(self):
        return self.processed_data.index[self.current_step]

    def _execute_trades(self, actions):
        """Execute trading decisions for all assets."""
        current_prices = self._get_current_prices()
        atrs = self._get_current_atrs()
        
        for asset, act in actions.items():
            direction = act['direction']
            current_pos = self.positions[asset]
            price_mid = current_prices[asset]
            atr = atrs[asset]
            
            if current_pos is None:
                if direction != 0:
                    entry_price = self._to_executable_price(asset, price_mid, direction)
                    self._open_position(asset, direction, act, entry_price, atr)
            elif current_pos['direction'] == direction:
                pass
            elif direction != 0 and current_pos['direction'] != direction:
                close_price = self._to_executable_price(asset, price_mid, -current_pos['direction'])
                entry_price = self._to_executable_price(asset, price_mid, direction)
                self._close_position(asset, close_price)
                self._open_position(asset, direction, act, entry_price, atr)
            elif direction == 0 and current_pos is not None:
                close_price = self._to_executable_price(asset, price_mid, -current_pos['direction'])
                self._close_position(asset, close_price)

    def _to_executable_price(self, asset, mid_price, direction):
        spread = self.SPREAD_PIPS.get(asset, 2.0) * self.PIP_SIZE.get(asset, 0.0001)
        return mid_price + (direction * spread / 2.0)

    def _open_position(self, asset, direction, act, price, atr):
        size = act['size'] * self.MAX_POS_SIZE_PCT * self.equity
        # Handle zero ATR
        atr = max(atr, price * self.MIN_ATR_MULTIPLIER)
        sl_dist = act['sl_mult'] * atr
        tp_dist = act['tp_mult'] * atr
        sl = price - (direction * sl_dist)
        tp = price + (direction * tp_dist)
        
        self.positions[asset] = {
            'direction': direction,
            'entry_price': price,
            'size': size,
            'sl': sl,
            'tp': tp,
            'entry_step': self.current_step,
            'sl_dist': sl_dist,
            'tp_dist': tp_dist,
            'sl_mult': float(act.get('sl_mult', 0.0)),
            'tp_mult': float(act.get('tp_mult', 0.0)),
            'size_pct': float(act.get('size', 0.0)),
            'initial_risk_dist': sl_dist,
            'best_price': price,
            'trailing_active': False,
            'trailing_distance': 0.0,
            'current_stop_loss': sl,
        }
        # Entry fee
        self.equity -= size * 0.00002

    def _close_position(self, asset, price):
        pos = self.positions[asset]
        if pos is None: return
        
        # Leveraged P&L calculation
        price_change_pct = (price - pos['entry_price']) / pos['entry_price'] * pos['direction']
        pnl = price_change_pct * (pos['size'] * self.leverage)
        
        self.equity += pnl
        # Exit fee
        self.equity -= pos['size'] * 0.00002

        spread_price = self.SPREAD_PIPS.get(asset, 2.0) * self.PIP_SIZE.get(asset, 0.0001)
        hold_bars = max(1, int(self.current_step - pos['entry_step']))
        hold_time_minutes = hold_bars * 5
        planned_rr = (pos['tp_dist'] / (pos['sl_dist'] + 1e-12)) if pos['sl_dist'] > 0 else 0.0
        risk_notional = (pos['sl_dist'] / max(pos['entry_price'], 1e-9)) * (pos['size'] * self.leverage)
        fees_total = pos['size'] * 0.00004
        spread_cost_est = (spread_price / max(pos['entry_price'], 1e-9)) * (pos['size'] * self.leverage)
        net_pnl = pnl - fees_total

        self.completed_trades.append({
            'timestamp': self._get_current_timestamp(),
            'asset': asset,
            'pnl': pnl,
            'net_pnl': net_pnl,
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'size': pos['size'],
            'size_pct': pos.get('size_pct', 0.0),
            'direction': pos['direction'],
            'sl': pos['sl'],
            'tp': pos['tp'],
            'sl_mult': pos.get('sl_mult', 0.0),
            'tp_mult': pos.get('tp_mult', 0.0),
            'rr_ratio': planned_rr,
            'hold_time': hold_time_minutes,
            'hold_bars': hold_bars,
            'fees': fees_total,
            'spread_cost_est': spread_cost_est,
            'risk_amount': risk_notional,
            'realized_r': (net_pnl / risk_notional) if risk_notional > 1e-9 else 0.0,
        })
        self.positions[asset] = None

    def _get_current_prices(self):
        return {asset: self.close_arrays[asset][self.current_step] for asset in self.assets}

    def _get_current_atrs(self):
        return {asset: self.atr_arrays[asset][self.current_step] for asset in self.assets}

    def _update_positions(self):
        """Check SL/TP for all open positions using Bid/Ask and High/Low."""
        for asset, pos in list(self.positions.items()):
            if pos is None: continue

            o = self.open_arrays[asset][self.current_step]
            h = self.high_arrays[asset][self.current_step]
            l = self.low_arrays[asset][self.current_step]

            spread = self.SPREAD_PIPS.get(asset, 2.0) * self.PIP_SIZE.get(asset, 0.0001)
            half_spread = spread / 2.0
            atr = self.atr_arrays[asset][self.current_step]

            if self.enable_trailing_stop:
                self._apply_trailing_logic(pos, atr, o, h, l, half_spread)

            if pos['direction'] == 1:  # Long: Triggered by BID
                bid_o, bid_h, bid_l = o - half_spread, h - half_spread, l - half_spread
                if bid_l <= pos['current_stop_loss']: self._close_position(asset, pos['current_stop_loss'])
                elif bid_o >= pos['tp']: self._close_position(asset, bid_o)
                elif bid_h >= pos['tp']: self._close_position(asset, pos['tp'])
            else:  # Short: Triggered by ASK
                ask_o, ask_h, ask_l = o + half_spread, h + half_spread, l + half_spread
                if ask_h >= pos['current_stop_loss']: self._close_position(asset, pos['current_stop_loss'])
                elif ask_o <= pos['tp']: self._close_position(asset, ask_o)
                elif ask_l <= pos['tp']: self._close_position(asset, pos['tp'])

    def _apply_trailing_logic(self, pos, atr, o, h, l, half_spread):
        atr = max(float(atr), max(pos['entry_price'], 1e-9) * self.MIN_ATR_MULTIPLIER)
        initial_r = max(float(pos.get('initial_risk_dist', pos.get('sl_dist', 0.0))), 1e-9)
        direction = pos['direction']
        prev_sl = float(pos.get('current_stop_loss', pos.get('sl', pos['entry_price'])))

        if direction == 1:
            bid_h = h - half_spread
            pos['best_price'] = max(float(pos.get('best_price', pos['entry_price'])), bid_h)
            favorable = pos['best_price'] - pos['entry_price']

            if not pos.get('trailing_active', False):
                if favorable >= self.breakeven_trigger_r * initial_r:
                    be_stop = pos['entry_price'] + (self.breakeven_buffer_atr * atr)
                    prev_sl = max(prev_sl, be_stop)

                # ATR-based stop adjustment is allowed only before trailing activation
                if favorable >= self.trailing_trigger_r * initial_r:
                    activation_price = bid_h
                    trailing_distance = abs(activation_price - prev_sl)
                    pos['trailing_distance'] = max(trailing_distance, 1e-9)
                    pos['trailing_active'] = True

            if pos.get('trailing_active', False):
                new_sl = bid_h - pos['trailing_distance']
                prev_sl = max(prev_sl, new_sl)
        else:
            ask_l = l + half_spread
            pos['best_price'] = min(float(pos.get('best_price', pos['entry_price'])), ask_l)
            favorable = pos['entry_price'] - pos['best_price']

            if not pos.get('trailing_active', False):
                if favorable >= self.breakeven_trigger_r * initial_r:
                    be_stop = pos['entry_price'] - (self.breakeven_buffer_atr * atr)
                    prev_sl = min(prev_sl, be_stop)

                # ATR-based stop adjustment is allowed only before trailing activation
                if favorable >= self.trailing_trigger_r * initial_r:
                    activation_price = ask_l
                    trailing_distance = abs(activation_price - prev_sl)
                    pos['trailing_distance'] = max(trailing_distance, 1e-9)
                    pos['trailing_active'] = True

            if pos.get('trailing_active', False):
                new_sl = ask_l + pos['trailing_distance']
                prev_sl = min(prev_sl, new_sl)

        pos['current_stop_loss'] = prev_sl
        pos['sl'] = prev_sl

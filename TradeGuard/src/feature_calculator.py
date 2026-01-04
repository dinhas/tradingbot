import numpy as np
import pandas as pd
import ta
import logging

logger = logging.getLogger(__name__)

class TradeGuardFeatureCalculator:
    """
    Lean Feature Calculator (V2) - Reduced to ~20 features per asset + Global state.
    Designed for Multi-Asset Allow/Block classification.
    """
    
    def __init__(self, df_dict):
        self.df_dict = df_dict
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.precomputed_features = self._precompute_market_features(df_dict)
        
    def _precompute_market_features(self, df_dict):
        self.ohlcv_arrays = {}
        precomputed = {}
        for asset in self.assets:
            if asset not in df_dict: continue
            df = df_dict[asset]
            n = len(df)
            
            # Handle column names (prefixed vs standard)
            if 'high' in df.columns:
                h = df['high'].values
                l = df['low'].values
                c = df['close'].values
                v = df['volume'].values
                o = df['open'].values
            else:
                # Try prefixed
                try:
                    h = df[f"{asset}_high"].values
                    l = df[f"{asset}_low"].values
                    c = df[f"{asset}_close"].values
                    v = df[f"{asset}_volume"].values
                    o = df[f"{asset}_open"].values
                except KeyError:
                    logger.error(f"Could not find OHLCV columns for {asset}. Columns: {df.columns}")
                    continue
            
            self.ohlcv_arrays[asset] = {'high': h, 'low': l, 'close': c}
            
            # 1. Volatility / Base Indicators
            tr = np.zeros(n)
            tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
            atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
            atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().values
            
            # ========================================
            # NORMALIZED FEATURES (all scaled to reasonable ranges)
            # ========================================
            
            # f1: Relative Volume - clip to [0, 5], then scale to [0, 1]
            rel_vol = v / (pd.Series(v).rolling(50).mean() + 1e-6)
            f1 = np.clip(rel_vol, 0, 5) / 5.0
            
            # f2: Volatility Ratio (ATR/ATR_MA) - clip to [0.2, 3], scale to [0, 1]
            vol_ratio = atr / (atr_ma + 1e-6)
            f2 = np.clip((vol_ratio - 0.2) / 2.8, 0, 1)
            
            # f3: ADX Approximation - already 0-100, scale to [0, 1]
            dm_p = np.zeros(n); dm_m = np.zeros(n)
            dm_p[1:] = np.maximum(h[1:] - h[:-1], 0)
            dm_m[1:] = np.maximum(l[:-1] - l[1:], 0)
            di_p = pd.Series(dm_p).rolling(14).mean() / (atr + 1e-6) * 100
            di_m = pd.Series(dm_m).rolling(14).mean() / (atr + 1e-6) * 100
            adx_raw = abs(di_p - di_m) / (di_p + di_m + 1e-6) * 100
            f3 = np.clip(adx_raw, 0, 100) / 100.0
            
            # f4: Efficiency Ratio - already 0-1
            net_chg = np.abs(pd.Series(c).diff(20))
            abs_sum = pd.Series(c).diff().abs().rolling(20).sum()
            f4 = np.clip(net_chg / (abs_sum + 1e-6), 0, 1)
            
            # f5: Distance from MA200 - clip to [-0.2, 0.2], scale to [-1, 1]
            ma200 = pd.Series(c).rolling(200, min_periods=1).mean()
            dist_ma = (c - ma200) / (ma200 + 1e-6)
            f5 = np.clip(dist_ma / 0.2, -1, 1)
            
            # f6: Bollinger Width - clip to [0, 0.1], scale to [0, 1]
            std20 = pd.Series(c).rolling(20).std()
            bb_width = (std20 * 4) / (pd.Series(c).rolling(20).mean() + 1e-6)
            f6 = np.clip(bb_width / 0.1, 0, 1)
            
            # f7: RSI - scale from [0, 100] to [-1, 1]
            c_series = pd.Series(c)
            rsi_raw = ta.momentum.RSIIndicator(c_series, window=14).rsi().values
            f7 = (np.clip(rsi_raw, 0, 100) - 50) / 50.0
            
            # f8: Bollinger Position - already ~0-1, clip for safety
            bb = ta.volatility.BollingerBands(c_series)
            bb_pos = (c - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-6)
            f8 = np.clip(bb_pos, 0, 1)
            
            # f9: MACD Diff - normalize by ATR (asset-agnostic)
            macd_raw = ta.trend.MACD(c_series).macd_diff().values
            f9 = np.clip(macd_raw / (atr + 1e-6), -3, 3) / 3.0
            
            # f10: EMA Alignment (9 vs 21) - clip to [-0.05, 0.05], scale to [-1, 1]
            ema9 = pd.Series(c).ewm(span=9).mean()
            ema21 = pd.Series(c).ewm(span=21).mean()
            ema_align = (ema9 - ema21) / (ema21 + 1e-6)
            f10 = np.clip(ema_align / 0.05, -1, 1)
            
            # f11: Body Ratio - already 0-1
            body = np.abs(c - o)
            range_val = h - l + 1e-8  # Avoid div by zero
            f11 = np.clip(body / range_val, 0, 1)
            
            # f12: Upper Wick Ratio - clip to [0, 5], scale to [0, 1]
            upper_wick = (h - np.maximum(o, c)) / (body + 1e-6)
            f12 = np.clip(upper_wick, 0, 5) / 5.0
            
            # f13: Lower Wick Ratio - clip to [0, 5], scale to [0, 1]
            lower_wick = (np.minimum(o, c) - l) / (body + 1e-6)
            f13 = np.clip(lower_wick, 0, 5) / 5.0
            
            # f14: Velocity (price change / ATR) - clip to [-5, 5], scale to [-1, 1]
            velocity = np.zeros(n)
            velocity[5:] = (c[5:] - c[:-5]) / (atr[5:] + 1e-6)
            f14 = np.clip(velocity, -5, 5) / 5.0
            
            # f15: Spread/Range proxy - normalize by close, clip to [0, 0.02], scale to [0, 1]
            spread_proxy = (h - l) / (c + 1e-6)
            f15 = np.clip(spread_proxy / 0.02, 0, 1)
            
            # Stack all 15 normalized features
            combined = np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15])
            precomputed[asset] = np.nan_to_num(combined, nan=0.0).astype(np.float32)
            
        return precomputed

    def get_multi_asset_obs(self, step, trade_infos, portfolio_state):
        """
        Constructs a single large vector for ALL 5 assets.
        Each asset: 20 features. Global: 5 features.
        Total: 105 features.
        
        ALL FEATURES ARE NORMALIZED to [-1, 1] or [0, 1] ranges.
        """
        full_obs = []
        
        for asset in self.assets:
            if asset not in self.precomputed_features:
                full_obs.extend([0] * 20)
                continue
                
            # 1. Market Precomputed (15 features) - already normalized
            f_market = self.precomputed_features[asset][step].tolist()
            
            # 2. Alpha Context (3 features) - NORMALIZED
            p_state = portfolio_state.get(asset, {})
            action_raw = p_state.get('action_raw', 0)  # Already -1, 0, 1
            signal_persistence = p_state.get('signal_persistence', 0)
            signal_reversal = p_state.get('signal_reversal', 0)  # Already 0 or 1
            
            f_alpha = [
                action_raw,  # Already in [-1, 1]
                np.clip(signal_persistence / 10.0, 0, 1),  # Normalize: 10+ bars = 1.0
                signal_reversal  # Already 0 or 1
            ]
            
            # 3. Execution (2 features) - NORMALIZED
            t_info = trade_infos.get(asset, {'entry': 0, 'sl': 0, 'tp': 0})
            sl_dist = abs(t_info['entry'] - t_info['sl'])
            tp_dist = abs(t_info['entry'] - t_info['tp'])
            
            # Use proper ATR for normalization (high - low is just 1-bar range, not ATR)
            # For better normalization, we use close price to make it asset-agnostic
            if asset in self.ohlcv_arrays and 'close' in self.ohlcv_arrays[asset]:
                close_price = self.ohlcv_arrays[asset]['close'][step]
            else:
                close_price = t_info['entry'] if t_info['entry'] > 0 else 1.0
            
            # SL as percentage of price, typical range 0.1% - 2% -> scale to [0, 1]
            sl_pct = (sl_dist / (close_price + 1e-8)) * 100  # Convert to percentage
            sl_normalized = np.clip(sl_pct / 2.0, 0, 1)  # 2% SL = 1.0
            
            # Risk:Reward ratio, typical range 0.5 - 4 -> scale to [0, 1]
            rr_ratio = tp_dist / (sl_dist + 1e-8)
            rr_normalized = np.clip(rr_ratio / 4.0, 0, 1)  # 4:1 RR = 1.0
            
            f_exec = [
                sl_normalized,
                rr_normalized
            ]
            
            # Total 15 + 3 + 2 = 20 per asset
            full_obs.extend(f_market + f_alpha + f_exec)
            
        # 4. Global Features (5 features) - already well-scaled
        first_asset = self.assets[0]
        if first_asset in self.df_dict:
            dt = self.df_dict[first_asset].index[step]
            hour = dt.hour
        else:
            hour = 0
            
        f_global = [
            np.sin(2 * np.pi * hour / 24),  # Already [-1, 1]
            np.cos(2 * np.pi * hour / 24),  # Already [-1, 1]
            1.0 if (13 <= hour <= 17) else 0.0,  # Already [0, 1]
            np.clip(portfolio_state.get('total_drawdown', 0), 0, 1),  # Clip to [0, 1]
            np.clip(portfolio_state.get('total_exposure', 0) / 5.0, 0, 1)  # 5x leverage = 1.0
        ]
        full_obs.extend(f_global)
        
        return np.array(full_obs, dtype=np.float32)
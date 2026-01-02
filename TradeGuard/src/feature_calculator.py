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
            
            self.ohlcv_arrays[asset] = {'high': h, 'low': l}
            
            # 1. Volatility / Base Indicators
            tr = np.zeros(n)
            tr[1:] = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
            atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
            atr_ma = pd.Series(atr).rolling(50, min_periods=1).mean().values
            
            # 2. Market State (10 Features)
            f1 = v / (pd.Series(v).rolling(50).mean() + 1e-6) # Rel Vol
            f2 = atr / (atr_ma + 1e-6) # Vol Ratio
            
            # ADX Approx
            dm_p = np.zeros(n); dm_m = np.zeros(n)
            dm_p[1:] = np.maximum(h[1:] - h[:-1], 0)
            dm_m[1:] = np.maximum(l[:-1] - l[1:], 0)
            di_p = pd.Series(dm_p).rolling(14).mean() / (atr + 1e-6) * 100
            di_m = pd.Series(dm_m).rolling(14).mean() / (atr + 1e-6) * 100
            # ADX: scale to [0, 1] instead of [0, 100]
            f3 = abs(di_p - di_m) / (di_p + di_m + 1e-6)  # ADX in [0, 1]
            
            # Efficiency / Hurst
            net_chg = np.abs(pd.Series(c).diff(20))
            abs_sum = pd.Series(c).diff().abs().rolling(20).sum()
            f4 = net_chg / (abs_sum + 1e-6) # Efficiency Ratio
            
            ma200 = pd.Series(c).rolling(200, min_periods=1).mean()
            f5 = (c - ma200) / (ma200 + 1e-6) # Dist MA200
            
            std20 = pd.Series(c).rolling(20).std()
            f6 = (std20 * 4) / (pd.Series(c).rolling(20).mean() + 1e-6) # BB Width
            
            # 3. Candle / Momentum (10 Features)
            c_series = pd.Series(c)
            # RSI: scale from [0, 100] to [-1, 1]
            rsi_raw = ta.momentum.RSIIndicator(c_series, window=14).rsi().values
            f7 = (rsi_raw - 50) / 50  # Now in [-1, 1]
            bb = ta.volatility.BollingerBands(c_series)
            f8 = (c - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-6) # BB Pos
            # MACD: normalize by ATR to make scale-invariant
            macd_raw = ta.trend.MACD(c_series).macd_diff().values
            f9 = macd_raw / (atr + 1e-6)  # Normalized MACD
            
            ema9 = pd.Series(c).ewm(span=9).mean()
            ema21 = pd.Series(c).ewm(span=21).mean()
            f10 = (ema9 - ema21) / (ema21 + 1e-6) # EMA Align
            
            body = np.abs(c - o)
            range_val = h - l
            f11 = body / (range_val + 1e-6)  # Body Ratio [0, 1]
            # Clip wick ratios to prevent explosion when body is tiny
            f12 = np.clip((h - np.maximum(o, c)) / (body + 1e-6), 0, 5)  # Upper Wick
            f13 = np.clip((np.minimum(o, c) - l) / (body + 1e-6), 0, 5)  # Lower Wick
            # Velocity: clip to reasonable range
            f14 = np.zeros(n)
            f14[5:] = np.clip((c[5:] - c[:-5]) / (atr[5:] + 1e-6), -5, 5)  # Velocity
            f15 = (h - l) / (c + 1e-6)  # Spread/Cost Proxy
            
            # We combine 15 market features here. 
            # The other 5 (Alpha Context + Execution) are added in real-time.
            combined = np.column_stack([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15])
            precomputed[asset] = np.nan_to_num(combined, nan=0.0).astype(np.float32)
            
        return precomputed

    def get_single_asset_obs(self, asset, step, trade_info, asset_portfolio_state, global_state):
        """
        Constructs a single vector for ONE asset decision.
        15 market + 3 alpha + 7 global/risk = 25 features.
        """
        if asset not in self.precomputed_features:
            return np.zeros(25, dtype=np.float32)

        # 1. Market Precomputed (15 features)
        f_market = self.precomputed_features[asset][step].tolist()

        # 2. Alpha Context (3 features)
        # Normalize signal_persistence to [0, 1] (max ~20 steps)
        persistence = asset_portfolio_state.get('signal_persistence', 0)
        f_alpha = [
            asset_portfolio_state.get('action_raw', 0),  # Already in {-1, 0, 1}
            min(persistence / 10.0, 1.0),  # Normalize to [0, 1]
            asset_portfolio_state.get('signal_reversal', 0)  # Already binary
        ]

        # 3. Execution & Risk Context (3 features)
        sl_dist = abs(trade_info['entry'] - trade_info['sl'])
        tp_dist = abs(trade_info['entry'] - trade_info['tp'])
        atr_val = (self.ohlcv_arrays[asset]['high'][step] - self.ohlcv_arrays[asset]['low'][step]) + 1e-6

        # Clip risk features to reasonable ranges
        f_risk = [
            np.clip(sl_dist / atr_val, 0, 5),  # SL distance in ATR units
            np.clip(tp_dist / (sl_dist + 1e-6), 0, 10),  # Risk Reward ratio
            trade_info.get('risk_val', 0.5)  # Risk Model Conviction [0, 1]
        ]

        # 4. Global State (4 features)
        # We derive time from the index of the first available asset
        dt = self.df_dict[self.assets[0]].index[step]
        f_global = [
            np.sin(2 * np.pi * dt.hour / 24),
            np.cos(2 * np.pi * dt.hour / 24),
            global_state.get('total_drawdown', 0),
            global_state.get('total_exposure', 0)
        ]

        return np.array(f_market + f_alpha + f_risk + f_global, dtype=np.float32)

    def get_multi_asset_obs(self, step, trade_infos, portfolio_state):
        """
        [DEPRECATED] Constructs a single large vector for ALL 5 assets.
        Each asset: 20 features. Global: 5 features. Total: 105 features.
        """
        full_obs = []
        
        for asset in self.assets:
            if asset not in self.precomputed_features:
                full_obs.extend([0] * 20)
                continue
                
            # Use the new single asset logic
            f_asset = self.get_single_asset_obs(
                asset, 
                step, 
                trade_infos.get(asset, {'entry':0, 'sl':0, 'tp':0}), 
                portfolio_state.get(asset, {})
            )
            full_obs.extend(f_asset.tolist())
            
        # 4. Global Features (5 features)
        dt = self.df_dict[self.assets[0]].index[step]
        f_global = [
            np.sin(2 * np.pi * dt.hour / 24),
            np.cos(2 * np.pi * dt.hour / 24),
            1.0 if (13 <= dt.hour <= 17) else 0.0, # NY/London Overlap
            portfolio_state.get('total_drawdown', 0),
            portfolio_state.get('total_exposure', 0)
        ]
        full_obs.extend(f_global)
        
        return np.array(full_obs, dtype=np.float32)

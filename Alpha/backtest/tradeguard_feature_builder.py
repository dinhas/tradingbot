
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)

class TradeGuardFeatureBuilder:
    """
    Handles feature engineering for TradeGuard inference, ensuring parity with training.
    """
    def __init__(self, df_dict):
        self.df_dict = df_dict
        self.assets = list(df_dict.keys())
        self.precomputed_market_features = self._precompute_market_features(df_dict)
        
        # Cache ATR arrays for Group E features
        self.atr_arrays = {}
        for asset, df in df_dict.items():
            self.atr_arrays[asset] = self._calculate_atr(df).values

    def _calculate_atr(self, df, window=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr = np.maximum(high - low, 
                       np.maximum(np.abs(high - close.shift(1)),
                                np.abs(low - close.shift(1))))
        atr = tr.rolling(window).mean()
        return atr.bfill().fillna(0.0001)

    def _precompute_market_features(self, df_dict):
        """
        Adapts precompute_market_features from generate_dataset.py for parity.
        Computes Groups B, C, D, F (40 features).
        """
        precomputed = {}
        start_time = time.time()
        
        for asset, df in df_dict.items():
            n = len(df)
            high = df['high'].values.astype(np.float64)
            low = df['low'].values.astype(np.float64)
            close = df['close'].values.astype(np.float64)
            open_p = df['open'].values.astype(np.float64)
            vol = df['volume'].values.astype(np.float64)
            
            range_val = high - low
            body = np.abs(close - open_p)
            
            # Fast ATR
            tr = np.zeros(n)
            tr[1:] = np.maximum(high[1:] - low[1:], 
                               np.maximum(np.abs(high[1:] - close[:-1]),
                                         np.abs(low[1:] - close[:-1])))
            atr_14 = pd.Series(tr).rolling(14, min_periods=1).mean().values
            atr_14 = np.nan_to_num(atr_14, nan=0.0001)

            def fast_rolling_mean(arr, window):
                result = np.zeros(len(arr))
                cumsum = np.cumsum(arr)
                result[window-1:] = (cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])) / window
                result[:window-1] = result[window-1]
                return result
            
            def fast_rolling_std(arr, window):
                mean = fast_rolling_mean(arr, window)
                sq_mean = fast_rolling_mean(arr**2, window)
                result = np.sqrt(np.maximum(sq_mean - mean**2, 0))
                return result

            vol_avg_50 = fast_rolling_mean(vol, 50)
            vol_std_50 = fast_rolling_std(vol, 50)
            
            # Group B: News Proxies (10 features)
            f11 = vol / (vol_avg_50 + 1e-6)
            f12 = (vol - vol_avg_50) / (vol_std_50 + 1e-6)
            f13 = range_val / (atr_14 + 1e-6)
            f14 = np.zeros(n)  # Placeholder
            f15 = body / (range_val + 1e-6)
            upper_wick = high - np.maximum(open_p, close)
            lower_wick = np.minimum(open_p, close) - low
            f16 = (upper_wick + lower_wick) / (body + 1e-6)
            f17 = np.zeros(n)
            f17[1:] = np.abs(open_p[1:] - close[:-1]) / (atr_14[1:] + 1e-6)
            vol_diff = np.zeros(n)
            vol_diff[1:] = vol[1:] - vol[:-1]
            vol_ma_10 = fast_rolling_mean(vol, 10)
            f18 = vol_diff / (vol_ma_10 + 1e-6)
            f19 = atr_14 / (close + 1e-6)
            f20 = np.zeros(n)

            # Group C: Market Regime (10 features)
            dm_plus = np.zeros(n); dm_minus = np.zeros(n)
            dm_plus[1:] = np.maximum(high[1:] - high[:-1], 0)
            dm_minus[1:] = np.maximum(low[:-1] - low[1:], 0)
            di_plus = fast_rolling_mean(dm_plus, 14) / (atr_14 + 1e-6) * 100
            di_minus = fast_rolling_mean(dm_minus, 14) / (atr_14 + 1e-6) * 100
            dx = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-6) * 100
            f21 = fast_rolling_mean(dx, 14)
            f22 = di_plus
            f23 = di_minus
            rolling_high_25 = pd.Series(high).rolling(25, min_periods=1).max().values
            rolling_low_25 = pd.Series(low).rolling(25, min_periods=1).min().values
            f24 = (high - rolling_low_25) / (rolling_high_25 - rolling_low_25 + 1e-6) * 100
            f25 = (rolling_high_25 - low) / (rolling_high_25 - rolling_low_25 + 1e-6) * 100
            f26 = np.full(n, 0.5)
            net_change_10 = np.zeros(n); net_change_10[10:] = np.abs(close[10:] - close[:-10])
            close_diff = np.zeros(n); close_diff[1:] = np.abs(close[1:] - close[:-1])
            abs_sum_10 = fast_rolling_mean(close_diff, 10) * 10
            f27 = net_change_10 / (abs_sum_10 + 1e-6)
            f28 = np.zeros(n); f28[20:] = (close[20:] - close[:-20]) / 20 / (close[20:] + 1e-6) * 10000
            sma_200 = fast_rolling_mean(close, min(200, n))
            f29 = close / (sma_200 + 1e-6)
            sma_20 = fast_rolling_mean(close, 20); std_20 = fast_rolling_std(close, 20)
            f30 = (std_20 * 4) / (sma_20 + 1e-6)

            # Group D: Session Edge (10 features)
            hours = df.index.hour.values
            dows = df.index.dayofweek.values
            f31 = np.sin(2 * np.pi * hours / 24)
            f32 = np.cos(2 * np.pi * hours / 24)
            f33 = np.sin(2 * np.pi * dows / 7)
            f34 = np.cos(2 * np.pi * dows / 7)
            f35 = ((hours >= 7) & (hours < 16)).astype(np.float64)
            f36 = ((hours >= 12) & (hours < 21)).astype(np.float64)
            f37 = ((hours >= 0) & (hours < 9)).astype(np.float64)
            f38 = df.index.minute.values / 60.0
            f39 = (f35 * f36)
            f40 = (hours * 60 + df.index.minute.values) / 1440.0

            # Group F: Price Action (10 features)
            f51 = (close > open_p).astype(np.float64)
            f52 = body / (atr_14 + 1e-6)
            f53 = upper_wick / (body + 1e-6)
            f54 = lower_wick / (body + 1e-6)
            f55 = np.full(n, 5.0)
            rolling_high_20 = pd.Series(high).rolling(20, min_periods=1).max().values
            rolling_low_20 = pd.Series(low).rolling(20, min_periods=1).min().values
            f56 = (rolling_high_20 - close) / (atr_14 + 1e-6)
            f57 = (close - rolling_low_20) / (atr_14 + 1e-6)
            f58 = np.zeros(n); f59 = np.zeros(n); f60 = np.zeros(n)

            combined = np.column_stack([
                f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,  # B
                f21, f22, f23, f24, f25, f26, f27, f28, f29, f30,  # C
                f31, f32, f33, f34, f35, f36, f37, f38, f39, f40,  # D
                f51, f52, f53, f54, f55, f56, f57, f58, f59, f60   # F
            ])
            precomputed[asset] = np.nan_to_num(combined, nan=0.0).astype(np.float32)
            
        logger.info(f"Precomputed TradeGuard features for {len(df_dict)} assets in {time.time() - start_time:.2f}s")
        return precomputed

    def build_features(self, asset, step, trade_info, portfolio_state):
        """
        Builds the full 60-feature vector for a given step.
        Order: Group A, Group B, Group C, Group E, Group D, Group F
        """
        # Group A: Alpha Confidence (10 features)
        recent_trades = portfolio_state['recent_trades']
        win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades) if recent_trades else 0.5
        pnl_sum = sum(t['pnl'] for t in recent_trades) if recent_trades else 0.0
        
        f_a = [
            portfolio_state['asset_action_raw'],
            abs(portfolio_state['asset_action_raw']),
            np.std(portfolio_state['asset_recent_actions']),
            portfolio_state['asset_signal_persistence'],
            portfolio_state['asset_signal_reversal'],
            1 - (portfolio_state['equity'] / portfolio_state['peak_equity']) if portfolio_state['peak_equity'] > 0 else 0,
            portfolio_state['open_positions_count'],
            portfolio_state['total_exposure'] / portfolio_state['equity'] if portfolio_state['equity'] > 0 else 0,
            win_rate,
            pnl_sum
        ]
        
        # Group B & C (20 features from precomputed)
        f_market = self.precomputed_market_features[asset][step]
        f_bc = f_market[:20].tolist()
        
        # Group E: Execution Stats (10 features)
        atr_val = self.atr_arrays[asset][step]
        sl_dist = abs(trade_info['entry_price'] - trade_info['sl'])
        tp_dist = abs(trade_info['entry_price'] - trade_info['tp'])
        
        f_e = [
            0.0, # entry_dist (relative to last close, usually 0 at bar start)
            sl_dist / (atr_val + 1e-6),
            tp_dist / (atr_val + 1e-6),
            tp_dist / (sl_dist + 1e-6),
            portfolio_state.get('position_value', 0) / portfolio_state['equity'] if portfolio_state['equity'] > 0 else 0,
            1 - (portfolio_state['equity'] / portfolio_state['peak_equity']) if portfolio_state['peak_equity'] > 0 else 0,
            0.0, 0.5, 0.5, 0.0 # Placeholders
        ]
        
        # Group D & F (20 features from precomputed)
        f_df = f_market[20:].tolist()
        
        # Combine in exact order: A(10), B(10), C(10), E(10), D(10), F(10)
        # Wait, the code in generate_dataset was: f_a + f_market[:30] + f_e + f_market[30:]
        # Which is A(10), B(10), C(10), D(10), E(10), F(10)
        # Let's re-verify generate_dataset.py logic:
        # all_features = f_a + f_market[:30] + f_e + f_market[30:]
        # f_market has 40 features: B(10), C(10), D(10), F(10)
        # So f_market[:30] is B, C, D.
        # So it is A, B, C, D, E, F. 
        # Total = 10 (A) + 30 (B,C,D) + 10 (E) + 10 (F) = 60.
        
        return f_a + f_market[:30].tolist() + f_e + f_market[30:].tolist()

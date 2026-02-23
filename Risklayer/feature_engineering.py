import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from typing import Dict, List, Tuple
from .config import config

class FeatureEngineer:
    def __init__(self):
        self.assets = config.ASSETS
        self.alpha_feature_cols = [
            'ret_1', 'ret_12', 'ret_24', 'bb_pos', 'squeeze',
            'ema9_dist', 'ema21_dist', 'ema_align', 'rsi', 'macd_hist',
            'rsi_slope', 'macd_mom', 'vol_ratio', 'vol_shock', 'vwap_dist',
            'pressure', 'wick_rejection', 'breakout_vel', 'swing_prox', 'hour_sin',
            'hour_cos', 'day_sin', 'day_cos', 'session_asian', 'session_london',
            'session_ny', 'session_overlap', 'ret_240', 'mkt_vol', 'ret_120'
        ]

    def calculate_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculates features for each asset."""
        processed_dict = {}

        # Calculate market-wide features first
        market_volatility = self._calculate_market_volatility(data_dict)

        for asset, df in data_dict.items():
            df = df.copy()

            # 1. Base technicals
            df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            df['atr_ma'] = df['atr'].rolling(window=20).mean()
            df['atr_ratio'] = df['atr'] / (df['atr_ma'] + 1e-9)

            # Volatility Percentile (rolling 1000 bars)
            df['vol_percentile'] = df['atr_ratio'].rolling(1000).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)

            # 30 Alpha Features
            # Returns
            df['ret_1'] = df['close'].pct_change(1)
            df['ret_12'] = df['close'].pct_change(12)
            df['ret_24'] = df['close'].pct_change(24)

            # Volatility
            bb = BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_pos'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
            df['squeeze'] = (bb.bollinger_hband() - bb.bollinger_lband()) / (df['atr'] + 1e-9)

            # Trend
            ema9 = EMAIndicator(df['close'], window=9).ema_indicator()
            ema21 = EMAIndicator(df['close'], window=21).ema_indicator()
            df['ema9_dist'] = (df['close'] - ema9) / (df['atr'] + 1e-9)
            df['ema21_dist'] = (df['close'] - ema21) / (df['atr'] + 1e-9)
            df['ema_align'] = (ema9 - ema21) / (df['atr'] + 1e-9)

            # Momentum
            df['rsi'] = RSIIndicator(df['close'], window=14).rsi() / 100.0
            macd = MACD(df['close'])
            df['macd_hist'] = macd.macd_diff() / (df['atr'] + 1e-9)
            df['rsi_slope'] = df['rsi'].diff(5)
            df['macd_mom'] = df['macd_hist'].diff(5)

            # Volume
            vol_ma = df['volume'].rolling(window=20).mean()
            df['vol_ratio'] = df['volume'] / (vol_ma + 1e-9)
            df['vol_shock'] = np.log(df['vol_ratio'] + 1e-9)

            # Price Action
            df['vwap'] = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9) # Simple cumulative VWAP
            df['vwap_dist'] = (df['close'] - df['vwap']) / (df['atr'] + 1e-9)

            direction = np.sign(df['close'] - df['open'])
            df['pressure'] = (df['volume'] * direction).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-9)

            body = (df['close'] - df['open']).abs()
            upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
            lower_wick = df[['open', 'close']].min(axis=1) - df['low']
            df['wick_rejection'] = (upper_wick + lower_wick) / (body + 1e-9)

            range_h = df['high'].rolling(20).max()
            range_l = df['low'].rolling(20).min()
            df['breakout_vel'] = np.where(df['close'] > range_h.shift(1), (df['close'] - range_h.shift(1))/(df['atr']+1e-9),
                                         np.where(df['close'] < range_l.shift(1), (df['close'] - range_l.shift(1))/(df['atr']+1e-9), 0))

            swing_high = df['high'].rolling(50).max()
            swing_low = df['low'].rolling(50).min()
            df['swing_prox'] = np.minimum((swing_high - df['close']), (df['close'] - swing_low)) / (df['atr'] + 1e-9)

            # Time & Session
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

            hours = df.index.hour
            df['session_asian'] = ((hours >= 0) & (hours < 9)).astype(float)
            df['session_london'] = ((hours >= 8) & (hours < 17)).astype(float)
            df['session_ny'] = ((hours >= 13) & (hours < 22)).astype(float)
            df['session_overlap'] = ((hours >= 13) & (hours < 17)).astype(float)

            # Close (log normalized or similar) - let's just use return from a long window
            df['ret_240'] = df['close'].pct_change(240)

            # Market Volatility
            df['mkt_vol'] = market_volatility.reindex(df.index).ffill()

            # 30 Features list:
            # 1. ret_1, 2. ret_12, 3. ret_24, 4. bb_pos, 5. squeeze,
            # 6. ema9_dist, 7. ema21_dist, 8. ema_align, 9. rsi, 10. macd_hist,
            # 11. rsi_slope, 12. macd_mom, 13. vol_ratio, 14. vol_shock, 15. vwap_dist,
            # 16. pressure, 17. wick_rejection, 18. breakout_vel, 19. swing_prox, 20. hour_sin,
            # 21. hour_cos, 22. day_sin, 23. day_cos, 24. session_asian, 25. session_london,
            # 26. session_ny, 27. session_overlap, 28. ret_240, 29. mkt_vol, 30. close_norm (let's use ret_120)
            df['ret_120'] = df['close'].pct_change(120)

            # Gating Scores (Simulated or loaded - here we simulate for RL training if not present)
            # In a real scenario, these would come from the Alpha model.
            if 'meta_score' not in df.columns:
                # Dummy scores for training purposes
                df['meta_score'] = 0.5 + 0.1 * np.random.randn(len(df))
            if 'quality_score' not in df.columns:
                df['quality_score'] = 0.3 + 0.1 * np.random.randn(len(df))

            df.dropna(inplace=True)
            processed_dict[asset] = df

        return processed_dict

    def _calculate_market_volatility(self, data_dict: Dict[str, pd.DataFrame]) -> pd.Series:
        atr_ratios = []
        for asset, df in data_dict.items():
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            atr_ma = atr.rolling(window=20).mean()
            atr_ratios.append(atr / (atr_ma + 1e-9))

        return pd.concat(atr_ratios, axis=1).mean(axis=1)

    def get_observation_cols(self) -> List[str]:
        return self.alpha_feature_cols + ['atr', 'vol_percentile']

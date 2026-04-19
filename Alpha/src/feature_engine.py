import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

class FeatureEngine:
    def __init__(self):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.feature_names = []
        self._define_feature_names()

    def _kalman_filter(self, data):
        """
        1D Kalman Filter for real-time price denoising.
        State: True Price. Measurement: Observed OHLC Price.
        No lookahead bias.
        """
        # Parameters: Q (Process Variance), R (Measurement Variance)
        Q = 1e-5 
        R = 1e-3
        
        xhat = data[0] # Initial guess
        P = 1.0        # Initial covariance
        filtered = []
        
        for z in data:
            # Prediction Step
            P = P + Q
            # Update Step
            K = P / (P + R) # Kalman Gain
            xhat = xhat + K * (z - xhat)
            P = (1 - K) * P
            filtered.append(xhat)
            
        return np.array(filtered, dtype=np.float32)

    def _define_feature_names(self):
        """Defines the list of 17 features based on user requirements."""
        self.feature_names = [
            # --- RAW OHLCV (5) ---
            "open", "high", "low", "close", "volume",
            
            # --- NORMALIZED (3) ---
            "log_return", "rolling_vol", "rolling_mean_ret",
            
            # --- TECHNICAL (5) ---
            "rsi", "macd_hist", "bollinger_pB", "atr", "adx",
            
            # --- TIME-FLAGS (3) ---
            "hour_of_day", "is_late_session", "is_friday",

            # --- NEW ADDITIONS TO REACH 17 (2) ---
            "volume_ratio", "relative_spread"
        ]

    def preprocess_data(self, data_dict):
        logger = logging.getLogger(__name__)
        aligned_df = self._align_data(data_dict)
        
        for col in aligned_df.columns:
            aligned_df[col] = aligned_df[col].astype(np.float32)
        
        all_new_cols = {}
        for asset in self.assets:
            logger.info(f"Calculating features for {asset} (with Kalman Filtering)...")
            asset_cols = self._get_asset_features(aligned_df, asset)
            all_new_cols.update(asset_cols)
            
        logger.info("Adding time features...")
        time_cols = self._get_time_features(aligned_df)
        all_new_cols.update(time_cols)

        new_features_df = pd.DataFrame(all_new_cols, index=aligned_df.index).astype(np.float32)
        aligned_df = pd.concat([aligned_df, new_features_df], axis=1)
        
        logger.info("Normalizing features...")
        normalized_df = aligned_df.copy()
        normalized_df = self._normalize_features(normalized_df)
        
        normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)
        aligned_df = aligned_df.ffill().fillna(0).astype(np.float32)
        
        return aligned_df, normalized_df

    def _align_data(self, data_dict):
        common_index = None
        for df in data_dict.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        aligned_parts = []
        for asset, df in data_dict.items():
            df_subset = df.loc[common_index].copy()
            df_subset.columns = [f"{asset}_{col}" for col in df_subset.columns]
            aligned_parts.append(df_subset)
            
        return pd.concat(aligned_parts, axis=1)

    def _get_asset_features(self, df, asset):
        raw_close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        volume = df[f"{asset}_volume"]
        
        # Apply Kalman Denoising to the close price (Causal, no lookahead)
        close_filled = raw_close.ffill().bfill().to_numpy(copy=True)
        close_denoised = self._kalman_filter(close_filled)
        close = pd.Series(close_denoised, index=raw_close.index)
        
        new_cols = {}
        # Use Kalman-denoised close for log returns and volatility
        log_ret = np.log(close / (close.shift(1) + 1e-8))
        new_cols[f"{asset}_log_return"] = log_ret
        new_cols[f"{asset}_rolling_vol"] = log_ret.rolling(window=20).std()
        new_cols[f"{asset}_rolling_mean_ret"] = log_ret.rolling(window=20).mean()
        
        # Technical indicators use the Kalman-denoised close for stability
        new_cols[f"{asset}_rsi"] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        new_cols[f"{asset}_macd_hist"] = macd.macd_diff()
        
        bb = BollingerBands(close, window=20, window_dev=2)
        new_cols[f"{asset}_bollinger_pB"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
        
        # ATR and ADX use raw prices to capture real market range
        new_cols[f"{asset}_atr"] = AverageTrueRange(high, low, raw_close, window=14).average_true_range()
        new_cols[f"{asset}_adx"] = ADXIndicator(high, low, raw_close, window=14).adx()

        # Volume and Relative Spread
        new_cols[f"{asset}_volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-8)
        new_cols[f"{asset}_relative_spread"] = (high - low) / (raw_close + 1e-8)
        
        return new_cols

    def _get_time_features(self, df):
        new_cols = {}
        hours = df.index.hour
        new_cols['hour_of_day'] = hours
        new_cols['is_late_session'] = ((hours >= 14) & (hours <= 20)).astype(int)
        new_cols['is_friday'] = (df.index.dayofweek == 4).astype(int)
        return new_cols

    def _normalize_features(self, df):
        """Robust Scaling using Rolling Median and Interquartile Range (IQR)."""
        for asset in self.assets:
            cols_to_scale = [
                f"{asset}_log_return", f"{asset}_rolling_vol", f"{asset}_rolling_mean_ret",
                f"{asset}_macd_hist", f"{asset}_atr", f"{asset}_volume_ratio", f"{asset}_relative_spread",
                f"{asset}_adx"
            ]
            for col in cols_to_scale:
                if col in df.columns:
                    # Robust Normalization: (x - median) / IQR
                    median = df[col].rolling(100).median()
                    q75 = df[col].rolling(100).quantile(0.75)
                    q25 = df[col].rolling(100).quantile(0.25)
                    iqr = q75 - q25 + 1e-8
                    
                    df[col] = (df[col] - median) / iqr
                    # Clip outliers to reduce noise during training
                    df[col] = df[col].clip(-5.0, 5.0)
            
            if f"{asset}_rsi" in df.columns:
                df[f"{asset}_rsi"] = (df[f"{asset}_rsi"] - 50) / 50.0
            
            if f"{asset}_bollinger_pB" in df.columns:
                df[f"{asset}_bollinger_pB"] = df[f"{asset}_bollinger_pB"] - 0.5
        
        if 'hour_of_day' in df.columns:
            df['hour_of_day'] = (df['hour_of_day'] - 12) / 12.0
            
        return df

    def get_observation_vectorized(self, df, asset):
        obs_cols = [
            f"{asset}_open", f"{asset}_high", f"{asset}_low", f"{asset}_close", f"{asset}_volume",
            f"{asset}_log_return", f"{asset}_rolling_vol", f"{asset}_rolling_mean_ret",
            f"{asset}_rsi", f"{asset}_macd_hist", f"{asset}_bollinger_pB", f"{asset}_atr",
            'hour_of_day', 'is_late_session', 'is_friday',
            f"{asset}_volume_ratio", f"{asset}_adx"
        ]

        return df.reindex(columns=obs_cols, fill_value=0).values.astype(np.float32)

    def get_observation(self, current_step_data, portfolio_state, asset):
        obs = [
            current_step_data.get(f"{asset}_open", 0),
            current_step_data.get(f"{asset}_high", 0),
            current_step_data.get(f"{asset}_low", 0),
            current_step_data.get(f"{asset}_close", 0),
            current_step_data.get(f"{asset}_volume", 0),
            current_step_data.get(f"{asset}_log_return", 0),
            current_step_data.get(f"{asset}_rolling_vol", 0),
            current_step_data.get(f"{asset}_rolling_mean_ret", 0),
            current_step_data.get(f"{asset}_rsi", 0),
            current_step_data.get(f"{asset}_macd_hist", 0),
            current_step_data.get(f"{asset}_bollinger_pB", 0),
            current_step_data.get(f"{asset}_atr", 0),
            current_step_data.get('hour_of_day', 0),
            current_step_data.get('is_late_session', 0),
            current_step_data.get('is_friday', 0),
            current_step_data.get(f"{asset}_volume_ratio", 0),
            current_step_data.get(f"{asset}_adx", 0)
        ]
        return np.array(obs, dtype=np.float32)

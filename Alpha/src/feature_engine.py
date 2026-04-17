import pandas as pd
import numpy as np
import logging
import pywt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange, BollingerBands

class FeatureEngine:
    def __init__(self):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.feature_names = []
        self._define_feature_names()

    def _wavelet_denoise(self, data, wavelet='db4', level=1):
        """
        Performs Wavelet Denoising on a price series using Soft Thresholding.
        De-noises signal while preserving sharp edges.
        """
        # Calculate coefficients
        coeff = pywt.wavedec(data, wavelet, mode="per")
        
        # Calculate sigma using Median Absolute Deviation of the highest frequency detail coefficients
        sigma = (1/0.6745) * np.median(np.abs(coeff[-level] - np.median(coeff[-level])))
        
        # Universal Threshold calculation
        uthresh = sigma * np.sqrt(2 * np.log(len(data)))
        
        # Apply threshold to all detail coefficients except approximation
        new_coeff = [coeff[0]]
        for i in range(1, len(coeff)):
            new_coeff.append(pywt.threshold(coeff[i], value=uthresh, mode='soft'))
            
        # Reconstruct signal
        denoised = pywt.waverec(new_coeff, wavelet, mode="per")
        return denoised[:len(data)]

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
            logger.info(f"Calculating features for {asset} (with Wavelet Denoising)...")
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
        
        # Apply Wavelet Smoothing to the close price
        close_filled = raw_close.ffill().bfill().values
        close_denoised = self._wavelet_denoise(close_filled)
        close = pd.Series(close_denoised, index=raw_close.index)
        
        new_cols = {}
        # Use denoised close for log returns and volatility
        log_ret = np.log(close / (close.shift(1) + 1e-8))
        new_cols[f"{asset}_log_return"] = log_ret
        new_cols[f"{asset}_rolling_vol"] = log_ret.rolling(window=20).std()
        new_cols[f"{asset}_rolling_mean_ret"] = log_ret.rolling(window=20).mean()
        
        # Technical indicators use the denoised close for stability
        new_cols[f"{asset}_rsi"] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        new_cols[f"{asset}_macd_hist"] = macd.macd_diff()
        
        bb = BollingerBands(close, window=20, window_dev=2)
        new_cols[f"{asset}_bollinger_pB"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
        
        # ATR uses raw OHLC prices to capture true volatility range
        new_cols[f"{asset}_atr"] = AverageTrueRange(high, low, raw_close, window=14).average_true_range()
        
        from ta.trend import ADXIndicator
        new_cols[f"{asset}_adx"] = ADXIndicator(high, low, raw_close, window=14).adx()

        # New 16 & 17
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
        for asset in self.assets:
            cols_to_scale = [
                f"{asset}_log_return", f"{asset}_rolling_vol", f"{asset}_rolling_mean_ret",
                f"{asset}_macd_hist", f"{asset}_atr", f"{asset}_volume_ratio", f"{asset}_relative_spread",
                f"{asset}_adx"
            ]
            for col in cols_to_scale:
                if col in df.columns:
                    df[col] = (df[col] - df[col].rolling(100).mean()) / (df[col].rolling(100).std() + 1e-8)
            
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

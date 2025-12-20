"""
TradeGuard Inference Engine
---------------------------
Wrapper class for the LightGBM TradeGuard model.
Used for real-time inference to filter trade signals.
"""

import os
import joblib
import logging
import lightgbm as lgb
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class TradeGuard:
    def __init__(self, model_path="Guard/models/tradeguard_lgbm_latest.txt"):
        self.model_path = model_path
        self.encoder_path = os.path.join(os.path.dirname(model_path), "asset_encoder.joblib")
        self.model = None
        self.encoder = None
        self._load_resources()
        
    def _load_resources(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Guard model not found at {self.model_path}")
        self.model = lgb.Booster(model_file=self.model_path)
        
        if os.path.exists(self.encoder_path):
            self.encoder = joblib.load(self.encoder_path)
        else:
            logger.warning(f"Asset encoder not found at {self.encoder_path}. Inference may fail on asset columns.")
        
    def predict_proba(self, market_features, sl_mult, tp_mult, risk_raw, asset_name, direction):
        """
        Predict the probability of a trade winning.
        
        Args:
            market_features (list/np.array): 140 market features
            sl_mult (float): Stop Loss Multiplier
            tp_mult (float): Take Profit Multiplier
            risk_raw (float): Risk Parameter (0-1)
            asset_name (str): Symbol name (e.g., 'EURUSD')
            direction (int): Trade direction (1 or -1)
            
        Returns:
            float: Probability of Win (0.0 to 1.0)
        """
        try:
            # Validate Inputs
            if len(market_features) != 140:
                logger.error(f"Invalid feature count: {len(market_features)}. Expected 140.")
                return 0.5
            
            if direction not in [1, -1]:
                logger.warning(f"Invalid direction: {direction}. Expected 1 or -1.")

            # Construct input DataFrame (must match training schema)
            # Schema: f_0...f_139, risk_raw, sl_mult, tp_mult, direction, asset_encoded
            
            data = {}
            
            # Add Features
            for i, val in enumerate(market_features):
                data[f'f_{i}'] = [val]
                
            # Add Risk Params & Direction
            data['risk_raw'] = [risk_raw]
            data['sl_mult'] = [sl_mult]
            data['tp_mult'] = [tp_mult]
            data['direction'] = [direction]
            
            # Add Asset (Encoded)
            if self.encoder:
                try:
                    # Check if known
                    if asset_name in self.encoder.classes_:
                        encoded_asset = self.encoder.transform([asset_name])[0]
                    else:
                        encoded_asset = self.encoder.transform(['UNKNOWN'])[0]
                except Exception as e:
                    logger.warning(f"Encoding failed for {asset_name}: {e}. Using 0.")
                    encoded_asset = 0
                data['asset_encoded'] = [encoded_asset]
            else:
                # Fallback if no encoder (legacy mode or error)
                data['asset'] = [asset_name]
            
            df = pd.DataFrame(data)

            # Fallback categorical conversion if needed
            if not self.encoder and 'asset' in df.columns:
                df['asset'] = df['asset'].astype('category')
            
            # Predict
            prob = self.model.predict(df)[0]
            
            # Validate output
            if not (0 <= prob <= 1):
                logger.warning(f"Invalid probability {prob} from Guard. Clipping.")
                prob = np.clip(prob, 0.0, 1.0)
                
            return prob
            
        except Exception as e:
            logger.error(f"Guard prediction failed: {e}")
            return 0.5 # Fail open (neutral) or closed depending on policy

    def check_trade(self, market_features, sl_mult, tp_mult, risk_raw, asset_name, direction, threshold=0.5):
        """
        Binary check: Should we take this trade?
        """
        prob = self.predict_proba(market_features, sl_mult, tp_mult, risk_raw, asset_name, direction)
        return prob > threshold, prob

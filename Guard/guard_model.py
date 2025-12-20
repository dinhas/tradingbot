"""
TradeGuard Inference Engine
---------------------------
Wrapper class for the LightGBM TradeGuard model.
Used for real-time inference to filter trade signals.
"""

import os
import lightgbm as lgb
import pandas as pd
import numpy as np

class TradeGuard:
    def __init__(self, model_path="Guard/models/tradeguard_lgbm.txt"):
        self.model_path = model_path
        self.model = None
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Guard model not found at {self.model_path}")
        self.model = lgb.Booster(model_file=self.model_path)
        
    def predict_proba(self, market_features, sl_mult, tp_mult, risk_raw, asset_name):
        """
        Predict the probability of a trade winning.
        
        Args:
            market_features (list/np.array): 140 market features
            sl_mult (float): Stop Loss Multiplier
            tp_mult (float): Take Profit Multiplier
            risk_raw (float): Risk Parameter (0-1)
            asset_name (str): Symbol name (e.g., 'EURUSD')
            
        Returns:
            float: Probability of Win (0.0 to 1.0)
        """
        # Construct input DataFrame (must match training schema)
        # Schema: f_0...f_139, risk_raw, sl_mult, tp_mult, asset
        
        data = {}
        
        # Add Features
        for i, val in enumerate(market_features):
            data[f'f_{i}'] = [val]
            
        # Add Risk Params
        data['risk_raw'] = [risk_raw]
        data['sl_mult'] = [sl_mult]
        data['tp_mult'] = [tp_mult]
        
        # Add Asset (Categorical)
        # Note: We pass it as string/category. LightGBM handles the mapping if trained correctly.
        data['asset'] = [asset_name]
        
        df = pd.DataFrame(data)
        
        # Convert asset to category type to match training
        df['asset'] = df['asset'].astype('category')
        
        # Predict
        prob = self.model.predict(df)[0]
        return prob

    def check_trade(self, market_features, sl_mult, tp_mult, risk_raw, asset_name, threshold=0.5):
        """
        Binary check: Should we take this trade?
        """
        prob = self.predict_proba(market_features, sl_mult, tp_mult, risk_raw, asset_name)
        return prob > threshold, prob

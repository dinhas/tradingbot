import torch
import os
import joblib

MODELS_DIR = r"e:\tradingbot\RiskLayer\models"
MODEL_PATH = os.path.join(MODELS_DIR, "risk_model_sl_best.pth")
SCALER_PATH = os.path.join(MODELS_DIR, "sl_risk_scaler.pkl")

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    weight_key = 'input_proj.0.weight'
    if weight_key in state_dict:
         print(f"Model Input Dimension: {state_dict[weight_key].shape[1]}")
    elif 'module.' + weight_key in state_dict:
         print(f"Model Input Dimension: {state_dict['module.' + weight_key].shape[1]}")
    else:
        # Look for the first linear layer
        for k, v in state_dict.items():
            if 'weight' in k and len(v.shape) == 2:
                print(f"Found weight key {k} with shape {v.shape}. Input dim likely {v.shape[1]}")
                break

if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    if hasattr(scaler, 'n_features_in_'):
        print(f"Scaler Input Dimension: {scaler.n_features_in_}")
    else:
        print(f"Scaler has no n_features_in_, but mean shape is {scaler.mean_.shape}")

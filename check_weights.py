import torch
import os

MODELS_DIR = r"e:\tradingbot\RiskLayer\models"
MODEL_PATH = os.path.join(MODELS_DIR, "risk_model_sl_best.pth")

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    print("Keys in state_dict:")
    for k, v in state_dict.items():
        if 'weight' in k or 'bias' in k:
            print(f"{k}: {v.shape}")
else:
    print(f"Model not found at {MODEL_PATH}")

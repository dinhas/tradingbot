import torch
from pathlib import Path

risk_model_path = Path("E:/tradingbot/RiskLayer/models/risk_model_sl_final.pth")
if risk_model_path.exists():
    state_dict = torch.load(risk_model_path, map_location="cpu")
    print(f"input_proj.0.weight shape: {state_dict['input_proj.0.weight'].shape}")
else:
    print("Risk model path not found.")

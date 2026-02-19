import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path("E:/tradingbot")
sys.path.append(str(PROJECT_ROOT))

from RiskLayer.src.risk_model_sl import RiskModelSL

model = RiskModelSL(input_dim=48)
print(f"Model input_proj[0].weight shape: {model.input_proj[0].weight.shape}")

risk_model_path = PROJECT_ROOT / "RiskLayer/models/risk_model_sl_final.pth"
if risk_model_path.exists():
    state_dict = torch.load(risk_model_path, map_location="cpu")
    print(f"Checkpoint input_proj.0.weight shape: {state_dict['input_proj.0.weight'].shape}")
else:
    print("Risk model path not found.")

import os
import logging
import torch
import torch.nn as nn
import joblib
import numpy as np
from stable_baselines3 import PPO
from pathlib import Path

# --- PyTorch SL Risk Model Architecture ---

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.ln1(x)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return residual + out

class RiskModelSL(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=256, num_res_blocks=3):
        super(RiskModelSL, self).__init__()

        # Initial Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim)
        )

        # Shared Residual Backbone
        self.backbone = nn.Sequential(*[
            ResidualBlock(hidden_dim) for _ in range(num_res_blocks)
        ])

        # --- Task Specific Heads ---

        # SL Head: Predicts multiplier (e.g., 0.2 to 5.0)
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus() # Ensures positive output
        )

        # TP Head: Predicts multiplier (e.g., 0.1 to 10.0)
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus()
        )

        # Size Head: Predicts confidence/size (0.0 to 1.0)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid() # Constrains to [0, 1]
        )

    def forward(self, x):
        features = self.input_proj(x)
        features = self.backbone(features)

        sl = self.sl_head(features)
        tp = self.tp_head(features)
        size = self.size_head(features)

        return {
            'sl': sl,
            'tp': tp,
            'size': size
        }

# --- Model Loader ---

class ModelLoader:
    """
    Loads and provides inference for Alpha, Risk, and TradeGuard models.
    """
    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.alpha_model = None
        self.risk_model = None
        self.risk_scaler = None

    def load_all_models(self):
        """Loads all models from their respective paths."""
        try:
            # 1. Alpha Model (v8.03) - PPO
            alpha_path = self.project_root / "models" / "alpha" / "8.03.zip"
            if not alpha_path.exists():
                 # Fallback to general checkpoints
                 alpha_path = self.project_root / "models" / "checkpoints" / "ppo_final_model.zip"

            self.logger.info(f"Loading Alpha model from {alpha_path}...")
            self.alpha_model = PPO.load(alpha_path, device='cpu')
            
            # 2. Risk Model (SL Best) - PyTorch
            risk_path = self.project_root / "models" / "risk_model_sl_best.pth"
            self.logger.info(f"Loading SL Risk model from {risk_path}...")
            self.risk_model = RiskModelSL(input_dim=60)
            self.risk_model.load_state_dict(torch.load(risk_path, map_location=self.device))
            self.risk_model.to(self.device)
            self.risk_model.eval()

            # 3. Risk Scaler
            scaler_path = self.project_root / "models" / "sl_risk_scaler.pkl"
            self.logger.info(f"Loading Risk scaler from {scaler_path}...")
            self.risk_scaler = joblib.load(scaler_path)
            
            self.logger.info("All models loaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False

    def get_alpha_action(self, observation):
        """Predicts direction (Long/Short/Hold) from Alpha model."""
        if self.alpha_model is None:
            raise RuntimeError("Alpha model not loaded.")
        # Ensure observation is 2D for SB3
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        action, _ = self.alpha_model.predict(observation, deterministic=True)
        return action.flatten()

    def get_risk_action(self, observation):
        """Predicts SL/TP multipliers and confidence (size) using SL model."""
        if self.risk_model is None or self.risk_scaler is None:
            raise RuntimeError("Risk model or scaler not loaded.")

        # 1. Scale observation
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        obs_scaled = self.risk_scaler.transform(observation).astype(np.float32)

        # 2. Convert to tensor
        obs_tensor = torch.from_numpy(obs_scaled).to(self.device)

        # 3. Inference
        with torch.no_grad():
            preds = self.risk_model(obs_tensor)

        return {
            'sl_mult': float(preds['sl'].item()),
            'tp_mult': float(preds['tp'].item()),
            'size': float(preds['size'].item())
        }


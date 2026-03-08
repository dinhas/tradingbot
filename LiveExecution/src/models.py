import os
import sys
import logging
import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO

# Add RiskLayer/src to path for custom policy loading
_risklayer_src = Path(__file__).resolve().parent.parent.parent / "Risklayer" / "src"
if str(_risklayer_src) not in sys.path:
    sys.path.insert(0, str(_risklayer_src))

# --- PyTorch Model Architectures ---


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


class AlphaSLModel(nn.Module):
    def __init__(
        self, input_dim: int = 40, hidden_dim: int = 256, num_res_blocks: int = 4
    ):
        super(AlphaSLModel, self).__init__()

        # Initial Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
        )

        # Residual Backbone
        self.backbone = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=0.2) for _ in range(num_res_blocks)]
        )

        # Head A: Direction (3 classes: -1, 0, 1)
        self.direction_head = nn.Linear(hidden_dim, 3)

        # Head B: Quality (Regression [0, 1])
        self.quality_head = nn.Linear(hidden_dim, 1)

        # Head C: Meta (Binary [0, 1])
        self.meta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.input_proj(x)
        features = self.backbone(features)

        # Direction: 3 classes
        direction_logits = self.direction_head(features)

        # Quality: Linear output
        quality = self.quality_head(features)

        # Meta: Raw Logits (For BCEWithLogitsLoss stability)
        meta_logits = self.meta_head(features)

        return direction_logits, quality, meta_logits


class RiskModelSL(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256, num_res_blocks=3):
        super(RiskModelSL, self).__init__()

        # Initial Projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.LayerNorm(hidden_dim),
        )

        # Shared Residual Backbone
        self.backbone = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        # --- Task Specific Heads ---

        # SL Head: Predicts multiplier (e.g., 0.2 to 5.0)
        self.sl_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus(),  # Ensures positive output
        )

        # TP Head: Predicts multiplier (e.g., 0.1 to 10.0)
        self.tp_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Softplus(),
        )

        # Size Head: Predicts confidence/size (0.0 to 1.0)
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Constrains to [0, 1]
        )

    def forward(self, x):
        features = self.input_proj(x)
        features = self.backbone(features)

        sl = self.sl_head(features)
        tp = self.tp_head(features)
        size = self.size_head(features)

        return {"sl": sl, "tp": tp, "size": size}


# --- Model Loader ---


class ModelLoader:
    """
    Loads and provides inference for Alpha and Risk models.
    Uses RL (PPO) risk model for live execution.
    """

    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.alpha_model = None
        self.risk_model = None
        self.risk_scaler = None
        self.use_rl = True

    def load_all_models(self):
        """Loads all models from their respective paths."""
        try:
            # 1. Alpha Model - Supervised Learning
            alpha_path = self.project_root / "Alpha" / "models" / "alpha_model.pth"
            if not alpha_path.exists():
                alpha_path = self.project_root / "models" / "alpha" / "alpha_model.pth"

            self.logger.info(f"Loading Alpha SL model from {alpha_path}...")
            self.alpha_model = AlphaSLModel(input_dim=40)
            self.alpha_model.load_state_dict(
                torch.load(alpha_path, map_location=self.device)
            )
            self.alpha_model.to(self.device)
            self.alpha_model.eval()

            # 2. Risk Model - RL (PPO)
            risk_model_path = (
                self.project_root
                / "RiskLayer"
                / "models"
                / "ppo_risk_model_final_v2_opt.zip"
            )
            if not risk_model_path.exists():
                self.logger.warning(
                    f"RL Risk model not found at {risk_model_path}, falling back to SL..."
                )
                self.use_rl = False
                risk_path = (
                    self.project_root
                    / "RiskLayer"
                    / "models"
                    / "risk_model_sl_best.pth"
                )
                if not risk_path.exists():
                    risk_path = (
                        self.project_root / "models" / "risk" / "risk_model_sl_best.pth"
                    )
                self.logger.info(f"Loading SL Risk model from {risk_path}...")
                self.risk_model = RiskModelSL(input_dim=40)
                self.risk_model.load_state_dict(
                    torch.load(risk_path, map_location=self.device)
                )
                self.risk_model.to(self.device)
                self.risk_model.eval()
            else:
                self.logger.info(
                    f"Loading RL Risk model (PPO) from {risk_model_path}..."
                )
                self.risk_model = PPO.load(risk_model_path, device=self.device)
                self.logger.info("RL Risk model loaded successfully.")

            # 3. Risk Scaler
            if self.use_rl:
                scaler_path = (
                    self.project_root / "RiskLayer" / "models" / "rl_risk_scaler.pkl"
                )
            else:
                scaler_path = (
                    self.project_root / "RiskLayer" / "models" / "sl_risk_scaler.pkl"
                )
                if not scaler_path.exists():
                    scaler_path = (
                        self.project_root / "models" / "risk" / "sl_risk_scaler.pkl"
                    )

            self.logger.info(f"Loading Risk scaler from {scaler_path}...")
            self.risk_scaler = joblib.load(scaler_path)

            self.logger.info("All models loaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False

    def get_alpha_action(self, observation):
        """Predicts direction, quality, and meta from Alpha SL model."""
        if self.alpha_model is None:
            raise RuntimeError("Alpha model not loaded.")

        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # SL model doesn't have a normalizer yet (as per user), so use raw features
        obs_tensor = torch.from_numpy(observation.astype(np.float32)).to(self.device)

        with torch.no_grad():
            direction_logits, quality, meta_logits = self.alpha_model(obs_tensor)

            # 1. Direction: Map (0, 1, 2) -> (-1, 0, 1)
            pred_class = torch.argmax(direction_logits, dim=1)
            direction = (pred_class - 1).cpu().numpy().astype(float)

            # 2. Quality: Linear output [0, 1]
            quality_val = quality.cpu().numpy().flatten().astype(float)

            # 3. Meta: Sigmoid of logits [0, 1]
            meta_val = torch.sigmoid(meta_logits).cpu().numpy().flatten().astype(float)

        return {"direction": direction, "quality": quality_val, "meta": meta_val}

    def get_risk_action(self, observation):
        """Predicts SL/TP multipliers and confidence (size) using RL or SL model."""
        if self.risk_model is None or self.risk_scaler is None:
            raise RuntimeError("Risk model or scaler not loaded.")

        # 1. Scale observation
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        obs_scaled = self.risk_scaler.transform(observation).astype(np.float32)

        if self.use_rl:
            # RL (PPO) model inference
            obs_tensor = torch.from_numpy(obs_scaled).to(self.device)
            with torch.no_grad():
                action, _ = self.risk_model.predict(obs_tensor, deterministic=True)

            sl_raw = float(action[0, 0])
            tp_raw = float(action[0, 1])
            size_raw = float(action[0, 2])

            # Convert from [-1, 1] to actual values
            sl_mult = 0.8 + (sl_raw + 1) / 2 * (3.5 - 0.8)
            tp_mult = 1.2 + (tp_raw + 1) / 2 * (8.0 - 1.2)
            # Use fixed conservative size for live execution
            size = 0.1 + (size_raw + 1) / 2 * (0.3 - 0.1)
            size = max(0.1, min(0.3, size))  # Clamp to conservative range

            return {"sl_mult": sl_mult, "tp_mult": tp_mult, "size": size}
        else:
            # SL model inference
            obs_tensor = torch.from_numpy(obs_scaled).to(self.device)
            with torch.no_grad():
                preds = self.risk_model(obs_tensor)

            return {
                "sl_mult": float(preds["sl"].item()),
                "tp_mult": float(preds["tp"].item()),
                "size": float(preds["size"].item()),
            }

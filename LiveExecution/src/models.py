import os
import logging
from stable_baselines3 import PPO
from pathlib import Path

class ModelLoader:
    """
    Loads and provides inference for Alpha, Risk, and TradeGuard models.
    """
    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        self.alpha_model = None
        self.risk_model = None

    def load_all_models(self):
        """Loads all RL models from their respective paths."""
        try:
            # Alpha Model (v8.03)
            alpha_path = self.project_root / "models" / "alpha" / "8.03.zip"
            self.logger.info(f"Loading Alpha model from {alpha_path}...")
            self.alpha_model = PPO.load(alpha_path)
            
            # Risk Model (v2.15)
            risk_path = self.project_root / "models" / "risk" / "2.15.zip"
            self.logger.info(f"Loading Risk model from {risk_path}...")
            self.risk_model = PPO.load(risk_path)
            
            self.logger.info("All models loaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False

    def get_alpha_action(self, observation):
        """Predicts direction (Long/Short/Hold) from Alpha model."""
        if self.alpha_model is None:
            raise RuntimeError("Alpha model not loaded.")
        action, _ = self.alpha_model.predict(observation, deterministic=True)
        return action

    def get_risk_action(self, observation):
        """Predicts position size and SL/TP from Risk model."""
        if self.risk_model is None:
            raise RuntimeError("Risk model not loaded.")
        action, _ = self.risk_model.predict(observation, deterministic=True)
        return action


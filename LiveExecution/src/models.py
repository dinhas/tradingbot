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
        self.tradeguard_model = None

    def load_all_models(self):
        """Loads all three RL models from their respective paths."""
        try:
            # Alpha Model (v8.03)
            alpha_path = self.project_root / "checkpoints" / "8.03.zip"
            self.logger.info(f"Loading Alpha model from {alpha_path}...")
            self.alpha_model = PPO.load(alpha_path)
            
            # Risk Model (v2.15)
            risk_path = self.project_root / "RiskLayer" / "models" / "2.15.zip"
            self.logger.info(f"Loading Risk model from {risk_path}...")
            self.risk_model = PPO.load(risk_path)
            
            # TradeGuard Model
            # PRD mentions LightGBM but codebase uses PPO for TradeGuard too
            tg_path = self.project_root / "TradeGuard" / "models" / "manual_test_model.zip"
            self.logger.info(f"Loading TradeGuard model from {tg_path}...")
            self.tradeguard_model = PPO.load(tg_path)
            
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

    def get_tradeguard_action(self, observation):
        """Predicts Allow/Block (1/0) from TradeGuard model."""
        if self.tradeguard_model is None:
            raise RuntimeError("TradeGuard model not loaded.")
        action, _ = self.tradeguard_model.predict(observation, deterministic=True)
        return action

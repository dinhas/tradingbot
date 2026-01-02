import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from pathlib import Path

class ModelLoader:
    """
    Loads and provides inference for Alpha, Risk, and TradeGuard models.
    """
    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        self.alpha_model = None
        self.alpha_norm = None
        self.risk_model = None
        self.risk_norm = None
        self.tradeguard_model = None

    def load_all_models(self):
        """Loads all three RL models and their normalizers from their respective paths."""
        try:
            # Alpha Model
            alpha_model_path = self.project_root / "models" / "checkpoints" / "alpha" / "ppo_final_model.zip"
            alpha_norm_path = self.project_root / "models" / "checkpoints" / "alpha" / "ppo_final_vecnormalize.pkl"
            
            self.logger.info(f"Loading Alpha model from {alpha_model_path}...")
            self.alpha_model = PPO.load(alpha_model_path)
            
            if alpha_norm_path.exists():
                self.logger.info(f"Loading Alpha normalizer from {alpha_norm_path}...")
                self.alpha_norm = VecNormalize.load(alpha_norm_path, venv=None)
                self.alpha_norm.training = False # Ensure we don't update stats
                self.alpha_norm.norm_reward = False
            else:
                self.logger.warning("Alpha normalizer not found!")

            # Risk Model
            risk_model_path = self.project_root / "models" / "checkpoints" / "risk" / "model10M.zip"
            risk_norm_path = self.project_root / "models" / "checkpoints" / "risk" / "model10M.pkl"
            
            self.logger.info(f"Loading Risk model from {risk_model_path}...")
            self.risk_model = PPO.load(risk_model_path)
            
            if risk_norm_path.exists():
                self.logger.info(f"Loading Risk normalizer from {risk_norm_path}...")
                self.risk_norm = VecNormalize.load(risk_norm_path, venv=None)
                self.risk_norm.training = False
                self.risk_norm.norm_reward = False
            else:
                self.logger.warning("Risk normalizer not found!")
            
            # TradeGuard Model
            tg_path = self.project_root / "TradeGuard" / "models" / "manual_test_model.zip"
            self.logger.info(f"Loading TradeGuard model from {tg_path}...")
            self.tradeguard_model = PPO.load(tg_path)
            
            self.logger.info("All models loaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def get_alpha_action(self, observation):
        """Predicts direction (Long/Short/Hold) from Alpha model."""
        if self.alpha_model is None:
            raise RuntimeError("Alpha model not loaded.")
            
        # Normalize observation if normalizer exists
        if self.alpha_norm is not None:
            observation = self.alpha_norm.normalize_obs(observation)
            
        action, _ = self.alpha_model.predict(observation, deterministic=True)
        return action

    def get_risk_action(self, observation):
        """Predicts SL/TP multipliers from Risk model."""
        if self.risk_model is None:
            raise RuntimeError("Risk model not loaded.")
            
        # Normalize observation if normalizer exists
        if self.risk_norm is not None:
            observation = self.risk_norm.normalize_obs(observation)
            
        action, _ = self.risk_model.predict(observation, deterministic=True)
        return action

    def get_tradeguard_action(self, observation):
        """Predicts Allow/Block (1/0) from TradeGuard model."""
        if self.tradeguard_model is None:
            raise RuntimeError("TradeGuard model not loaded.")
        action, _ = self.tradeguard_model.predict(observation, deterministic=True)
        return action

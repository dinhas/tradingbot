import os
import logging
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pathlib import Path

class ModelLoader:
    """
    Loads and provides inference for Alpha and Risk models.
    Supports VecNormalize for proper feature scaling.
    """
    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        self.alpha_model = None
        self.alpha_norm = None
        
        self.risk_model = None
        self.risk_norm = None

    def _load_vec_normalize(self, venv_path, model):
        """Loads VecNormalize statistics and wraps a dummy environment."""
        try:
            # VecNormalize needs a dummy env to be loaded
            from gymnasium import spaces
            # Create a dummy env with the same spaces as the model
            class DummyEnv:
                def __init__(self, observation_space, action_space):
                    self.observation_space = observation_space
                    self.action_space = action_space
                def reset(self): return np.zeros(self.observation_space.shape)
                def step(self, action): return np.zeros(self.observation_space.shape), 0, False, False, {}
            
            # We don't actually need a real env, just something with spaces
            # But SB3 VecNormalize.load expects a VecEnv
            dummy_venv = DummyVecEnv([lambda: DummyEnv(model.observation_space, model.action_space)])
            vn = VecNormalize.load(str(venv_path), dummy_venv)
            vn.training = False # Disable updates
            vn.norm_reward = False # Only normalize observations
            return vn
        except Exception as e:
            self.logger.warning(f"Could not load VecNormalize from {venv_path}: {e}")
            return None

    def load_all_models(self):
        """Loads RL models and their normalizers from the checkpoints directory."""
        try:
            checkpoint_dir = self.project_root / "models" / "checkpoints"
            
            # 1. Alpha Model (Single-Asset Refactored)
            alpha_dir = checkpoint_dir / "alpha"
            alpha_path = alpha_dir / "ppo_final_model.zip"
            alpha_norm_path = alpha_dir / "ppo_final_vecnormalize.pkl"
            
            self.logger.info(f"Loading Alpha model from {alpha_path}...")
            self.alpha_model = PPO.load(alpha_path)
            self.alpha_norm = self._load_vec_normalize(alpha_norm_path, self.alpha_model)
            if self.alpha_norm:
                self.logger.info("Alpha VecNormalize stats loaded.")
            
            # 2. Risk Model (45-dim Refactored)
            risk_dir = checkpoint_dir / "risk"
            risk_path = risk_dir / "10M.zip"
            risk_norm_path = risk_dir / "10M.pkl"
            
            self.logger.info(f"Loading Risk model from {risk_path}...")
            self.risk_model = PPO.load(risk_path)
            self.risk_norm = self._load_vec_normalize(risk_norm_path, self.risk_model)
            if self.risk_norm:
                self.logger.info("Risk VecNormalize stats loaded.")
            
            self.logger.info("All models and normalizers loaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def get_alpha_action(self, observation):
        """Predicts directional signal for a single asset."""
        if self.alpha_model is None:
            raise RuntimeError("Alpha model not loaded.")
        
        # Apply normalization if available
        if self.alpha_norm:
            observation = self.alpha_norm.normalize_obs(observation.reshape(1, -1)).flatten()
            
        action, _ = self.alpha_model.predict(observation, deterministic=True)
        # Should return a single value or (1,) array
        return action

    def get_risk_action(self, observation):
        """Predicts position risk parameters (SL, TP)."""
        if self.risk_model is None:
            raise RuntimeError("Risk model not loaded.")
        
        # Apply normalization if available
        if self.risk_norm:
            observation = self.risk_norm.normalize_obs(observation.reshape(1, -1)).flatten()
            
        action, _ = self.risk_model.predict(observation, deterministic=True)
        # Should return (2,) array for [SL_Mult, TP_Mult]
        return action


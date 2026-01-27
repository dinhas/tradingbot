"""
Curriculum Learning Callback for Alpha Model Training.

Gradually increases spread/fees simulation during training to help the model
learn to overcome transaction costs step by step.
"""
from stable_baselines3.common.callbacks import BaseCallback
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AlphaCurriculumCallback(BaseCallback):
    """
    Adjusts environment difficulty (spreads/fees) over time for Alpha Model.
    
    The 'spread_modifier' in the environment controls all fees:
    - 0.0: No spreads, no slippage, no commission (Pure theoretical price)
    - 0.5: 50% of real-world costs
    - 1.0: 100% of real-world costs (Full reality)
    
    Schedule Options (for 8M steps):
    
    Option 'original' (User's request):
    - 0       - 3.5M: 0% Fees
    - 3.5M    - 5.0M: 20% Fees
    - 5.0M    - 6.0M: 45% Fees
    - 6.0M    - 7.0M: 75% Fees
    - 7.0M    - 8.0M: 100% Fees
    
    Option 'recommended' (Smoother progression):
    - 0       - 2.0M: 0% Fees
    - 2.0M    - 3.5M: 25% Fees
    - 3.5M    - 5.0M: 50% Fees
    - 5.0M    - 6.5M: 75% Fees
    - 6.5M    - 8.0M: 100% Fees
    """
    
    def __init__(self, 
                 total_timesteps: int = 8_000_000, 
                 schedule: str = "recommended",
                 verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_modifier = -1.0
        self.schedule_name = schedule
        
        # Define curriculum schedules as (threshold_steps, modifier)
        if schedule == "original":
            self.spread_schedule = [
                (int(total_timesteps * 0.4375), 0.0),   # 0 - 3.5M: 0%
                (int(total_timesteps * 0.625), 0.20),   # 3.5M - 5M: 20%
                (int(total_timesteps * 0.75), 0.45),    # 5M - 6M: 45%
                (int(total_timesteps * 0.875), 0.75),   # 6M - 7M: 75%
                (float('inf'), 1.0)                     # 7M+: 100%
            ]
        else:  # recommended (default)
            self.spread_schedule = [
                (int(total_timesteps * 0.25), 0.0),     # 0 - 2M: 0%
                (int(total_timesteps * 0.4375), 0.25),  # 2M - 3.5M: 25%
                (int(total_timesteps * 0.625), 0.50),   # 3.5M - 5M: 50%
                (int(total_timesteps * 0.8125), 0.75),  # 5M - 6.5M: 75%
                (float('inf'), 1.0)                     # 6.5M+: 100%
            ]
        
        if verbose > 0:
            logger.info(f"AlphaCurriculumCallback initialized with '{schedule}' schedule:")
            for threshold, modifier in self.spread_schedule:
                if threshold == float('inf'):
                    logger.info(f"  Final stage: {modifier*100:.0f}% Fees")
                else:
                    logger.info(f"  Up to {threshold:,} steps: {modifier*100:.0f}% Fees")

    def _on_step(self) -> bool:
        """Called at every step. Updates spread modifier when crossing thresholds."""
        current_steps = self.num_timesteps
        target_modifier = 0.0
        
        # Find the current stage based on timesteps
        for threshold, modifier in self.spread_schedule:
            if current_steps < threshold:
                target_modifier = modifier
                break
        else:
            target_modifier = 1.0
            
        # Only update if modifier changed (avoid unnecessary calls)
        if target_modifier != self.last_modifier:
            self.last_modifier = target_modifier
            
            if self.verbose > 0:
                stage_name = f"{target_modifier*100:.0f}% Fees"
                logger.info(f"[Curriculum] Step {current_steps:,}: Advancing to {stage_name}")
                print(f"\n{'='*60}")
                print(f"  ALPHA CURRICULUM UPDATE: {stage_name}")
                print(f"  Step: {current_steps:,} / {self.total_timesteps:,}")
                print(f"  Schedule: {self.schedule_name}")
                print(f"{'='*60}\n")
            
            # Apply to all vectorized environments
            try:
                self.training_env.env_method("set_spread_modifier", target_modifier)
            except Exception as e:
                logger.warning(f"Failed to set spread modifier: {e}")
                
        return True
    
    def _on_training_start(self) -> None:
        """Called at the start of training. Ensures we start with correct initial spread."""
        if self.verbose > 0:
            logger.info("[Curriculum] Training started - Initializing fees...")
            
        target_modifier = self.spread_schedule[0][1]
        try:
            self.training_env.env_method("set_spread_modifier", target_modifier)
            self.last_modifier = target_modifier
            if self.verbose > 0:
                logger.info(f"[Curriculum] Initial fees set to {target_modifier*100:.0f}%")
        except Exception as e:
            logger.warning(f"Failed to set initial spread modifier: {e}")


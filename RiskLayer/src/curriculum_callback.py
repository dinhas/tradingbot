"""
Curriculum Learning Callback for Risk Model Training.

Gradually increases spread simulation during training to help the model
learn to survive market friction step by step.
"""
from stable_baselines3.common.callbacks import BaseCallback
import logging

logger = logging.getLogger(__name__)


class CurriculumCallback(BaseCallback):
    """
    Adjusts environment difficulty (spreads) over time for Risk Model.
    
    Schedule (6M total timesteps):
    - 0       - 1.5M: 0% Spread (Learn base risk management)
    - 1.5M    - 2.5M: 25% Spread
    - 2.5M    - 3.5M: 50% Spread  
    - 3.5M    - 4.5M: 75% Spread
    - 4.5M    - 6.0M: 100% Spread (Full Reality)
    
    This gradual increase helps the model:
    1. First learn SL/TP placement without friction
    2. Then adapt to increasing trading costs
    3. Finally master full spread simulation
    """
    
    def __init__(self, total_timesteps: int = 6_000_000, verbose: int = 1):
        super(CurriculumCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_modifier = -1.0
        
        # Define curriculum schedule as (threshold_steps, spread_modifier)
        # Thresholds are calculated as fractions of total training
        self.spread_schedule = [
            (int(total_timesteps * 0.25), 0.0),   # First 25%: No spread
            (int(total_timesteps * 0.40), 0.25),  # 25% - 40%: 25% spread
            (int(total_timesteps * 0.55), 0.50),  # 40% - 55%: 50% spread
            (int(total_timesteps * 0.70), 0.75),  # 55% - 70%: 75% spread
            (float('inf'), 1.0)                    # 70%+: Full spread
        ]
        
        if verbose > 0:
            logger.info("CurriculumCallback initialized with schedule:")
            for threshold, modifier in self.spread_schedule:
                if threshold == float('inf'):
                    logger.info(f"  After {int(total_timesteps * 0.70):,}+ steps: {modifier*100:.0f}% spread")
                else:
                    logger.info(f"  Up to {threshold:,} steps: {modifier*100:.0f}% spread")

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
                stage_name = f"{target_modifier*100:.0f}% Spread"
                logger.info(f"[Curriculum] Step {current_steps:,}: Advancing to {stage_name}")
                print(f"\n{'='*50}")
                print(f"  CURRICULUM UPDATE: {stage_name}")
                print(f"  Step: {current_steps:,} / {self.total_timesteps:,}")
                print(f"{'='*50}\n")
            
            # Apply to all vectorized environments
            # env_method calls the method on each sub-environment
            try:
                self.training_env.env_method("set_spread_modifier", target_modifier)
            except Exception as e:
                logger.warning(f"Failed to set spread modifier: {e}")
                
        return True
    
    def _on_training_start(self) -> None:
        """Called at the start of training. Ensures we start with 0% spread."""
        if self.verbose > 0:
            logger.info("[Curriculum] Training started - Setting initial spread to 0%")
        try:
            self.training_env.env_method("set_spread_modifier", 0.0)
            self.last_modifier = 0.0
        except Exception as e:
            logger.warning(f"Failed to set initial spread modifier: {e}")

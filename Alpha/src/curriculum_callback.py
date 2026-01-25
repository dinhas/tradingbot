from stable_baselines3.common.callbacks import BaseCallback
import logging

logger = logging.getLogger(__name__)

class CurriculumCallback(BaseCallback):
    """
    Adjusts environment difficulty (spreads) over time.
    
    Schedule:
    - 0       - 5.0M: 0% Spread (Master Direction)
    - 5.0M    - 6.5M: 15% Spread
    - 6.5M    - 7.5M: 30% Spread
    - 7.5M    - 8.5M: 50% Spread
    - 8.5M    - 9.5M: 75% Spread
    - 9.5M    - 10.5M: 100% Spread
    """
    def __init__(self, verbose=1):
        super(CurriculumCallback, self).__init__(verbose)
        self.current_stage = 0
        self.spread_schedule = [
            (1000000, 0.0),   # Up to 1M (Warmup)
            (2000000, 0.25),  # 1M - 2M
            (3000000, 0.50),  # 2M - 3M
            (4000000, 0.75),  # 3M - 4M
            (float('inf'), 1.0) # After 4M (Full Reality)
        ]
        self.last_modifier = -1.0

    def _on_step(self) -> bool:
        # Determine target spread modifier based on current total timesteps
        current_steps = self.num_timesteps
        target_modifier = 0.0
        
        for threshold, modifier in self.spread_schedule:
            if current_steps < threshold:
                target_modifier = modifier
                break
        else:
            target_modifier = 1.0
            
        # Update only if changed
        if target_modifier != self.last_modifier:
            self.last_modifier = target_modifier
            
            if self.verbose > 0:
                logger.info(f"Step {current_steps}: Updating Spread Modifier to {target_modifier*100:.0f}%")
            
            # Apply to all environments
            # training_env is usually a VecEnv
            self.training_env.env_method("set_spread_modifier", target_modifier)
            
        return True

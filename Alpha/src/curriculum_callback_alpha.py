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
    Also decays learning rate and entropy to encourage consolidation.
    
    The 'spread_modifier' in the environment controls all fees:
    - 0.0: No spreads, no slippage, no commission (Pure theoretical price)
    - 0.5: 50% of real-world costs
    - 1.0: 100% of real-world costs (Full reality)
    """
    
    def __init__(self, 
                 total_timesteps: int = 8_000_000, 
                 schedule: str = "recommended",
                 initial_lr: float = 0.0001,
                 final_lr: float = 0.00003,
                 initial_ent: float = 0.01,
                 final_ent: float = 0.002,
                 warmup_steps: int = 150_000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.last_modifier = -1.0
        self.schedule_name = schedule
        
        # Hyperparameters for decay
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.initial_ent = initial_ent
        self.final_ent = final_ent
        self.warmup_steps = warmup_steps
        
        # Define curriculum schedules as (threshold_steps, modifier)
        if schedule == "original":
            self.spread_schedule = [
                (warmup_steps, 0.0),                     # Plateau at 0 until warmup
                (int(total_timesteps * 0.4375), 0.0),   
                (int(total_timesteps * 0.625), 0.20),   
                (int(total_timesteps * 0.75), 0.45),    
                (int(total_timesteps * 0.875), 0.75),   
                (float('inf'), 1.0)                     
            ]
        else:  # recommended (default)
            self.spread_schedule = [
                (warmup_steps, 0.0),                    # 0 - 150k: Stay at 0
                (int(total_timesteps * 0.35), 0.25),    # Start ramp to 25% early
                (int(total_timesteps * 0.55), 0.50),    # 50%
                (int(total_timesteps * 0.75), 0.75),    # 75%
                (float('inf'), 1.0)                     # 100%
            ]
        
        if verbose > 0:
            logger.info(f"AlphaCurriculumCallback initialized with '{schedule}' schedule:")
            for threshold, modifier in self.spread_schedule:
                if threshold == float('inf'):
                    logger.info(f"  Final stage: {modifier*100:.0f}% Fees")
                else:
                    logger.info(f"  Up to {threshold:,} steps: {modifier*100:.0f}% Fees")

    def _on_step(self) -> bool:
        """Called at every step. Updates spread modifier with discrete plateaus."""
        current_steps = self.num_timesteps
        total = self.total_timesteps
        
        # 1. Update Spread Modifier (Discrete Plateaus)
        # We pick the target modifier from the first threshold we haven't crossed yet
        target_modifier = 1.0
        for threshold, modifier in self.spread_schedule:
            if current_steps < threshold:
                target_modifier = modifier
                break

        # Apply Spread Modifier
        if abs(target_modifier - self.last_modifier) > 0.001:
            self.last_modifier = target_modifier
            try:
                self.training_env.env_method("set_spread_modifier", target_modifier)
            except Exception:
                pass

        # 2. Update Learning Rate & Entropy Coefficient (Global Linear Decay)
        # Keep global decay as it helps finalize the policy regardless of fees
        global_progress = min(current_steps / total, 1.0)
        new_lr = self.initial_lr + (self.final_lr - self.initial_lr) * global_progress
        new_ent = self.initial_ent + (self.final_ent - self.initial_ent) * global_progress
        
        self.model.ent_coef = new_ent
        for param_group in self.model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr

        # Logging
        if current_steps % 100_000 == 0 and self.verbose > 0:
            logger.info(
                f"[Curriculum] Step {current_steps:,}: "
                f"PLATEAU={target_modifier*100:.0f}% Fees, "
                f"LR={new_lr:.6f}, "
                f"Ent={new_ent:.4f}"
            )
                
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


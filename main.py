"""
Main Orchestrator for Multi-Asset AI Trading System
Runs the complete pipeline: Data -> Test -> Train -> Backtest
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================

ASSETS = ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']
DATA_DIR = "./"
REQUIRED_FILES = [
    'data_BTC_final.parquet',
    'data_ETH_final.parquet', 
    'data_SOL_final.parquet',
    'data_EUR_final.parquet',
    'data_GBP_final.parquet',
    'data_JPY_final.parquet',
    'volatility_baseline.json'
]

# Training Configuration
TRAINING_CONFIG = {
    'total_timesteps': 600_000,  # 600k steps
    'checkpoint_freq': 50_000,     # Save every 50k steps
    'eval_freq': 10_000,           # Evaluate every 10k steps
    'n_eval_episodes': 5
}

# ============================================================================
# PHASE 1: DATA PIPELINE
# ============================================================================

def check_data_exists():
    """Check if all required data files exist."""
    missing = []
    for file in REQUIRED_FILES:
        if not os.path.exists(os.path.join(DATA_DIR, file)):
            missing.append(file)
    return missing

def run_data_pipeline():
    """Run ctraderservice.py to fetch historical data."""
    logger.info("=" * 80)
    logger.info("PHASE 1: DATA PIPELINE")
    logger.info("=" * 80)
    
    missing = check_data_exists()
    
    if not missing:
        logger.info("✅ All data files already exist. Skipping data fetch.")
        return True
    
    logger.warning(f"❌ Missing data files: {missing}")
    logger.info("Starting data fetch... (This may take 30-60 minutes)")
    
    try:
        import subprocess
        # Run ctradercervice.py as a separate process
        subprocess.check_call([sys.executable, "ctradercervice.py"])
        
        # Verify files exist after run
        still_missing = check_data_exists()
        if still_missing:
            logger.error(f"❌ Data fetch ran but files are still missing: {still_missing}")
            return False
            
        logger.info("✅ Data pipeline completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Data pipeline failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"❌ Data pipeline failed: {e}")
        return False

# ============================================================================
# PHASE 2: ENVIRONMENT VALIDATION
# ============================================================================

def run_environment_test():
    """Run test_env.py to verify environment setup."""
    logger.info("=" * 80)
    logger.info("PHASE 2: ENVIRONMENT VALIDATION")
    logger.info("=" * 80)
    
    try:
        from trading_env import TradingEnv
        import numpy as np
        
        logger.info("Initializing environment...")
        env = TradingEnv(data_dir=DATA_DIR, volatility_file="volatility_baseline.json")
        
        logger.info("Testing reset...")
        obs, info = env.reset()
        
        # Verify observation shape
        if obs.shape != (97,):
            logger.error(f"❌ Observation shape mismatch! Expected (97,), got {obs.shape}")
            return False
        logger.info(f"✅ Observation shape correct: {obs.shape}")
        
        # Run simulation
        logger.info("Running 10-step simulation...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if i == 0:
                logger.info(f"  Step 1: Portfolio=${info['portfolio_value']:.2f}, Reward={reward:.4f}")
            
            if terminated or truncated:
                logger.warning(f"  Episode ended early at step {i+1}")
                break
        
        logger.info("✅ Environment test passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# PHASE 3: MODEL TRAINING
# ============================================================================

def apply_critical_fixes():
    """Apply critical fixes to trading_env.py before training."""
    logger.info("Applying critical fixes to reward calculation...")
    
    # The fix is already applied in the updated code
    # This is just a verification step
    try:
        with open('trading_env.py', 'r') as f:
            content = f.read()
            if 'VecNormalize' in content or 'normalized_reward * 10' in content:
                logger.info("✅ Fixes detected in trading_env.py")
            else:
                logger.warning("⚠️  Reward scaling may need adjustment")
    except:
        pass

def run_training():
    """Run the training pipeline with RecurrentPPO."""
    logger.info("=" * 80)
    logger.info("PHASE 3: MODEL TRAINING")
    logger.info("=" * 80)
    
    apply_critical_fixes()
    
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from callbacks.debug_logger import DebugLoggingCallback
        from sb3_contrib import RecurrentPPO
        from trading_env import TradingEnv
        import torch
        
        # Create directories
        os.makedirs("./models/", exist_ok=True)
        os.makedirs("./logs/", exist_ok=True)
        os.makedirs("./models/best/", exist_ok=True)
        
        # Environment factory
        def make_env(rank):
            def _init():
                return TradingEnv(data_dir=DATA_DIR, volatility_file="volatility_baseline.json")
            return _init
        
        import multiprocessing
        
        # Dynamically use all available CPU cores
        n_envs = multiprocessing.cpu_count()
        logger.info(f"Detected {n_envs} CPU cores. Creating {n_envs} parallel environments...")
        
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        
        # CRITICAL FIX: Wrap with VecNormalize for reward stabilization
        logger.info("Wrapping with VecNormalize (reward normalization)...")
        vec_env = VecNormalize(
            vec_env,
            norm_obs=False,      # We already normalize observations
            norm_reward=True,    # CRITICAL: Normalize rewards
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99
        )
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=TRAINING_CONFIG['checkpoint_freq'] // n_envs,
            save_path='./models/',
            name_prefix='recurrent_ppo_trading',
            save_vecnormalize=True
        )
        
        # Create eval env wrapped in VecNormalize to match training env
        eval_env = DummyVecEnv([lambda: TradingEnv(data_dir=DATA_DIR, volatility_file="volatility_baseline.json")])
        eval_env = VecNormalize(
            eval_env,
            norm_obs=False,       # Match training env
            norm_reward=False,    # Don't normalize reward for eval (we want to see real PnL)
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
            training=False        # Do not update stats during eval
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models/best/',
            log_path='./logs/eval/',
            eval_freq=TRAINING_CONFIG['eval_freq'] // n_envs,
            n_eval_episodes=TRAINING_CONFIG['n_eval_episodes'],
            deterministic=True
        )
        
        # Debug logging callback
        debug_callback = DebugLoggingCallback(
            log_freq=50000,
            log_dir='./debuglogs/',
            verbose=1
        )
        
        # Create model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Setting up RecurrentPPO model (Device: {device})...")
        
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                lstm_hidden_size=128,
                n_lstm_layers=2,
                enable_critic_lstm=True
            ),
            verbose=1,
            tensorboard_log="./logs/",
            device=device
        )
        
        logger.info("=" * 80)
        logger.info(f"STARTING TRAINING: {TRAINING_CONFIG['total_timesteps']:,} steps")
        logger.info(f"Estimated time: ~24 hours on A100 GPU")
        logger.info(f"Checkpoints: ./models/ (every 50k steps)")
        logger.info(f"Logs: ./logs/ (TensorBoard)")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # TRAIN
        model.learn(
            total_timesteps=TRAINING_CONFIG['total_timesteps'],
            callback=[checkpoint_callback, eval_callback, debug_callback],
            progress_bar=True
        )
        
        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"✅ TRAINING COMPLETED!")
        logger.info(f"Total time: {elapsed/3600:.2f} hours")
        logger.info("=" * 80)
        
        # Save final model
        final_path = "./models/final_model"
        model.save(final_path)
        vec_env.save(f"{final_path}_vecnormalize.pkl")
        logger.info(f"Final model saved to: {final_path}")
        
        return True, model, vec_env
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user! Saving emergency checkpoint...")
        model.save("./models/interrupted_model")
        vec_env.save("./models/interrupted_model_vecnormalize.pkl")
        logger.info("✅ Interrupted model saved to ./models/interrupted_model")
        raise

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

# ============================================================================
# PHASE 4: SUMMARY REPORT
# ============================================================================

def generate_summary_report():
    """Generate a summary of the training session."""
    logger.info("=" * 80)
    logger.info("PHASE 4: SUMMARY REPORT")
    logger.info("=" * 80)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'data_files': check_data_exists() == [],
        'training_config': TRAINING_CONFIG,
        'checkpoints': len([f for f in os.listdir('./models/') if f.endswith('.zip')]),
        'logs_dir': './logs/',
        'best_model': './models/best/'
    }
    
    report_path = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ Report saved to: {report_path}")
    logger.info(f"📊 Checkpoints created: {report['checkpoints']}")
    logger.info(f"📂 TensorBoard logs: {report['logs_dir']}")
    logger.info(f"🏆 Best model: {report['best_model']}")
    
    return report

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def send_notification(message):
    """Send notification (placeholder - extend with email/SMS/Discord)."""
    logger.info(f"📢 NOTIFICATION: {message}")
    # TODO: Add email/SMS/Discord webhook here
    # Example: requests.post(DISCORD_WEBHOOK, json={"content": message})

def main():
    """Main pipeline orchestrator."""
    logger.info("🚀 STARTING MULTI-ASSET AI TRADING SYSTEM PIPELINE")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    pipeline_start = time.time()
    
    try:
        # Phase 1: Data
        if not run_data_pipeline():
            send_notification("❌ Data pipeline failed!")
            return False
        
        # Phase 2: Environment Test
        if not run_environment_test():
            send_notification("❌ Environment validation failed!")
            return False
        
        # Phase 3: Training
        success, model, vec_env = run_training()
        if not success:
            send_notification("❌ Training failed!")
            return False
        
        # Phase 4: Summary
        report = generate_summary_report()
        
        # Success!
        elapsed = time.time() - pipeline_start
        logger.info("=" * 80)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Total pipeline time: {elapsed/3600:.2f} hours")
        logger.info("=" * 80)
        
        send_notification(f"✅ Training complete! Time: {elapsed/3600:.2f}h")
        return True
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        send_notification("⚠️ Training interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        send_notification(f"❌ Pipeline crashed: {str(e)[:100]}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
Verify Transfer Learning: Stage 1 → Stage 2
Checks if direction weights were correctly transferred
"""
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from .src.trading_env import TradingEnv
import torch

def verify_transfer():
    """Verify that Stage 1 directions are preserved in Stage 2."""
    
    print("="*80)
    print("TRANSFER LEARNING VERIFICATION")
    print("="*80)

    project_root = Path(__file__).resolve().parent.parent
    stage1_model_path = project_root / "models" / "checkpoints" / "stage_1_final.zip"
    stage2_model_path = project_root / "models" / "checkpoints" / "stage_2_final.zip"
    data_path = project_root / "data"
    
    # Load Stage 1 model
    try:
        stage1_model = PPO.load(stage1_model_path)
        print("✓ Loaded Stage 1 model")
    except Exception as e:
        print(f"✗ Failed to load Stage 1 model: {e}")
        return
    
    # Load Stage 2 model (if exists)
    stage2_model = None
    try:
        stage2_model = PPO.load(stage2_model_path)
        print("✓ Loaded Stage 2 model")
    except Exception as e:
        print(f"⚠ No Stage 2 model found (this is OK if you're starting fresh)")
        print(f"  Error: {e}")
    
    # Create test environments
    env1 = TradingEnv(data_dir=data_path, stage=1, is_training=False, initial_balance=1000)
    env2 = TradingEnv(data_dir=data_path, stage=2, is_training=False, initial_balance=1000)
    
    # Test Stage 1 performance
    print("\n" + "="*80)
    print("STAGE 1 PERFORMANCE TEST")
    print("="*80)
    
    obs1, _ = env1.reset()
    wins1 = 0
    losses1 = 0
    
    for i in range(100):
        action1, _ = stage1_model.predict(obs1, deterministic=True)
        obs1, reward1, terminated, truncated, info1 = env1.step(action1)
        
        if len(info1.get('trades', [])) > 0:
            for trade in info1['trades']:
                if trade['pnl'] > 0:
                    wins1 += 1
                else:
                    losses1 += 1
        
        if terminated or truncated:
            obs1, _ = env1.reset()
    
    winrate1 = wins1 / (wins1 + losses1) if (wins1 + losses1) > 0 else 0
    print(f"Trades: {wins1 + losses1} | Wins: {wins1} | Losses: {losses1}")
    print(f"Win Rate: {winrate1*100:.1f}%")
    
    if winrate1 < 0.40:
        print("⚠️  WARNING: Stage 1 win rate is LOW (<40%)")
        print("   → Your Stage 1 model might not be well-trained")
        print("   → This explains why Stage 2 directions are also bad")
        print("\n   SOLUTION: Retrain Stage 1 first!")
    else:
        print("✓ Stage 1 looks OK")
    
    # Test Stage 2 (if model exists)
    if stage2_model is not None:
        print("\n" + "="*80)
        print("STAGE 2 DIRECTION COMPARISON")
        print("="*80)
        
        # Reset to same state
        np.random.seed(42)
        obs_test, _ = env1.reset()
        obs2, _ = env2.reset()
        
        # Get predictions
        action1, _ = stage1_model.predict(obs_test, deterministic=True)
        action2, _ = stage2_model.predict(obs2, deterministic=True)
        
        # Extract direction outputs
        # Stage 1: 5 outputs (all directions)
        # Stage 2: 10 outputs (direction at indices 0,2,4,6,8)
        directions1 = action1  # [5]
        directions2 = action2[::2]  # Take every 2nd starting from 0: [0,2,4,6,8]
        
        print("\nDirection Outputs (5 assets):")
        print(f"Stage 1: {directions1}")
        print(f"Stage 2: {directions2}")
        print(f"\nDifference: {np.abs(directions1 - directions2)}")
        print(f"Mean Abs Difference: {np.abs(directions1 - directions2).mean():.4f}")
        
        if np.abs(directions1 - directions2).mean() > 0.3:
            print("\n⚠️  CRITICAL: Directions are VERY DIFFERENT!")
            print("   → Transfer learning probably FAILED")
            print("   → Stage 2 is relearning directions from scratch")
            print("\n   SOLUTION: Check if you used --transfer-from flag!")
        else:
            print("\n✓ Directions are similar (transfer likely worked)")
    
    env1.close()
    env2.close()
    
    # Final recommendation
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if winrate1 < 0.40:
        print("1. ⚠️  Your Stage 1 model is weak (winrate < 40%)")
        print("   → RETRAIN Stage 1 first with more steps")
        print("   → Or debug Stage 1 reward function")
    
    if stage2_model is None:
        print("\n2. ℹ️  No Stage 2 model found yet")
        print("   → Make sure you're using --transfer-from:")
        print("   → python -m Alpha.src.train --stage 2 --transfer-from models/checkpoints/stage_1_final.zip")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    verify_transfer()

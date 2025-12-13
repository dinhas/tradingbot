"""
Quick validation script to test backtesting infrastructure setup.

This script checks:
1. Directory structure is correct
2. Required files exist
3. Imports work correctly
4. Basic functionality is accessible
"""

import os
import sys

def check_directory_structure():
    """Verify backtest directory structure"""
    print("Checking directory structure...")
    
    required_dirs = [
        "backtest",
        "backtest/data",
        "backtest/results"
    ]
    
    required_files = [
        "backtest/__init__.py",
        "backtest/data_fetcher_backtest.py",
        "backtest/backtest.py",
        "backtest/compare_stages.py",
        "backtest/README.md"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            all_good = False
    
    return all_good


def check_imports():
    """Verify required imports work"""
    print("\nChecking imports...")
    
    try:
        sys.path.insert(0, os.path.abspath('.'))
        
        # Test basic imports
        import pandas as pd
        print("  ✅ pandas")
        
        import numpy as np
        print("  ✅ numpy")
        
        import matplotlib.pyplot as plt
        print("  ✅ matplotlib")
        
        from stable_baselines3 import PPO
        print("  ✅ stable_baselines3")
        
        # Test project imports
        from src.trading_env import TradingEnv
        print("  ✅ src.trading_env")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False


def check_data_availability():
    """Check if 2025 data is available"""
    print("\nChecking 2025 backtest data...")
    
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    data_dir = "backtest/data"
    
    data_available = True
    
    for asset in assets:
        file_path = os.path.join(data_dir, f"{asset}_5m_2025.parquet")
        if os.path.exists(file_path):
            print(f"  ✅ {asset}_5m_2025.parquet")
        else:
            print(f"  ⚠️  {asset}_5m_2025.parquet - Not downloaded yet")
            data_available = False
    
    if not data_available:
        print("\n  💡 Run: python backtest/data_fetcher_backtest.py")
    
    return data_available


def check_models():
    """Check if trained models are available"""
    print("\nChecking for trained models...")
    
    models_dir = "models/checkpoints"
    
    if not os.path.exists(models_dir):
        print(f"  ⚠️  {models_dir} - Directory not found")
        print("  💡 Train models first using: python src/train.py")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    
    if model_files:
        print(f"  ✅ Found {len(model_files)} model(s):")
        for model in model_files[:5]:  # Show first 5
            print(f"     - {model}")
        if len(model_files) > 5:
            print(f"     ... and {len(model_files) - 5} more")
        return True
    else:
        print("  ⚠️  No trained models found")
        print("  💡 Train models first using: python src/train.py")
        return False


def main():
    """Run all validation checks"""
    print("="*60)
    print("BACKTESTING INFRASTRUCTURE VALIDATION")
    print("="*60)
    
    structure_ok = check_directory_structure()
    imports_ok = check_imports()
    data_ok = check_data_availability()
    models_ok = check_models()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print(f"Directory Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"Required Imports:    {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"2025 Data:           {'✅ READY' if data_ok else '⚠️  NOT READY'}")
    print(f"Trained Models:      {'✅ READY' if models_ok else '⚠️  NOT READY'}")
    
    print("\n" + "="*60)
    
    if structure_ok and imports_ok:
        print("✅ BACKTESTING INFRASTRUCTURE IS SET UP CORRECTLY!")
        
        if not data_ok:
            print("\n📋 NEXT STEP: Download 2025 data")
            print("   python backtest/data_fetcher_backtest.py")
        
        if not models_ok:
            print("\n📋 NEXT STEP: Train models")
            print("   python src/train.py --stage 1 --dry-run")
        
        if data_ok and models_ok:
            print("\n🚀 READY TO BACKTEST!")
            print("   python backtest/backtest.py --model models/checkpoints/[MODEL].zip --stage [1|2|3]")
    else:
        print("❌ SETUP INCOMPLETE - Please fix the issues above")
    
    print("="*60)


if __name__ == "__main__":
    main()

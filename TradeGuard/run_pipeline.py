import os
import sys
import logging
import multiprocessing
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from twisted.internet import reactor

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Imports
try:
    from TradeGuard.src.download_data import DataFetcherTraining
    from TradeGuard.src.generate_dataset import DatasetGenerator
    from TradeGuard.src.train_guard import DataLoader, ModelTrainer
    from TradeGuard.src.visualization import ModelVisualizer
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import modules. Ensure you are running from the project root or correct environment. {e}")
    sys.exit(1)

def run_pipeline():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("TradeGuard.Pipeline")
    
    # Paths
    # We assume we are running from project root, but use absolute paths to be safe
    BASE_DIR = PROJECT_ROOT
    DATA_DIR = BASE_DIR / "TradeGuard/data"
    DATASET_PATH = BASE_DIR / "TradeGuard/data/guard_dataset.parquet"
    MODEL_DIR = BASE_DIR / "TradeGuard/models"
    ALPHA_MODEL = BASE_DIR / "Alpha/models/checkpoints/8.03.zip" 
    
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Step 1: Download Data
    # ---------------------------------------------------------
    logger.info("="*60)
    logger.info(">>> STEP 1: Downloading Training Data")
    logger.info("="*60)
    
    # Check if data already exists to avoid unnecessary reactor startup/shutdown risk
    # (Though DataFetcherTraining handles existing files, skipping the reactor start entirely is safer if all files exist)
    required_assets = ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDCHF', 'USDJPY']
    all_exist = True
    for asset in required_assets:
        if not (DATA_DIR / f"{asset}_5m.parquet").exists():
            all_exist = False
            break
            
    if all_exist:
        logger.info("All data files appear to exist. Skipping download to preserve reactor state.")
    else:
        try:
            logger.info("Starting Data Fetcher...")
            # Note: twisted reactor cannot be restarted in the same process.
            # This must be the only time we run it.
            fetcher = DataFetcherTraining(output_dir=str(DATA_DIR))
            fetcher.start() 
            logger.info("Step 1 Complete.")
        except Exception as e:
            logger.error(f"Error in Step 1 (Download): {e}")
            # If download fails, we probably can't continue unless data exists
            if not all_exist:
                return

    # ---------------------------------------------------------
    # Step 2: Generate Dataset
    # ---------------------------------------------------------
    logger.info("="*60)
    logger.info(">>> STEP 2: Generating Dataset")
    logger.info("="*60)
    
    try:
        # Check if dataset already exists
        if DATASET_PATH.exists():
            logger.info(f"Dataset found at {DATASET_PATH}. Skipping generation.")
        else:
            # Check if Alpha model exists
            if not ALPHA_MODEL.exists():
                logger.error(f"Alpha model not found at {ALPHA_MODEL}. Cannot generate dataset.")
                return

            generator = DatasetGenerator(data_dir=str(DATA_DIR))
            generator.run(model_path=str(ALPHA_MODEL), output_path=str(DATASET_PATH))
            logger.info("Step 2 Complete.")
    except Exception as e:
        logger.error(f"Error in Step 2 (Dataset Gen): {e}")
        import traceback
        traceback.print_exc()
        return

    # ---------------------------------------------------------
    # Step 3: Train Model
    # ---------------------------------------------------------
    logger.info("="*60)
    logger.info(">>> STEP 3: Training TradeGuard Model")
    logger.info("="*60)
    
    try:
        if not DATASET_PATH.exists():
            logger.error(f"Dataset not found at {DATASET_PATH}. Stopping.")
            return

        loader = DataLoader(str(DATASET_PATH))
        trainer = ModelTrainer()
        viz = ModelVisualizer(str(MODEL_DIR))

        logger.info("Loading and splitting data...")
        dev_df, holdout_df = loader.get_train_val_split()
        
        if dev_df.empty or holdout_df.empty:
            logger.error("Data split resulted in empty dataframes. Check dataset date range.")
            return

        train_sub, tune_sub = loader.get_internal_tuning_split(dev_df)

        logger.info(f"Training Data: {len(train_sub)} rows")
        logger.info(f"Tuning Data: {len(tune_sub)} rows")
        logger.info(f"Holdout Data: {len(holdout_df)} rows")

        logger.info("Optimizing hyperparameters (LightGBM)...")
        best_params = trainer.optimize_hyperparameters(train_sub, tune_sub)

        logger.info("Training final model on full development set...")
        final_model = trainer.train_final_model(dev_df, best_params)

        logger.info("Optimizing decision threshold...")
        best_threshold, tuning_metrics = trainer.optimize_threshold(final_model, tune_sub, target_precision=0.6)
        logger.info(f"Optimal threshold: {best_threshold:.4f}")
        logger.info(f"Tuning Metrics: {tuning_metrics}")

        logger.info("Evaluating on Hold-out set (2024)...")
        holdout_metrics = trainer.evaluate_model(final_model, holdout_df, threshold=best_threshold)
        logger.info(f"Hold-out Metrics: {holdout_metrics}")

        logger.info("Saving artifacts...")
        viz.save_model(final_model)
        viz.save_metadata(holdout_metrics, best_threshold)
        
        logger.info("Generating performance plots...")
        drop_cols = ['label', 'asset', 'timestamp']
        X_holdout = holdout_df.drop(columns=[c for c in drop_cols if c in holdout_df.columns])
        y_holdout = holdout_df['label']
        y_prob = final_model.predict(X_holdout)
        y_pred = (y_prob >= best_threshold).astype(int)
        
        viz.plot_confusion_matrix(y_holdout, y_pred)
        viz.plot_feature_importance(final_model)
        viz.plot_calibration_curve(y_holdout, y_prob)
        viz.plot_roc_curve(y_holdout, y_prob)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE SUCCESS")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in Step 3 (Training): {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_pipeline()

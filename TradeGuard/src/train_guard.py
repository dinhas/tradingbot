import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import logging

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path


    def get_train_val_split(self):
        """
        Loads the dataset and splits it into training (Development) and validation (Hold-out) sets.
        
        Splitting Logic:
        - Training (Development): 2016-01-01 to 2023-12-31
        - Validation (Hold-out): 2024-01-01 to 2024-12-31
        
        Returns:
            tuple: (train_df, val_df)
        """
        # Load the data
        df = pd.read_parquet(self.file_path)
        
        # Ensure the index is a DatetimeIndex
        # If it's not in the index, look for a 'date' or 'timestamp' column
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                # If no date column and no DatetimeIndex, this is an issue.
                # However, for now we assume standard format.
                pass
        
        # Define split date
        split_date = pd.Timestamp('2024-01-01')
        
        # Split data
        train_df = df[df.index < split_date]
        val_df = df[df.index >= split_date]
        
        return train_df, val_df

    def get_internal_tuning_split(self, df: pd.DataFrame):
        """
        Splits the development set into internal training and tuning sets.
        
        Splitting Logic:
        - Internal Training: 2016-01-01 to 2021-12-31
        - Internal Tuning: 2022-01-01 to 2023-12-31
        
        Args:
            df (pd.DataFrame): The development set (2016-2023).
            
        Returns:
            tuple: (train_sub_df, tune_sub_df)
        """
        split_date = pd.Timestamp('2022-01-01')
        
        train_sub_df = df[df.index < split_date]
        tune_sub_df = df[df.index >= split_date]
        
        return train_sub_df, tune_sub_df

class ModelTrainer:
    def __init__(self):
        pass

    def optimize_hyperparameters(self, train_df: pd.DataFrame, tune_df: pd.DataFrame) -> dict:
        """
        Performs hyperparameter tuning using the provided training and tuning sets.
        
        Args:
            train_df (pd.DataFrame): Internal training set.
            tune_df (pd.DataFrame): Internal tuning set.
            
        Returns:
            dict: Best hyperparameters.
        """
        # Prepare datasets
        # Assuming 'label' is the target column
        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']
        X_tune = tune_df.drop(columns=['label'])
        y_tune = tune_df['label']
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_tune = lgb.Dataset(X_tune, label=y_tune, reference=lgb_train)
        
        # Define parameter grid
        param_grid = [
            {'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9},
            {'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.9},
            {'num_leaves': 31, 'learning_rate': 0.01, 'feature_fraction': 0.8},
        ]
        
        best_score = -float('inf')
        best_params = {}
        
        for params in param_grid:
            # Add static params
            current_params = params.copy()
            current_params.update({
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'verbose': -1,
                'seed': 42
            })
            
            # Train with early stopping
            model = lgb.train(
                current_params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_tune],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10),
                    lgb.log_evaluation(0)
                ]
            )
            
            # Evaluate (using the best iteration)
            preds = model.predict(X_tune, num_iteration=model.best_iteration)
            try:
                score = roc_auc_score(y_tune, preds)
            except ValueError:
                # Handle cases where only one class is present in y_tune
                score = 0.5
            
            if score > best_score:
                best_score = score
                best_params = current_params
        
        logging.info(f"Best parameters found: {best_params}")
        return best_params

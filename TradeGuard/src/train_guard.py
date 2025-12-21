import pandas as pd
from pathlib import Path

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

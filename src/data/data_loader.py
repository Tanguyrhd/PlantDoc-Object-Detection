"""
Data Loader
Handles loading and cleaning of CSV data files.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple


class DataLoader:
    """Loads and cleans data from CSV files."""

    @staticmethod
    def load_data(train_csv: Path, test_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data from CSV files.

        Args:
            train_csv: Path to training labels CSV
            test_csv: Path to test labels CSV

        Returns:
            Tuple of (train_df, test_df)
        """
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)

        print(f"✓ Loaded: {len(df_train['filename'].unique())} train images, {len(df_test['filename'].unique())} test images")

        return df_train, df_test

    @staticmethod
    def clean_class_names(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean class names by removing 'leaf', extra spaces, and underscores.

        Args:
            df: DataFrame with 'class' column

        Returns:
            DataFrame with cleaned class names
        """
        df = df.copy()
        df['class'] = (
            df['class']
            .str.replace(r'(?i)leaf', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.replace(r'_', ' ', regex=True)
            .str.strip()
        )
        return df

    @staticmethod
    def load_and_clean(train_csv: Path, test_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and clean data in one step.

        Args:
            train_csv: Path to training labels CSV
            test_csv: Path to test labels CSV

        Returns:
            Tuple of (cleaned_train_df, cleaned_test_df)
        """
        df_train, df_test = DataLoader.load_data(train_csv, test_csv)

        df_train = DataLoader.clean_class_names(df_train)
        df_test = DataLoader.clean_class_names(df_test)

        print("✓ Class names cleaned")

        return df_train, df_test

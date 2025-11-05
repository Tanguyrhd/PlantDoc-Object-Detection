"""
Data Validator
Validates data integrity (file existence, dimensions, etc.).
"""

import pandas as pd
from pathlib import Path
from PIL import Image


class DataValidator:
    """Validates and fixes data integrity issues."""

    @staticmethod
    def fix_zero_dimensions(df: pd.DataFrame, image_folder: Path, dataset_type: str) -> pd.DataFrame:
        """
        Fix rows with zero width or height by reading actual image dimensions.

        Args:
            df: DataFrame with image metadata
            image_folder: Path to folder containing images

        Returns:
            DataFrame with fixed dimensions
        """
        df = df.copy()
        image_folder = Path(image_folder)
        fixed_count = 0

        for idx, row in df.iterrows():
            if row['width'] == 0 or row['height'] == 0:
                image_path = image_folder / row['filename']
                if image_path.exists():
                    with Image.open(image_path) as img:
                        w, h = img.size
                        df.at[idx, 'width'] = w
                        df.at[idx, 'height'] = h
                        fixed_count += 1

        if fixed_count > 0:
            print(f"✓ Fixed {fixed_count} images with zero dimensions on the {dataset_type} dataset")

        return df

    @staticmethod
    def verify_files_exist(df: pd.DataFrame, image_folder: Path, dataset_type: str) -> pd.DataFrame:
        """
        Filter DataFrame to keep only rows where image files exist.

        Args:
            df: DataFrame with 'filename' column
            image_folder: Path to folder containing images

        Returns:
            Filtered DataFrame with only existing files
        """
        image_folder = Path(image_folder)
        existing_mask = []

        for _, row in df.iterrows():
            existing_mask.append((image_folder / row['filename']).exists())

        df_filtered = df[existing_mask].copy()
        removed_count = len(df['filename'].unique()) - len(df_filtered['filename'].unique())

        if removed_count > 0:
            print(f"⚠ Removed {removed_count} images with missing files on the {dataset_type} dataset")
            print(df[~df.index.isin(df_filtered.index)]['filename'].unique())
        return df_filtered

    @staticmethod
    def validate_and_fix(df: pd.DataFrame, image_folder: Path, dataset_type: str) -> pd.DataFrame:
        """
        Run all validation and fixing steps.

        Args:
            df: DataFrame to validate
            image_folder: Path to folder containing images

        Returns:
            Validated and fixed DataFrame
        """
        df = DataValidator.fix_zero_dimensions(df, image_folder, dataset_type)
        df = DataValidator.verify_files_exist(df, image_folder, dataset_type)

        print(f"✓ Validated: {len(df['filename'].unique())} images on the {dataset_type} dataset")

        return df

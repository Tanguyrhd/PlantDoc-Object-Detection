"""
Data Validation and Cleaning Utilities

This module provides pure functions for validating and cleaning image annotation datasets
before YOLO format conversion.

Main Operations:
----------------
- clean_class_column: Remove "leaf" substring, normalize spaces and underscores
- fix_zero_dimensions: Read actual image dimensions for rows with zero width/height
- verify_files_exist: Filter out annotations with missing image files
- balance_by_column: Balance dataset by duplicating minority class samples

Usage:
------
These functions are typically used in the BasePipeline data preparation workflow:
    1. Load raw CSV annotations
    2. Clean class names (clean_class_column)
    3. Fix metadata issues (fix_zero_dimensions)
    4. Validate file existence (verify_files_exist)
    5. Optional balancing (balance_by_column)

All functions are stateless and return modified DataFrames without side effects.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def clean_class_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean class names in a DataFrame.
        --> removes "leaf", normalise espaces/underscores

    Args:
        df (pd.DataFrame): DataFrame with 'class' column

    Returns:
        pd.DataFrame: DataFrame with cleaned class names
    """
    df['class'] = (
        df['class']
        .str.replace(r'(?i)leaf', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.replace(r'_', ' ', regex=True)
        .str.strip()
    )
    return df

def fix_zero_dimensions(df: pd.DataFrame, image_folder: Path, dataset_type: str) -> pd.DataFrame:
    """
    Fix rows with zero width or height by reading actual image dimensions.

    Args:
        df: DataFrame with image metadata
        image_folder: Path to folder containing images
        dataset_type: train or test

    Returns:
        DataFrame with fixed dimensions
    """
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
        logger.info(f"✓ Fixed {fixed_count} rows with zero dimensions on the {dataset_type} dataset")
    else:
        logger.info(f"✓ No rows with zero dimensions on the {dataset_type} dataset")

    return df

def verify_files_exist(df: pd.DataFrame, image_folder: Path, dataset_type: str) -> pd.DataFrame:
    """
    Filter DataFrame to keep only rows where image files exist.

    Args:
        df: DataFrame with 'filename' column
        image_folder: Path to folder containing images
        dataset_type: train or test

    Returns:
        Filtered DataFrame with only existing files
    """
    existing_mask = []

    for _, row in df.iterrows():
        existing_mask.append((image_folder / row['filename']).exists())

    df_filtered = df[existing_mask].copy()
    removed_count_image = len(df['filename'].unique()) - len(df_filtered['filename'].unique())
    removed_count_rows = len(df['filename']) - len(df_filtered['filename'])

    if removed_count_image > 0:
        logger.info(f"Removed {removed_count_image} image with missing files on the {dataset_type} dataset")
        logger.info(f"(correspond to {removed_count_rows} rows on the {dataset_type} dataset)")
        logger.info(df[~df.index.isin(df_filtered.index)]['filename'].unique())
    else:
        logger.info(f"✓ No missing files on the {dataset_type} dataset")

    return df_filtered

def balance_by_column(
        df: pd.DataFrame,
        column: str,
        target_samples_per_class: int,
        keep_above_target: bool = True
    ) -> pd.DataFrame:
    """
    Balance dataset by duplicating samples to reach target per class.

    Args:
        df: DataFrame to balance
        column: Column name to group by (e.g., 'species', 'disease')
        target_samples_per_class: Target number of samples per class
        keep_above_target: If True, keep all samples for classes above target.
                            If False, downsample to target.

    Returns:
        Balanced DataFrame
    """
    balanced_dfs = []

    for class_value, group in df.groupby(column):
        n_samples = len(group)
        n_to_add = target_samples_per_class - n_samples

        if n_to_add > 0:
            logger.info(f"  {class_value}: {n_samples} → {target_samples_per_class} "
                    f"(adding {n_to_add} duplicates)")

            # Keep original samples
            balanced_dfs.append(group)

            # Add duplicates with modified filenames
            duplicates_added = 0
            while duplicates_added < n_to_add:
                # Cycle through samples
                idx = duplicates_added % n_samples
                sample = group.iloc[idx:idx+1].copy()

                # Modify filename to avoid conflicts
                original_filename = sample['filename'].values[0]
                stem = Path(original_filename).stem
                suffix = Path(original_filename).suffix
                new_filename = f"{stem}_dup{duplicates_added}{suffix}"
                sample.loc[:, 'filename'] = new_filename

                balanced_dfs.append(sample)
                duplicates_added += 1

        else:
            if keep_above_target:
                logger.info(f"  {class_value}: {n_samples} (already >= target, keeping all)")
                balanced_dfs.append(group)
            else:
                logger.info(f"  {class_value}: {n_samples} → {target_samples_per_class} "
                        f"(downsampling)")
                balanced_dfs.append(group.iloc[:target_samples_per_class])

    df_balanced = pd.concat(balanced_dfs, ignore_index=True)

    logger.info(f"\n✓ Dataset balanced! Total samples: {len(df_balanced)}")

    return df_balanced

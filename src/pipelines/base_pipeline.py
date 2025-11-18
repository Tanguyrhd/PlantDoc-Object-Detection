from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
import re

from ..config import PipelineConfig
from ..data import DataLoader, DataValidator, FeatureExtractor
from ..processing import DataBalancer, YOLOConverter

import logging

logger = logging.getLogger(__name__)


"""
To Do here:
- Change the pass in the abstract classes to NotImplementedError
    DONE - CHECK if there are all pertinant

"""

"""
Structure changes:

load_and_prepare_data:
- Would separate this into a load method, a prepare (or process) method and then a wrapper method
    - Gives you the flexibility to change the loading or the processing without affecting everything else
    - Makes it easier to understand
    - Single responsibility methods

Abstract methods
- Make sure that those are methods that will be used by all of the child classes. (I have not checked... just saying. at first glance it seems okay)

yolo_converter
- Funny, but would name it something more explicit. What makes it yolo?

get_pipeline_type
- Could be a __init__ arg instead of a method

get_class_column
- Could be a __init__ arg instead of a method

"""
class BasePipeline(ABC):
    """
    Abstract base class for all pipelines.
    Set the standard method to modify raw data into specific structure to train a YOLO model

        raw :
            --> Dataset/train/image.jpeg and image.xml
            --> Dataset/test/image.jpeg and image.xml
            --> Dataset/train_labels.csv
            --> Dataset/test_labels.csv

        final :
            --> Dataset/model_name/images/train/image.jpeg
            --> Dataset/model_name/images/val/image.jpeg
            --> Dataset/model_name/labels/train/image.txt
            --> Dataset/model_name/labels/val/image.txt
            --> Dataset/model_name/dataset.yaml

    Parameters
    ----------
    config : Pipeline configuration object

    df_train : Training Dataframe

    df_test : Testing Dataframe

    df_train_processed : Training processed Dataframe

    df_train_processed : Training processed Dataframe

    """

    def __init__(
        self,
        config: PipelineConfig
        ):

        self.config = config

    def load_data(self):
        """
        Load training and test data from CSV files and store as instance attributes.

        Sets:
            self.df_train: Training labels DataFrame
            self.df_test: Test labels DataFrame
        """

        self.df_train = pd.read_csv(self.config.train_labels_csv)
        self.df_test = pd.read_csv(self.config.test_labels_csv)

        logger.info(
            f"✓ Loaded: train_labels_csv with {len(self.df_train)} rows, "
            f"and test_labels_csv with {len(self.df_test)} rows")

    def _clean_class_column(self, df):
        """
        Clean class names in a DataFrame.

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

    def clean_data(self):
        """
        Clean class names by removing 'leaf', extra spaces, and underscores.

        Modifies:
            self.df_train['class']: Cleaned class names
            self.df_test['class']: Cleaned class names
        """
        self.df_train = self._clean_class_column(self.df_train)
        self.df_test = self._clean_class_column(self.df_test)

        logger.info(
            f"Cleaned class names for train ({len(self.df_train)} rows) "
            f"and test ({len(self.df_test)} rows)"
        )

    def _fix_zero_dimensions(self, df: pd.DataFrame, image_folder: Path, dataset_type: str):
        """
        Fix rows with zero width or height by reading actual image dimensions.

        Args:
            df: DataFrame with image metadata
            image_folder: Path to folder containing images

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

    def _verify_files_exist(self, df: pd.DataFrame, image_folder: Path, dataset_type: str):
        """
        Filter DataFrame to keep only rows where image files exist.

        Args:
            df: DataFrame with 'filename' column
            image_folder: Path to folder containing images

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
            logger.info(f"⚠ Removed {removed_count_image} image with missing files on the {dataset_type} dataset")
            logger.info(f"(correspond to {removed_count_rows} rows on the {dataset_type} dataset)")
            logger.info(df[~df.index.isin(df_filtered.index)]['filename'].unique())
        else:
            logger.info(f"✓ No missing files on the {dataset_type} dataset")

        return df_filtered

    def verify_and_fix(self):
        """
        Fix dimension problem and verify if the file exist (ifnot, delete the lines)

        Modifies:
            self.df_train
            self.df_test
        """
        self.df_train = self._fix_zero_dimensions(self.df_train, self.config.train_images_dir, "train")
        self.df_train = self._verify_files_exist(self.df_train, self.config.train_images_dir, "train")

        self.df_test = self._fix_zero_dimensions(self.df_test, self.config.test_images_dir, "test")
        self.df_test = self._verify_files_exist(self.df_test, self.config.test_images_dir, "test")

        logger.info(
            f"✓ Validated: {len(self.df_train)} rows on the train dataset "
            f"and {len(self.df_test)} rows on the test dataset")

    def _extract_species(self, text: str):
        """
        Extract plant species from text.

        Args:
            text: Class label text

        Returns:
            Species name or None if not found
        """
        for plant in self.config.plant_species:
            if re.search(rf"\b{plant}\b", text, flags=re.IGNORECASE):
                return plant
        return None

    def _extract_disease(self, text: str):
        """
        Extract disease name from text by removing species name.

        Args:
            text: Class label text

        Returns:
            Disease name or "healthy" if no disease
        """
        for plant in self.config.plant_species:
            text = re.sub(rf"\b{plant}\b", "", text, flags=re.IGNORECASE).strip()

        # Normalize to title case to avoid duplicates
        return text.title() if text else "healthy"

    def add_features(self):
        """
        Add 'species' and 'disease' columns to train and test DataFrame.

        Modified:
            self.df_train
            self.df_test
        """
        self.df_train['species'] = self.df_train['class'].apply(self._extract_species)
        self.df_train['disease'] = self.df_train['class'].apply(self._extract_disease)

        self.df_test['species'] = self.df_test['class'].apply(self._extract_species)
        self.df_test['disease'] = self.df_test['class'].apply(self._extract_disease)

        logger.info(f"✓ Features extracted for the train and test dataset (species, disease)")

    def load_clean_extract_fix_verify(self):
        """
        Load, clean, extract features, fix, and validate data pipeline.

        This method orchestrates the complete data preparation workflow:
        1. Load train and test CSV files
        2. Clean class names
        3. Extract features by creating two columns : species and diseases
        4. Fix zero dimensions by reading actual images
        5. Verify and remove rows with missing image files

        Modifies:
            self.df_train: Loaded, cleaned, and validated training DataFrame
            self.df_test: Loaded, cleaned, and validated test DataFrame
        """

        logger.info(f"\n{'='*60}")
        logger.info(f"LOADING AND PREPARING DATA")
        logger.info(f"{'='*60}\n")

        # Step 1: Load data
        self.load_data()

        # Step 2: Clean class names
        self.clean_data()

        # Step 3 : Extract features
        self.add_features()

        # Step 3: Fix dimensions and verify files
        self.verify_and_fix()

        logger.info(f"\n{'='*60}")
        logger.info(f"DATA PREPARATION COMPLETE")
        logger.info(f"{'='*60}\n")

    @abstractmethod
    def filter_data(self):
        """
        Filter data based on pipeline-specific criteria.
        """
        raise NotImplementedError("Subclasses must implement the 'filter_data' method.")

    @abstractmethod
    def balance_data(self):
        """
        Balance data for training.
        """
        raise NotImplementedError("Subclasses must implement the 'balance_data' method.")

    @abstractmethod
    def get_class_column(self) -> str:
        """
        Get the column name used for classification.

        Returns:
            Column name (e.g., 'binary_class', 'species', 'disease')
        """
        raise NotImplementedError("Subclasses must implement the 'get_class_column' method.")

    @abstractmethod
    def get_pipeline_type(self) -> str:
        """
        Get pipeline type identifier.

        Returns:
            Pipeline type ('binary', 'species', or 'disease')
        """
        raise NotImplementedError("Subclasses must implement the 'get_pipeline_type' method.")

    def create_class_mapping(self) -> Dict[str, int]:
        """
        Create class-to-index mapping.

        Returns:
            Dictionary mapping class names to indices
        """
        class_column = self.get_class_column()
        return self.yolo_converter.create_class_mapping(
            self.df_train_processed,
            class_column
        )

    def export_data(self):
        """Export processed data to YOLO format."""
        print(f"\n{'='*60}")
        print(f"EXPORTING TO YOLO FORMAT")
        print(f"{'='*60}\n")

        # Get output paths
        output_paths = self.config.get_output_paths(self.get_pipeline_type())

        # Create class mapping
        class_column = self.get_class_column()
        class_mapping = self.create_class_mapping()

        # Export training data
        print(f"\nExporting TRAINING data...")
        exported_train, skipped_train = self.yolo_converter.export_to_yolo(
            df=self.df_train_processed,
            source_images_dir=self.config.train_images_dir,
            output_images_dir=output_paths['images_train'],
            output_labels_dir=output_paths['labels_train'],
            class_mapping=class_mapping,
            class_column=class_column
        )
        print(f"✓ Exported: {exported_train} images, Skipped: {skipped_train}")

        # Export validation data
        print(f"\nExporting VALIDATION data...")
        exported_val, skipped_val = self.yolo_converter.export_to_yolo(
            df=self.df_test_processed,
            source_images_dir=self.config.test_images_dir,
            output_images_dir=output_paths['images_val'],
            output_labels_dir=output_paths['labels_val'],
            class_mapping=class_mapping,
            class_column=class_column
        )
        print(f"✓ Exported: {exported_val} images, Skipped: {skipped_val}")

        # Create YAML config
        yaml_path = self.yolo_converter.create_yaml_config(
            output_dir=output_paths['base_dir'],
            class_mapping=class_mapping
        )

        print(f"\n{'='*60}")
        print(f"EXPORT COMPLETE")
        print(f"{'='*60}")
        print(f"Training: {exported_train} images")
        print(f"Validation: {exported_val} images")
        print(f"Config: {yaml_path}")
        print(f"{'='*60}\n")

    def run(self):
        """
        Run the complete pipeline from start to finish.
        This is the main entry point for executing a pipeline.
        """
        print(f"\n{'#'*60}")
        print(f"RUNNING {self.get_pipeline_type().upper()} PIPELINE")
        print(f"{'#'*60}\n")

        # Step 1: Load and prepare
        self.load_and_prepare_data()

        # Step 2: Filter (pipeline-specific)
        self.filter_data()

        # Step 3: Balance (pipeline-specific)
        self.balance_data()

        # Step 4: Export
        self.export_data()

        print(f"\n{'#'*60}")
        print(f"{self.get_pipeline_type().upper()} PIPELINE COMPLETE")
        print(f"{'#'*60}\n")

from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from PIL import Image
import re

from ..config import PipelineConfig
from ..processing.yolo_converter import (
    create_class_mapping,
    export_to_yolo,
    create_yaml_config
)
from ..processing.data_validation import (
    clean_class_column,
    fix_zero_dimensions,
    verify_files_exist,
    balance_by_column
)

import logging

logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    """
    Abstract base class for YOLO model data preparation pipelines.

    Transforms raw annotated images into YOLO-compatible datasets through:
    1. Loading and cleaning CSV labels
    2. Extracting features (species, disease)
    3. Validating image existence and dimensions
    4. Optional class balancing
    5. Exporting to YOLO format

    Raw Data Structure:
        Dataset/train/{image.jpeg, image.xml}
        Dataset/test/{image.jpeg, image.xml}
        Dataset/{train_labels.csv, test_labels.csv}

    Output Structure:
        Dataset/{model_name}/images/{train,val}/image.jpeg
        Dataset/{model_name}/labels/{train,val}/image.txt
        Dataset/{model_name}/dataset.yaml

    Args:
        config: Pipeline configuration containing paths and parameters

    Attributes:
        config: PipelineConfig instance
        df_train: Training labels DataFrame (set by load_data)
        df_test: Test labels DataFrame (set by load_data)
        df_train_processed: Processed training data (set by filter_data/balance_data)
        df_test_processed: Processed test data (set by filter_data/balance_data)
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

    def clean_data(self):
        """
        Clean class names by removing 'leaf', extra spaces, and underscores.
        """
        self.df_train = clean_class_column(self.df_train)
        self.df_test = clean_class_column(self.df_test)

        logger.info(
            f"Cleaned class names for train ({len(self.df_train)} rows) "
            f"and test ({len(self.df_test)} rows)"
        )

    def verify_and_fix(self):
        """
        Fix zero dimensions and verify files exist, removing invalid rows.
        """
        self.df_train = fix_zero_dimensions(self.df_train, self.config.train_images_dir, "train")
        self.df_train = verify_files_exist(self.df_train, self.config.train_images_dir, "train")

        self.df_test = fix_zero_dimensions(self.df_test, self.config.test_images_dir, "test")
        self.df_test = verify_files_exist(self.df_test, self.config.test_images_dir, "test")

        logger.info(
            f"✓ Validated: {len(self.df_train)} rows on the train dataset "
            f"and {len(self.df_test)} rows on the test dataset")

    def _extract_species(self, text: str):
        """
        Extract plant species name from class label text.

        Searches for any species name from config.plant_species using case-insensitive
        word boundary matching.

        Args:
            text: Class label text (e.g., "Tomato Early Blight")

        Returns:
            Matched species name from config.plant_species, or None if not found
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
            Disease name normalized to Title case, or "healthy" if no disease
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

    def balance_data(self, interactive: bool = True) -> None:
        """
        Balance the dataset by letting user choose to balance with a specific target or keep the natural balance

        Args:
            interactive: If True, ask user for target samples. If False, use default.
        """
        logger.info(f"\n{'='*60}")
        logger.info("PREPARING DATASETS")
        logger.info(f"{'='*60}\n")

        # Ask user for balancing choice

        distribution = self.df_train[self.class_column].value_counts().sort_index()
        apply_balancing = False

        if interactive:
            logger.info(f"\n{'-'*60}")
            logger.info("BALANCING OPTIONS")
            logger.info(f"{'-'*60}")
            logger.info("Do you want to balance the training dataset?")
            logger.info("  1. Yes, with custom target")
            logger.info("  2. No, keep natural distribution")

            while True:
                choice = input("\nMake a choice between 1 and 2: ").strip()

                if choice == "1":
                    apply_balancing = True
                    while True:
                        try:
                            target_samples = int(input("Enter target samples per class: "))
                            max_possible = distribution.min()
                            if target_samples > max_possible*2:
                                logger.info(f"⚠️  Warning: We recomand to maximum double the size of the minority class ({max_possible * 2})")
                                confirm = input(f"Continue with {target_samples}? (y/n): ").strip().lower()
                                if confirm == 'y':
                                    break
                            elif target_samples > 0:
                                break
                            else:
                                logger.info("⚠️  Please enter a positive number")
                        except ValueError:
                            logger.info("⚠️  Please enter a valid number")
                    break
                elif choice == '2':
                    apply_balancing = False
                    break
                else:
                    logger.info("⚠️  Please enter a valid choice between 1 and 2")

        # Apply balancing if requested
        if apply_balancing:
            logger.info(f"\n Balancing training dataset to {target_samples} samples per class...")
            self.df_train = balance_by_column(
                self.df_train,
                column=self.class_column,
                target_samples_per_class=target_samples
            )

            # Show new distribution
            new_distribution = self.df_train[self.class_column].value_counts().sort_index()
            logger.info("\n📊 Training set - Balanced distribution:")
            for label, count in new_distribution.items():
                label_name = "Healthy" if label == 0 else "Diseased"
                percentage = (count / len(self.df_train)) * 100
                logger.info(f"  {label_name:12} (label {label}): {count:5} samples ({percentage:5.1f}%)")

            logger.info(f"\n  Total training: {len(self.df_train)} samples")
            logger.info("✓ Training dataset balanced successfully")
        else:
            logger.info("\n✓ Keeping natural distribution (no balancing)")

    @abstractmethod
    def filter_data(self):
        """
        Filter data based on pipeline-specific criteria.
        """
        raise NotImplementedError("Subclasses must implement the 'filter_data' method.")

    def export_data(self):
        """Export processed data to YOLO format."""
        logger.info(f"\n{'='*60}")
        logger.info(f"EXPORTING TO YOLO FORMAT")
        logger.info(f"{'='*60}\n")

        # Get output paths
        output_paths = self.config.get_output_paths(self.pipeline_type)

        # Create class mapping
        class_column = self.class_column
        class_mapping = create_class_mapping(self.df_train, class_column)

        # Export training data
        logger.info(f"\nExporting TRAINING data...")
        exported_train, skipped_train = export_to_yolo(
            df=self.df_train,
            source_images_dir=self.config.train_images_dir,
            output_images_dir=output_paths['images_train'],
            output_labels_dir=output_paths['labels_train'],
            class_mapping=class_mapping,
            class_column=class_column
        )
        logger.info(f"✓ Exported: {exported_train} images, Skipped: {skipped_train}")

        # Export validation data
        logger.info(f"\nExporting VALIDATION data...")
        exported_val, skipped_val = export_to_yolo(
            df=self.df_test,
            source_images_dir=self.config.test_images_dir,
            output_images_dir=output_paths['images_val'],
            output_labels_dir=output_paths['labels_val'],
            class_mapping=class_mapping,
            class_column=class_column
        )
        logger.info(f"✓ Exported: {exported_val} images, Skipped: {skipped_val}")

        # Create YAML config
        yaml_path = create_yaml_config(
            output_dir=output_paths['base_dir'],
            class_mapping=class_mapping
        )

        logger.info(f"\n{'='*60}")
        logger.info(f"EXPORT COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Training: {exported_train} images")
        logger.info(f"Validation: {exported_val} images")
        logger.info(f"Config: {yaml_path}")
        logger.info(f"{'='*60}\n")

    def run(self):
        """
        Run the complete pipeline from start to finish.
        This is the main entry point for executing a pipeline.
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"RUNNING {self.pipeline_type.upper()} PIPELINE")
        logger.info(f"{'#'*60}\n")

        # Step 1: Load and prepare
        self.load_clean_extract_fix_verify()

        # Step 2: Filter (pipeline-specific)
        self.filter_data()

        # Step 3: Balance (pipeline-specific)
        self.balance_data()

        # Step 4: Export
        self.export_data()

        logger.info(f"\n{'#'*60}")
        logger.info(f"{self.pipeline_type.upper()} PIPELINE COMPLETE")
        logger.info(f"{'#'*60}\n")

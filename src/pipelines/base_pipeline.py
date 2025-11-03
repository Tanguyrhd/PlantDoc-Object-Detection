"""
Base Pipeline
Abstract base class for all data processing pipelines.
"""

from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

from ..config import PipelineConfig
from ..data import DataLoader, DataValidator, FeatureExtractor
from ..processing import DataBalancer, YOLOConverter


class BasePipeline(ABC):
    """Abstract base class for all pipelines."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize base pipeline.

        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.feature_extractor = FeatureExtractor(config.plant_species)
        self.balancer = DataBalancer()
        self.yolo_converter = YOLOConverter()

        # DataFrames (to be set by subclasses)
        self.df_train: pd.DataFrame = None
        self.df_test: pd.DataFrame = None
        self.df_train_processed: pd.DataFrame = None
        self.df_test_processed: pd.DataFrame = None

    def load_and_prepare_data(self):
        """Load, clean, extract features, and validate data."""
        print(f"\n{'='*60}")
        print(f"LOADING AND PREPARING DATA")
        print(f"{'='*60}\n")

        # Load and clean
        self.df_train, self.df_test = self.data_loader.load_and_clean(
            self.config.train_labels_csv,
            self.config.test_labels_csv
        )

        # Extract features
        self.df_train = self.feature_extractor.add_features(self.df_train, "train")
        self.df_test = self.feature_extractor.add_features(self.df_test, "test")

        # Validate
        self.df_train = self.data_validator.validate_and_fix(
            self.df_train,
            self.config.train_images_dir,
            "train"
        )
        self.df_test = self.data_validator.validate_and_fix(
            self.df_test,
            self.config.test_images_dir,
            "test"
        )

        print(f"\n✓ Data preparation complete")

    @abstractmethod
    def filter_data(self):
        """
        Filter data based on pipeline-specific criteria.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def balance_data(self):
        """
        Balance data for training.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_class_column(self) -> str:
        """
        Get the column name used for classification.
        Must be implemented by subclasses.

        Returns:
            Column name (e.g., 'binary_class', 'species', 'disease')
        """
        pass

    @abstractmethod
    def get_pipeline_type(self) -> str:
        """
        Get pipeline type identifier.
        Must be implemented by subclasses.

        Returns:
            Pipeline type ('binary', 'species', or 'disease')
        """
        pass

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

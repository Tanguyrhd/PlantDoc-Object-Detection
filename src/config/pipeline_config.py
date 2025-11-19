"""
Pipeline Configuration Module

Centralized configuration management for YOLO data preparation pipelines.
Loads dataset paths and parameters from environment variables.

This module requires a .env file at the project root with the following structure:

    TRAIN_LABELS_CSV=dataset/train_labels.csv
    TEST_LABELS_CSV=dataset/test_labels.csv
    TRAIN_IMAGES_DIR=dataset/train
    TEST_IMAGES_DIR=dataset/test
    PLANT_SPECIES=Tomato,Potato,Pepper,Apple,Grape

Usage:
------
    from src.config import PipelineConfig

    config = PipelineConfig()
    pipeline = BinaryPipeline(config)
    pipeline.run()
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv


class PipelineConfig:
    """
    Central configuration manager for YOLO data preparation pipelines.

    This class loads configuration from environment variables (.env file) and provides
    centralized access to dataset paths, plant species list, and output directories.

    Required Environment Variables (.env):
    --------------------------------------
    - TRAIN_LABELS_CSV: Path to training labels CSV
    - TEST_LABELS_CSV: Path to test labels CSV
    - TRAIN_IMAGES_DIR: Directory containing training images
    - TEST_IMAGES_DIR: Directory containing test images
    - PLANT_SPECIES: Comma-separated list of plant species (e.g., "Tomato,Potato,Apple")

    Attributes:
    -----------
    Input paths:
        train_labels_csv: Absolute path to training labels CSV
        test_labels_csv: Absolute path to test labels CSV
        train_images_dir: Absolute path to training images directory
        test_images_dir: Absolute path to test images directory

    Configuration:
        plant_species: List of plant species names for feature extraction
        rare_disease_threshold: Threshold for filtering rare diseases (default: 0.001)
        excluded_diseases: List of disease names to exclude from training

    Output directories:
        output_base_dir: Base directory for all outputs (project_root/dataset)
        binary_output_dir: Binary classification output
        species_output_dir: Species classification output
        disease_output_dir: Disease classification output

    Example:
    --------
        config = PipelineConfig()
        print(config.train_images_dir)  # /path/to/project/dataset/train
        paths = config.get_output_paths('binary')
        print(paths['images_train'])  # /path/to/project/dataset/binary/images/train
    """

    def __init__(self, project_root: Path = None):
        """
        Initialize pipeline configuration from environment variables.

        Automatically loads .env file from project root and initializes all paths
        and configuration parameters.

        Args:
            project_root: Root directory of the project. If None, auto-detects by
                        navigating up from src/config/ directory.

        Raises:
            ValueError: If required environment variables are missing
            FileNotFoundError: If .env file is not found (warning only)

        Side Effects:
            - Loads environment variables from .env file via dotenv
            - Creates absolute paths from relative paths in .env
        """
        # Load environment variables
        load_dotenv()

        # Set project root
        if project_root is None:
            # Auto-detect: go up from src/config to project root
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        # Input paths
        self.train_labels_csv = self._make_absolute(os.getenv('TRAIN_LABELS_CSV'))
        self.test_labels_csv = self._make_absolute(os.getenv('TEST_LABELS_CSV'))
        self.train_images_dir = self._make_absolute(os.getenv('TRAIN_IMAGES_DIR'))
        self.test_images_dir = self._make_absolute(os.getenv('TEST_IMAGES_DIR'))

        # Plant species
        self.plant_species: List[str] = [
            s.strip() for s in os.getenv('PLANT_SPECIES', '').split(',')
        ]

        # Output base directory
        self.output_base_dir = self.project_root / 'dataset'

        # Pipeline-specific output directories
        self.binary_output_dir = self.output_base_dir / 'binary'
        self.species_output_dir = self.output_base_dir / 'species'
        self.disease_output_dir = self.output_base_dir / 'diseases'

        # Disease filtering settings
        self.rare_disease_threshold = 0.001  # 0.1%
        self.excluded_diseases = ['Blight', 'Mold', 'Spot', 'Black Rot', 'Gray Spot']

    def _make_absolute(self, path_str: str) -> Path:
        """
        Convert relative path from .env to absolute path.

        Resolves paths relative to project_root and normalizes absolute paths.

        Args:
            path_str: Path string from environment variable

        Returns:
            Absolute Path object resolved from project root

        Raises:
            ValueError: If path_str is empty or None
        """
        if not path_str:
            raise ValueError("Path string is empty")

        path = Path(path_str)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def get_output_paths(self, pipeline_type: str) -> dict:
        """
        Get standardized output directory structure for a specific pipeline.

        Args:
            pipeline_type: Pipeline identifier ('binary', 'species', or 'disease')

        Returns:
            Dictionary with the following keys:
                - 'base_dir': Base directory for this pipeline
                - 'images_train': Training images directory
                - 'images_val': Validation images directory
                - 'labels_train': Training labels directory
                - 'labels_val': Validation labels directory
                - 'yaml_path': Path to dataset.yaml configuration file

        Raises:
            ValueError: If pipeline_type is not recognized

        Example:
            >>> config = PipelineConfig()
            >>> paths = config.get_output_paths('binary')
            >>> paths['images_train']
            PosixPath('/path/to/dataset/binary/images/train')
        """

        if pipeline_type == 'binary':
            base_dir = self.binary_output_dir
        elif pipeline_type == 'species':
            base_dir = self.species_output_dir
        elif pipeline_type == 'disease':
            base_dir = self.disease_output_dir
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")

        return {
            'base_dir': base_dir,
            'images_train': base_dir / 'images' / 'train',
            'images_val': base_dir / 'images' / 'val',
            'labels_train': base_dir / 'labels' / 'train',
            'labels_val': base_dir / 'labels' / 'val',
            'yaml_path': base_dir / 'dataset.yaml'
        }

    def __repr__(self) -> str:
        return (
            f"PipelineConfig(\n"
            f"  project_root={self.project_root}\n"
            f"  train_images={self.train_images_dir}\n"
            f"  plant_species={len(self.plant_species)} species\n"
            f")"
        )

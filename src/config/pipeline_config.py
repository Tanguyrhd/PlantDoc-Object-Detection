"""
Pipeline Configuration
Centralized configuration for all data pipelines.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv


class PipelineConfig:
    """Central configuration for all pipelines."""

    def __init__(self, project_root: Path = None):
        """
        Initialize pipeline configuration.

        Args:
            project_root: Root directory of the project. If None, auto-detect.
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

        Args:
            path_str: Path string from environment variable

        Returns:
            Absolute Path object
        """
        if not path_str:
            raise ValueError("Path string is empty")

        path = Path(path_str)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def get_output_paths(self, pipeline_type: str) -> dict:
        """
        Get output paths for a specific pipeline type.

        Args:
            pipeline_type: One of 'binary', 'species', or 'disease'

        Returns:
            Dictionary with output paths
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

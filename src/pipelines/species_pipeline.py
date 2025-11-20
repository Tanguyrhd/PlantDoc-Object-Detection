"""
Species Multi-Class Classification Pipeline

Pipeline for training YOLO models to classify different plant species.
Includes both healthy and diseased plants from all species.

Filtering Strategy:
-------------------
1. Remove samples with missing species information (species == None)
2. Keep all samples regardless of disease status (healthy or diseased)
3. Train on species identification across all plant types

This pipeline is useful for identifying which plant species is present in an image,
regardless of whether the plant is healthy or diseased.

Example:
--------
    config = PipelineConfig()
    pipeline = SpeciesPipeline(config)
    pipeline.run()  # Interactive mode with balancing options

    # Output: dataset/species/ with samples grouped by plant species
"""

import logging

logger = logging.getLogger(__name__)

import pandas as pd
from .base_pipeline import BasePipeline
from ..config import PipelineConfig

class SpeciesPipeline(BasePipeline):
    """
    Multi-class species classification pipeline.

    Extends BasePipeline to create a species classifier that identifies plant species
    regardless of health status (includes both healthy and diseased samples).

    Filtering includes:
    - Removal of samples with missing species information (species == None)
    - Keeps all disease states (healthy and diseased plants)

    Args:
        config: Pipeline configuration with species list
        pipeline_type: Pipeline identifier for output paths (default: 'species')
        class_column: DataFrame column name for species labels (default: 'species')

    Attributes:
        pipeline_type: String identifier ('species')
        class_column: Column containing species class names

    Example:
        >>> config = PipelineConfig()
        >>> config.plant_species = ['Tomato', 'Potato', 'Pepper']
        >>> pipeline = SpeciesPipeline(config)
        >>> pipeline.run()
    """

    def __init__(
            self,
            config: PipelineConfig,
            pipeline_type='species',
            class_column='species'):

        super().__init__(config)
        self.pipeline_type = pipeline_type
        self.class_column = class_column

    def filter_data(self):
        """
        Filter dataset to keep only samples with valid species information.

        Removes samples where species extraction failed (species == None), which can
        occur when the class label doesn't match any species in config.plant_species.

        Modifies:
            self.df_train: Filtered to samples with valid species only
            self.df_test: Filtered to samples with valid species only

        Side Effects:
            - Logs filtering statistics (kept vs removed samples)
            - Logs species distribution to console

        Example:
            Before: 10,000 samples (9,500 with species + 500 with species=None)
            After:  9,500 samples (only samples with identified species)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"FILTERING BY SPECIES")
        logger.info(f"{'='*60}\n")

        # Keep only samples with valid species
        df_train_filtered = self.df_train[self.df_train['species'].notna()].copy()
        df_test_filtered = self.df_test[self.df_test['species'].notna()].copy()

        removed_train = len(self.df_train) - len(df_train_filtered)
        removed_test = len(self.df_test) - len(df_test_filtered)

        logger.info(f"Training: {len(df_train_filtered)} samples (removed {removed_train})")
        logger.info(f"Validation: {len(df_test_filtered)} samples (removed {removed_test})")

        # Update dataframes
        self.df_train = df_train_filtered
        self.df_test = df_test_filtered

        # Show species distribution
        counts = df_train_filtered['species'].value_counts().sort_index()
        for label, count in counts.items():
            percentage = (count / len(df_train_filtered)) * 100
            logger.info(f"  {label:12}: {count:5} samples ({percentage:5.1f}%)")

        logger.info(f"\n  Total training: {len(df_train_filtered)} samples")

        logger.info(f"\n✓ Filtering complete")

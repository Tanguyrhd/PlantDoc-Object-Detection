"""
Disease Multi-Class Classification Pipeline

Pipeline for training YOLO models to classify different plant diseases.
Excludes healthy plants and filters out rare/problematic diseases based on
configuration thresholds.

Filtering Strategy:
-------------------
1. Remove all healthy samples (disease == 'healthy')
2. Remove rare diseases below threshold (default: 0.1% of dataset)
3. Remove manually excluded diseases from config.excluded_diseases
4. Train only on remaining disease classes

This pipeline is useful for detailed disease identification after an initial
binary screening (healthy vs diseased).

Example:
--------
    config = PipelineConfig()
    pipeline = DiseasePipeline(config)
    pipeline.run()  # Interactive mode with balancing options

    # Output: dataset/diseases/ with only disease samples
"""

import logging

logger = logging.getLogger(__name__)

import pandas as pd
from .base_pipeline import BasePipeline
from ..config import PipelineConfig

class DiseasePipeline(BasePipeline):
    """
    Multi-class disease classification pipeline.

    Extends BasePipeline to create a disease-only classifier by removing healthy
    plants and filtering rare or problematic disease classes.

    Filtering includes:
    - Removal of all healthy samples
    - Rare diseases below config.rare_disease_threshold (default: 0.1%)
    - Manually excluded diseases from config.excluded_diseases

    Args:
        config: Pipeline configuration with disease filtering parameters
        pipeline_type: Pipeline identifier for output paths (default: 'disease')
        class_column: DataFrame column name for disease labels (default: 'disease')

    Attributes:
        pipeline_type: String identifier ('disease')
        class_column: Column containing disease class names

    Example:
        >>> config = PipelineConfig()
        >>> config.rare_disease_threshold = 0.001  # 0.1%
        >>> config.excluded_diseases = ['Blight', 'Mold']
        >>> pipeline = DiseasePipeline(config)
        >>> pipeline.run()
    """

    def __init__(
            self,
            config: PipelineConfig,
            pipeline_type='disease',
            class_column='disease'):

        super().__init__(config)
        self.pipeline_type = pipeline_type
        self.class_column = class_column

    def _filter_data(self):
        """
        Filter dataset to include only valid disease samples.

        Performs a multi-step filtering process:
        1. Remove all healthy samples (disease != 'healthy')
        2. Identify rare diseases below config.rare_disease_threshold
        3. Combine rare diseases with config.excluded_diseases
        4. Remove all excluded diseases from train and test sets

        Modifies:
            self.df_train: Filtered to disease samples only (excludes healthy + rare/excluded)
            self.df_test: Filtered to disease samples only

        Side Effects:
            - Logs filtering statistics (removed samples, disease distribution)
            - Prints final disease distribution to console

        Example:
            Before: 10,000 samples (2,000 healthy + 8,000 diseased with 50 disease classes)
            After:  7,500 samples (only 30 most common disease classes)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"FILTERING DISEASE SAMPLES")
        logger.info(f"{'='*60}\n")

        # Step 1: Filter out healthy samples
        df_diseases_train = self.df_train[self.df_train['disease'] != 'healthy'].copy()
        df_diseases_test = self.df_test[self.df_test['disease'] != 'healthy'].copy()

        logger.info(f"After removing healthy samples:")
        logger.info(f"  Training: {len(df_diseases_train)} disease samples")
        logger.info(f"  Validation: {len(df_diseases_test)} disease samples")

        # Step 2: Identify and remove rare diseases
        disease_proportions = df_diseases_train['disease'].value_counts(normalize=True)
        rare_diseases = disease_proportions[
            disease_proportions < self.config.rare_disease_threshold
        ].index.tolist()

        logger.info(f"\nRare diseases (< {self.config.rare_disease_threshold*100}%):")
        for disease in rare_diseases:
            count = len(df_diseases_train[df_diseases_train['disease'] == disease])
            logger.info(f"  {disease}: {count} samples")

        # Step 3: Combine rare and manually excluded diseases
        all_excluded = list(set(rare_diseases + self.config.excluded_diseases))

        logger.info(f"\nManually excluded diseases:")
        for disease in self.config.excluded_diseases:
            logger.info(f"  {disease}")

        logger.info(f"\nAll excluded diseases: {all_excluded}")

        # Step 4: Remove excluded diseases
        df_diseases_clean_train = df_diseases_train[
            ~df_diseases_train['disease'].isin(all_excluded)
        ].copy()

        df_diseases_clean_test = df_diseases_test[
            ~df_diseases_test['disease'].isin(all_excluded)
        ].copy()

        logger.info(f"\nAfter removing rare and excluded diseases:")
        logger.info(f"  Training: {len(df_diseases_clean_train)} samples")
        logger.info(f"  Validation: {len(df_diseases_clean_test)} samples")

        # Update dataframes
        self.df_train = df_diseases_clean_train
        self.df_test = df_diseases_clean_test

        # Show disease distribution
        counts = df_diseases_clean_train['disease'].value_counts().sort_index()
        for label, count in counts.items():
            percentage = (count / len(df_diseases_clean_train)) * 100
            logger.info(f"  {label:12}: {count:5} samples ({percentage:5.1f}%)")

        logger.info(f"\n  Total training: {len(df_diseases_clean_train)} samples")

        logger.info(f"\n✓ Filtering complete")

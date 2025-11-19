import logging

logger = logging.getLogger(__name__)

import pandas as pd
from .base_pipeline import BasePipeline
from ..config import PipelineConfig

class BinaryPipeline(BasePipeline):
    """
    Binary classification pipeline: healthy (0) vs diseased (1).

    Extends BasePipeline to create a binary classifier by grouping all diseases
    into a single "diseased" class. This is useful for initial disease screening
    before multi-class classification.

    Args:
        config: Pipeline configuration
        pipeline_type: Pipeline identifier for output paths (default: 'binary')
        class_column: DataFrame column name for binary labels (default: 'binary_class')

    Attributes:
        pipeline_type: String identifier ('binary')
        class_column: Column containing 0 (healthy) or 1 (diseased) labels
    """

    def __init__(
            self,
            config: PipelineConfig,
            pipeline_type='binary',
            class_column='binary_class'):

        super().__init__(config)
        self.pipeline_type = pipeline_type
        self.class_column = class_column

    def filter_data(self):
        """
        Create binary labels: 0 = healthy, 1 = disease.

        Modified:
            self.df_train
            self.df_test
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"CREATING BINARY LABELS")
        logger.info(f"{'='*60}\n")

        # Create binary class column
        self.df_train[self.class_column] = (self.df_train['disease'] != 'healthy').astype(int)
        self.df_test[self.class_column] = (self.df_test['disease'] != 'healthy').astype(int)

        # Show distribution
        healthy_train = (self.df_train[self.class_column] == 0).sum()
        disease_train = (self.df_train[self.class_column] == 1).sum()

        healthy_test = (self.df_test[self.class_column] == 0).sum()
        disease_test = (self.df_test[self.class_column] == 1).sum()

        logger.info(f"Training set:")
        logger.info(f"  Class 0 (Healthy): {healthy_train} samples")
        logger.info(f"  Class 1 (Disease): {disease_train} samples")
        logger.info(f"  Ratio: {healthy_train/disease_train:.2f}:1 (Healthy:Disease)")

        logger.info(f"\nValidation set:")
        logger.info(f"  Class 0 (Healthy): {healthy_test} samples")
        logger.info(f"  Class 1 (Disease): {disease_test} samples")

        logger.info(f"\n✓ Binary labels created")

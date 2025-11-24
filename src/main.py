"""
Main Pipeline Orchestrator
Run all three pipelines from A to Z.
"""
from src.utils.logging_config import setup_logging

setup_logging()

import logging
logger = logging.getLogger(__name__)

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipelines import BinaryPipeline, SpeciesPipeline, DiseasePipeline

def run_all_pipelines(config: PipelineConfig):
    """
    Run all three pipelines in sequence.

    Args:
        config: Pipeline configuration
    """
    logger.info("\n" + "="*80)
    logger.info(" "*20 + "PLANTDOC DATASET PIPELINE")
    logger.info("="*80)

    # Pipeline 1: Binary Classification (Healthy vs Disease)
    logger.info("\n[1/3] Running Binary Pipeline...")
    binary_pipeline = BinaryPipeline(config)
    binary_pipeline.run()

    # Pipeline 2: Species Classification
    logger.info("\n[2/3] Running Species Pipeline...")
    species_pipeline = SpeciesPipeline(config)
    species_pipeline.run()

    # Pipeline 3: Disease Classification
    logger.info("\n[3/3] Running Disease Pipeline...")
    disease_pipeline = DiseasePipeline(config)
    disease_pipeline.run()

    # Summary
    logger.info("\n" + "="*80)
    logger.info(" "*20 + "ALL PIPELINES COMPLETE!")
    logger.info("="*80)
    logger.info("\nGenerated datasets:")
    logger.info(f"  1. Binary:  {config.binary_output_dir}")
    logger.info(f"  2. Species: {config.species_output_dir}")
    logger.info(f"  3. Disease: {config.disease_output_dir}")
    logger.info("\nYou can now train YOLO models using the dataset.yaml files in each directory.")
    logger.info("="*80 + "\n")


def run_single_pipeline(pipeline_type: str, config: PipelineConfig):
    """
    Run a single pipeline.

    Args:
        pipeline_type: Type of pipeline ('binary', 'species', or 'disease')
        config: Pipeline configuration
    """
    if pipeline_type == 'binary':
        pipeline = BinaryPipeline(config)
    elif pipeline_type == 'species':
        pipeline = SpeciesPipeline(config)
    elif pipeline_type == 'disease':
        pipeline = DiseasePipeline(config)
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    pipeline.run()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PlantDoc Dataset Pipeline - Generate YOLO datasets for plant disease detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run all three pipelines
            python -m src.main --all

            # Run only binary pipeline
            python -m src.main --pipeline binary

            # Run only species pipeline
            python -m src.main --pipeline species

            # Run only disease pipeline
            python -m src.main --pipeline disease
                    """
                )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all three pipelines (binary, species, disease)'
    )

    parser.add_argument(
        '--pipeline',
        type=str,
        choices=['binary', 'species', 'disease'],
        help='Run a specific pipeline'
    )

    parser.add_argument(
        '--config',
        type=Path,
        help='Path to custom .env file (optional)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.pipeline:
        parser.error("You must specify either --all or --pipeline")

    # Load configuration
    config = PipelineConfig()

    # Run pipelines
    if args.all:
        run_all_pipelines(config)
    else:
        run_single_pipeline(args.pipeline, config)


if __name__ == '__main__':
    main()

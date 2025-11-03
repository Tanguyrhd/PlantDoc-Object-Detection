"""
Test script for the pipeline - Step by step testing
"""

from src.config import PipelineConfig
from src.pipelines import BinaryPipeline, SpeciesPipeline, DiseasePipeline


def test_config():
    """Test 1: Configuration loading"""
    print("\n" + "="*60)
    print("TEST 1: Configuration")
    print("="*60)

    config = PipelineConfig()
    print(config)
    print(f"\nTrain CSV: {config.train_labels_csv}")
    print(f"Test CSV: {config.test_labels_csv}")
    print(f"Train Images: {config.train_images_dir}")
    print(f"Test Images: {config.test_images_dir}")
    print(f"Plant Species: {config.plant_species}")

    return config


def test_binary_pipeline(config):
    """Test 2: Binary Pipeline"""
    print("\n" + "="*60)
    print("TEST 2: Binary Pipeline")
    print("="*60)

    pipeline = BinaryPipeline(config)

    # Step by step
    print("\n[Step 1] Loading and preparing data...")
    pipeline.load_and_prepare_data()

    print("\n[Step 2] Filtering data...")
    pipeline.filter_data()

    print("\n[Step 3] Balancing data...")
    pipeline.balance_data()

    print("\n[Step 4] Exporting data...")
    pipeline.export_data()

    print("\n✓ Binary pipeline completed!")


def test_species_pipeline(config):
    """Test 3: Species Pipeline"""
    print("\n" + "="*60)
    print("TEST 3: Species Pipeline")
    print("="*60)

    pipeline = SpeciesPipeline(config)

    # Step by step
    print("\n[Step 1] Loading and preparing data...")
    pipeline.load_and_prepare_data()

    print("\n[Step 2] Filtering data...")
    pipeline.filter_data()

    print("\n[Step 3] Balancing data...")
    pipeline.balance_data()

    print("\n[Step 4] Exporting data...")
    pipeline.export_data()

    print("\n✓ Species pipeline completed!")


def test_disease_pipeline(config):
    """Test 4: Disease Pipeline"""
    print("\n" + "="*60)
    print("TEST 4: Disease Pipeline")
    print("="*60)

    pipeline = DiseasePipeline(config)

    # Step by step
    print("\n[Step 1] Loading and preparing data...")
    pipeline.load_and_prepare_data()

    print("\n[Step 2] Filtering data...")
    pipeline.filter_data()

    print("\n[Step 3] Balancing data...")
    pipeline.balance_data()

    print("\n[Step 4] Exporting data...")
    pipeline.export_data()

    print("\n✓ Disease pipeline completed!")


if __name__ == '__main__':
    # Test 1: Config
    config = test_config()

    # Test 2: Binary Pipeline
    # test_binary_pipeline(config)

    # Test 3: Species Pipeline
    # test_species_pipeline(config)

    # Test 4: Disease Pipeline
    # test_disease_pipeline(config)

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print("\nUncomment the pipeline you want to test in test_pipeline.py")

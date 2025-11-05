# PlantDoc Dataset Pipeline

Production-ready pipeline for generating YOLO datasets for plant disease detection.

## Architecture

```bash
src/
├── config/              # Configuration management
│   └── pipeline_config.py
├── data/                # Data loading and validation
│   ├── data_loader.py
│   ├── data_validator.py
│   └── feature_extractor.py
├── processing/          # Data processing utilities
│   ├── balancer.py
│   └── yolo_converter.py
├── pipelines/           # Pipeline implementations
│   ├── base_pipeline.py
│   ├── binary_pipeline.py
│   ├── species_pipeline.py
│   └── disease_pipeline.py
└── main.py              # Main orchestrator
```

## Features

### Common Operations (All Pipelines)

- Load and clean CSV data
- Extract species and disease features
- Fix zero dimensions
- Verify file existence
- Convert to YOLO format
- Generate YAML configuration

### Pipeline-Specific Operations

**Binary Pipeline** (`binary_pipeline.py`)

- Creates binary classification: healthy (0) vs disease (1)
- Natural distribution (no balancing)

**Species Pipeline** (`species_pipeline.py`)

- Classifies plant species (13 classes)
- Balances classes via duplication (target: 1000 samples/class)
- Includes both healthy and diseased plants

**Disease Pipeline** (`disease_pipeline.py`)

- Classifies disease types (9 classes)
- Filters out healthy samples
- Excludes rare and irrelevant diseases
- Balances classes via duplication (target: 1000 samples/class)

## Usage

### Run All Pipelines

```bash
python -m src.main --all
```

This will generate three datasets:

- `dataset/binary/` - Binary classification (healthy vs disease)
- `dataset/species/` - Species classification
- `dataset/diseases/` - Disease classification

### Run Single Pipeline

```bash
# Binary only
python -m src.main --pipeline binary

# Species only
python -m src.main --pipeline species

# Disease only
python -m src.main --pipeline disease
```

### Using in Python Code

```python
from src.config import PipelineConfig
from src.pipelines import BinaryPipeline, SpeciesPipeline, DiseasePipeline

# Load configuration
config = PipelineConfig()

# Run binary pipeline
binary_pipeline = BinaryPipeline(config)
binary_pipeline.run()

# Run species pipeline
species_pipeline = SpeciesPipeline(config)
species_pipeline.run()

# Run disease pipeline
disease_pipeline = DiseasePipeline(config)
disease_pipeline.run()
```

## Configuration

The pipeline uses environment variables from `.env`:

```env
TRAIN_LABELS_CSV=data/raw/train.csv
TEST_LABELS_CSV=data/raw/test.csv
TRAIN_IMAGES_DIR=data/raw/train_images
TEST_IMAGES_DIR=data/raw/test_images
PLANT_SPECIES=Apple,Bell Pepper,Blueberry,Cherry,Corn,Grape,Peach,Potato,Raspberry,Soyabean,Squash,Strawberry,Tomato
```

You can customize pipeline behavior in [pipeline_config.py](src/config/pipeline_config.py):

- `target_samples_per_class`: Target samples per class (default: 1000)
- `rare_disease_threshold`: Threshold for rare diseases (default: 0.001)
- `excluded_diseases`: Manually excluded diseases

## Output Structure

Each pipeline generates:

```bash
dataset/{pipeline_type}/
├── images/
│   ├── train/          # Training images
│   └── val/            # Validation images
├── labels/
│   ├── train/          # Training labels (YOLO format)
│   └── val/            # Validation labels (YOLO format)
└── dataset.yaml        # YOLO configuration
```

## YOLO Training

After running the pipelines, you can train YOLO models:

```python
from ultralytics import YOLO

# Binary classification
model = YOLO('yolov8n.pt')
model.train(data='dataset/binary/dataset.yaml', epochs=100)

# Species classification
model = YOLO('yolov8n.pt')
model.train(data='dataset/species/dataset.yaml', epochs=100)

# Disease classification
model = YOLO('yolov8n.pt')
model.train(data='dataset/diseases/dataset.yaml', epochs=100)
```

## Extending the Pipeline

To create a custom pipeline:

1. Create a new class inheriting from `BasePipeline`
2. Implement abstract methods:
   - `get_pipeline_type()`: Return pipeline identifier
   - `get_class_column()`: Return classification column name
   - `filter_data()`: Implement filtering logic
   - `balance_data()`: Implement balancing logic

Example:

```python
from src.pipelines.base_pipeline import BasePipeline

class CustomPipeline(BasePipeline):
    def get_pipeline_type(self) -> str:
        return 'custom'

    def get_class_column(self) -> str:
        return 'custom_class'

    def filter_data(self):
        # Your filtering logic
        pass

    def balance_data(self):
        # Your balancing logic
        pass
```

Acknowledgment: This project builds on a fork for the idea and dataset from my mentor, Jean-Philippe Petit-Frere. All implementation and development in this repository were done independently.

# PlantDoc Object Detection - Implementation Guide

A production-ready plant disease detection system using YOLO, featuring a three-pipeline architecture for binary classification, species identification, and disease diagnosis. Deployed as an asynchronous FastAPI service on Google Cloud Run.

**Live Demo**: [View on Streamlit](https://plantdoc-tanguy.streamlit.app)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Results Snapshot](#results-snapshot)
- [Architecture](#architecture)
- [Key Technical Decisions](#key-technical-decisions)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Improvements](#future-improvements)

---

## Project Overview

### The Problem

Plant diseases cause 35% of crop yield loss in India annually. Early detection requires expensive laboratory infrastructure and plant pathology expertise that isn't widely accessible.

### The Solution

A computer vision system that analyzes leaf images to:

1. Detect if a plant is healthy or diseased (Binary Classification)
2. Identify the plant species from 13 categories (Species Identification)
3. Diagnose the specific disease from 9 conditions (Disease Classification)

### Tech Stack

- **ML Framework**: YOLO (Ultralytics)
- **Backend**: FastAPI + Uvicorn
- **Async Processing**: asyncio + ThreadPoolExecutor
- **Image Processing**: OpenCV, Pillow
- **Data Processing**: Pandas, NumPy
- **Deployment**: Docker, Google Cloud Run
- **Language**: Python 3.10

---

## Results Snapshot

| Model            | Task                    | Performance          | Status              |
|------------------|-------------------------|----------------------|---------------------|
| Binary           | Healthy vs Diseased     | 87-93% accuracy      | Production Ready    |
| Species          | Plant Identification    | 67-99% per class     | Production Ready    |
| Disease          | Disease Diagnosis       | 44-95% per disease   | Limited Deployment* |

*Disease model reliable only for distinctive diseases (Rust, Scab, Powdery Mildew). See [analysis.md](analysis.md) for details.

**Training Results Visualization**:

- Binary Model: [results/binary/results/results.png](results/binary/results/results.png)
- Species Model: [results/species/results/results.png](results/species/results/results.png)
- Disease Model: [results/diseases/results/results.png](results/diseases/results/results.png)

---

## Architecture

### System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                         Client Request                      │
│                    (Image Upload via API)                   │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Global Model Loading (Startup)                       │  │
│  │  • Binary Model (Healthy vs Diseased)                 │  │
│  │  • Species Model (13 plant species)                   │  │
│  │  • Disease Model (9 disease types)                    │  │
│  └───────────────────────────────────────────────────────┘  │
│                             │                               │
│                             ▼                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Async Request Handler                                │  │
│  │  • Async file I/O (aiofiles)                          │  │
│  │  • ThreadPoolExecutor for CPU-bound prediction        │  │
│  │  • Image preprocessing                                │  │
│  └───────────────────────────────────────────────────────┘  │
│                             │                               │
│                             ▼                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  YOLO Prediction                                      │  │
│  │  • Bounding box detection                             │  │
│  │  • Class probability                                  │  │
│  │  • Confidence scoring                                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                             │                               │
│                             ▼                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Response Generation                                  │  │
│  │  • Annotated image (Base64)                           │  │
│  │  • Detection results JSON                             │  │
│  │  • Confidence scores                                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                         Client Response                     │
│              (Annotated Image + Predictions)                │
└─────────────────────────────────────────────────────────────┘
```

### Data Processing Pipeline Architecture

The project uses an **Abstract Base Class pattern** with three specialized pipeline implementations:

```python
BasePipeline (Abstract)
    │
    ├─ load_clean_extract_fix_verify()
    │   ├─ Load CSV annotations
    │   ├─ Clean class names (remove "leaf", normalize)
    │   ├─ Extract features (species, disease)
    │   └─ Fix/verify data (dimensions, file existence)
    │
    ├─ filter_data() [Abstract - implemented by subclasses]
    ├─ balance_data() [Optional class balancing]
    └─ export_data() [YOLO format conversion]

         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
  BinaryPipeline  SpeciesPipeline  DiseasePipeline
  (2 classes)     (13 classes)     (9 classes)
```

**Pipeline Specializations**:

1. **BinaryPipeline** ([src/pipelines/binary_pipeline.py](src/pipelines/binary_pipeline.py))
   - Creates binary labels: 0 (healthy) vs 1 (diseased)
   - Simplest model, best performance

2. **SpeciesPipeline** ([src/pipelines/species_pipeline.py](src/pipelines/species_pipeline.py))
   - Extracts plant species from class labels
   - 13 distinct plant categories

3. **DiseasePipeline** ([src/pipelines/disease_pipeline.py](src/pipelines/disease_pipeline.py))
   - Extracts disease names from class labels
   - Filters out healthy samples (diseased only)
   - 9 disease categories

**Shared Utilities** ([src/processing/](src/processing/)):

- `data_validation.py`: Data cleaning and validation functions
- `yolo_converter.py`: YOLO format export utilities

---

## Key Technical Decisions

### 1. Abstract Base Class for Pipelines

**Decision**: Use abstract base class ([src/pipelines/base_pipeline.py](src/pipelines/base_pipeline.py)) with template method pattern.

**Benefits**:

- **DRY Principle**: Data loading, cleaning, and export logic shared across all pipelines
- **Flexibility**: Each pipeline overrides `filter_data()` for task-specific processing
- **Maintainability**: Bug fixes in base class propagate to all pipelines
- **Extensibility**: New pipelines (e.g., severity classification) require minimal code

**Implementation**:

```python
class BasePipeline(ABC):
    def run(self):
        self.load_clean_extract_fix_verify()
        self.filter_data()  # Abstract - implemented by subclasses
        self.balance_data()
        self.export_data()

    @abstractmethod
    def filter_data(self):
        raise NotImplementedError
```

### 2. Async API with ThreadPoolExecutor

**Decision**: Combine asyncio for I/O operations with ThreadPoolExecutor for CPU-bound ML inference ([api/fast.py](api/fast.py)).

**Problem Solved**:

- YOLO inference is CPU-intensive and blocks the event loop
- Multiple concurrent requests would serialize without threading
- File I/O (image upload/save) can be async

**Implementation**:

```python
executor = ThreadPoolExecutor(max_workers=4)

async def predict_with_model(file: UploadFile, model: YOLO):
    # Async I/O operations
    async with aiofiles.open(temp_path, "wb") as buffer:
        await buffer.write(await file.read())

    # CPU-bound prediction in thread pool
    loop = asyncio.get_event_loop()
    predictions, img = await loop.run_in_executor(executor, run_prediction)

    # Async cleanup
    await asyncio.to_thread(os.remove, temp_path)
```

**Benefits**:

- **Concurrency**: Handle multiple requests simultaneously
- **Responsiveness**: API remains responsive during inference
- **Resource Efficiency**: Max 4 workers prevents CPU thrashing

### 3. Global Model Loading vs Lazy Loading

**Decision**: Load all three YOLO models at application startup ([api/fast.py:22-32](api/fast.py#L22-L32)).

**Trade-offs**:

| Approach           | Startup Time | First Request | Memory Usage | Request Latency |
|--------------------|--------------|---------------|--------------|-----------------|
| Global Loading     | High (~10s)  | Fast          | High         | Low             |
| Lazy Loading       | Fast         | Slow (first)  | Lower        | Variable        |

**Chosen**: Global Loading

**Rationale**:

- **Predictable Performance**: Every request has consistent low latency
- **Cloud Run Constraint**: Startup time amortized over container lifetime
- **User Experience**: Users don't wait 10s on first prediction
- **Memory Acceptable**: Cloud Run instance has sufficient RAM

**Implementation**:

```python
app.state.models = load_models()  # Loaded once at startup

@app.post('/predict/binary')
async def predict_binary(file: UploadFile = File(...)):
    return await predict_with_model(file, app.state.models['binary'])
```

### 4. Docker Optimization

**Decision**: Use `python:3.10-slim` base image with CPU-only PyTorch ([Dockerfile](Dockerfile)).

**Optimizations Applied**:

1. **Slim Python Base**: `python:3.10-slim` instead of full image
   - Saved: ~400MB

2. **CPU-Only PyTorch**: No CUDA libraries
   - Saved: ~2GB
   - Justification: Cloud Run doesn't provide GPU instances

3. **Minimal File Copy**: Only copy `api/` and `results/models/best.pt` directories
   - Training data, notebooks excluded via `.gitignore`

4. **No Cache Pip Installs**: `pip install --no-cache-dir`
   - Saved: ~200MB

**Final Image Size**: ~1.2GB (down from ~4GB)

**Dockerfile**:

```dockerfile
FROM python:3.10-slim
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \  # OpenCV dependencies only
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY results/ ./results/  # Models only, no dataset

EXPOSE 8000
CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
```

---

## Installation

### Prerequisites

- Python 3.10+
- pip
- (Optional) Docker for containerized deployment

### Local Development Setup

- **Clone the repository**:

  - ```bash
    git clone https://github.com/Tanguyrhd/personal-projects/PlantDoc-Object-Detection.git
    cd PlantDoc-Object-Detection
    ```

- **Create virtual environment**:

  - ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

- **Install dependencies**:

  For training pipelines:

  - ```bash
    make setup
    ```

- **Verify installation**:

  - ```bash
    python -c "from ultralytics import YOLO; print('YOLO ready')"
    ```

---

## Usage

### Training New Models

- **Ensure dataset is in place**:

```bash
dataset/
├── TRAIN/          # Training images
├── TEST/           # Test images
├── train_labels.csv
└── test_labels.csv
```

- **Run a pipeline**:

```bash
# Binary classification
make run-binary

# Species identification
make run-species

# Disease classification
make run-diseases

# All three pipeline
make run-all
```

- **Pipeline steps**:
  - Loads and cleans CSV annotations
  - Extracts features (species, disease)
  - Fixes corrupted data
  - Optionally balances classes
  - Exports to YOLO format
  - Trains YOLOv8 model
  - Saves results to `results/{pipeline_name}/`

### Running the API Locally

**Ensure models are trained**:

```bash
ls results/binary/models/best.pt
ls results/species/models/best.pt
ls results/diseases/models/best.pt
```

**Start and test the API**:

```bash
make run-api
```

### Docker Deployment

**Build the image**:

```bash
make docker-build
```

**Run the container**:

```bash
make docker-run-local
```

**Test**:

```bash
curl http://localhost:8000/
```

### Cloud Run Deployment

**Deploy on Google Cloud Run with Artifact Registry**:

```bash
make docker-deploy-cloud
```

---

## API Documentation

### Base URL

```text
Local: http://localhost:8000
Production: https://plantdoc-tanguy.streamlit.app
```

### Endpoints

#### `GET /`

**Health check endpoint**:

**Response**:

```json
{
  "message": "PlantDoc API",
  "models_loaded": ["binary", "species", "diseases"]
}
```

#### `POST /predict/binary`

**Binary classification: healthy vs diseased**:

**Request**:

- Content-Type: `multipart/form-data`
- Body: `file` (image file, JPG/PNG)

**Response**:

```json
{
  "predictions": [
    {
      "class_id": 1,
      "class_name": "diseased",
      "confidence": 0.92,
      "bbox": [120, 45, 340, 280]
    }
  ],
  "annotated_image": "data:image/png;base64,iVBORw0KGgo..."
}
```

#### `POST /predict/species`

**Species identification (13 plant species)**:

**Request**: Same as binary endpoint

**Response**:

```json
{
  "predictions": [
    {
      "class_id": 12,
      "class_name": "Tomato",
      "confidence": 0.87,
      "bbox": [95, 30, 420, 310]
    }
  ],
  "annotated_image": "data:image/png;base64,..."
}
```

#### `POST /predict/diseases`

**Disease diagnosis (9 disease types)**:

**Request**: Same as binary endpoint

**Response**:

```json
{
  "predictions": [
    {
      "class_id": 5,
      "class_name": "Rust",
      "confidence": 0.81,
      "bbox": [110, 55, 330, 275]
    }
  ],
  "annotated_image": "data:image/png;base64,..."
}
```

### Error Handling

**400 Bad Request**: Invalid file format

```json
{
  "detail": "Invalid file format. Please upload JPG or PNG."
}
```

**500 Internal Server Error**: Model inference failure

```json
{
  "detail": "Prediction failed. Please try again."
}
```

---

## Challenges and Solutions

### Challenge 1: Corrupted Data

**Problem**:

- Some images had `width=0` or `height=0` in CSV metadata
- Missing image files referenced in annotations
- Training would crash on corrupted data

**Solution** ([src/processing/data_validation.py](src/processing/data_validation.py)):

```python
def fix_zero_dimensions(df, image_folder, dataset_type):
    """Read actual image dimensions to fix zero values"""
    for idx, row in df.iterrows():
        if row['width'] == 0 or row['height'] == 0:
            with Image.open(image_folder / row['filename']) as img:
                w, h = img.size
                df.at[idx, 'width'] = w
                df.at[idx, 'height'] = h
    return df

def verify_files_exist(df, image_folder, dataset_type):
    """Filter out rows with missing image files"""
    existing_mask = [
        (image_folder / row['filename']).exists()
        for _, row in df.iterrows()
    ]
    return df[existing_mask].copy()
```

**Impact**: 100% of data validated before training, preventing crashes

### Challenge 2: Disease Model Low Precision

**Problem**:

- Disease model struggled with accuracy (44-95% per class)
- High confusion between similar diseases (Bacterial Spot, Early Blight, Late Blight)
- Some diseases barely detected (Bacterial spot: 44%)

**Root Causes** (from data analysis):

1. **Overly Broad Categories**: Some diseases visually similar
2. **Data Quality**: Internet-sourced images are mislabeled, not precise enough, etc
3. **Visual Ambiguity**: Disease stages/severities look like different diseases

**Solutions Attempted**:

1. **Class Balancing**: Oversample minority classes
2. **Data Augmentation**: Aggressive augmentation for confused classes
3. **Longer Training**: Increased epochs to 50+

**Outcome**: Moderate improvement, but fundamental issue is data quality

**Recommendation** (see [analysis_claude.md](analysis_claude.md)):

- Merge visually similar diseases into broader categories
- Collect expert-verified disease images
- Consider hierarchical classification (disease type → specific disease)

### Challenge 3: Docker Image Size

**Problem**:

- Initial Docker image: ~4GB
- Google Cloud Run charges by memory-time
- Slow deployment and high costs

**Solutions Applied**:

| Optimization             | Size Saved | Implementation                        |
|--------------------------|------------|---------------------------------------|
| Slim base image          | ~400MB     | `python:3.10-slim` vs `python:3.10`  |
| CPU-only PyTorch         | ~2GB       | `requirements-api.txt` excludes CUDA |
| Exclude training data    | ~500MB     | `.dockerignore`, copy only `results/`|
| No pip cache             | ~200MB     | `pip install --no-cache-dir`         |

**Final Image**: ~1.2GB (70% reduction)

---

## Future Improvements

### Short-term (1-3 months)

**1. Confidence Thresholding**:

- Reject low-confidence predictions (< 0.7)
- Return "uncertain - consult expert" for ambiguous cases
- Prevents confident wrong predictions

**2. Multi-model Ensemble**:

- Combine predictions from multiple models
- Voting mechanism for disease classification
- Improves reliability for confused classes

**3. API Enhancements**:

- Add batch prediction endpoint
- Implement caching for repeated images
- Add request rate limiting

### Medium-term (3-6 months)

**1. Enhanced Data Collection**:

- Partner with agricultural research institutions
- Collect 500+ expert-verified images per confused disease
- Include disease severity labels (mild, moderate, severe)

**2. Hierarchical Classification**:

```text
Binary (Healthy vs Diseased) [89% reliable]
    ↓ if diseased
Disease Category (Fungal vs Bacterial vs Viral) [new model]
    ↓
Specific Disease [focused models per category]
```

**3. Model Improvements**:

- Try higher-resolution inputs (1024x1024)
- Experiment with YOLOv9 or YOLOv10
- Add attention mechanisms for disease-specific features

### Long-term (6-12 months)

**1. Mobile Application**:

- Edge deployment with TensorFlow Lite
- Offline prediction capability
- Real-time camera integration

**2. Multi-spectral Imaging**:

- Incorporate near-infrared (NIR) imaging
- Detect diseases before visible symptoms
- Requires specialized camera hardware

**3. Disease Progression Tracking**:

- Time-series analysis of same plant over time
- Track disease severity changes
- Recommend treatment timing

**4. Integration with Treatment Database**:

- Link disease predictions to treatment recommendations
- Pesticide/fungicide suggestions
- Organic treatment alternatives

**5. Field Testing**:

- Pilot deployment with farmers
- Collect real-world performance data
- Iterate based on user feedback

---

## Project Structure

```bash
PlantDoc-Object-Detection/
├── api/
│   ├── __init__.py
│   └── fast.py                 # FastAPI application
├── dataset/
│   ├── TRAIN/                  # Training images
│   ├── TEST/                   # Test images
│   ├── train_labels.csv
│   └── test_labels.csv
├── results/
│   ├── binary/
│   │   ├── models/best.pt      # Trained binary model
│   │   └── results/            # Training metrics
│   ├── species/
│   │   ├── models/best.pt      # Trained species model
│   │   └── results/
│   └── diseases/
│       ├── models/best.pt      # Trained disease model
│       └── results/
├── src/
│   ├── config/
│   │   └── pipeline_config.py  # Configuration settings
│   ├── pipelines/
│   │   ├── base_pipeline.py    # Abstract base class
│   │   ├── binary_pipeline.py  # Binary classification
│   │   ├── species_pipeline.py # Species identification
│   │   └── disease_pipeline.py # Disease diagnosis
│   ├── processing/
│   │   ├── data_validation.py  # Data cleaning utilities
│   │   └── yolo_converter.py   # YOLO format export
│   └── main.py                 # Training entry point
├── Dockerfile                  # Container definition
├── requirements.txt            # Training dependencies
├── requirements-api.txt        # API dependencies (minimal)
├── README.md                   # Original README
├── readme_claude.md           # This file (implementation guide)
└── analysis_claude.md         # Data science analysis

```

---

## Contributing

This is a personal project, but suggestions and feedback are welcome!

**To report issues**:

1. Check existing issues on GitHub
2. Provide clear reproduction steps
3. Include sample images if reporting prediction bugs

**To suggest improvements**:

1. Open an issue describing the enhancement
2. Explain the expected benefit
3. Reference relevant code sections

---

## License

Creative Commons Attribution 4.0 International - Free to use for learning and research purposes.

**Original Dataset**:

- PlantDoc Dataset by Singh et al. (2020)
- Paper: [ArXiv](https://arxiv.org/abs/1911.10317) | [ACM](https://dl.acm.org/doi/10.1145/3371158.3371196)

---

## Acknowledgments

- **PlantDoc Dataset**: Singh, Jain, Jain, Kayal, Kumawat, and Batra (2020)
- **YOLOv8**: Ultralytics team for the excellent ML framework
- **FastAPI**: Sebastián Ramírez for the modern async web framework
- **Google Cloud**: For generous Cloud Run free tier

---

## Contact

**Author**: Tanguy Richard
**GitHub**: [Tanguyrhd](https://github.com/Tanguyrhd)
**Project**: [PlantDoc-Object-Detection](https://github.com/Tanguyrhd/PlantDoc-Object-Detection)

For questions about implementation details, see the code comments or open an issue.

For questions about model performance and data analysis, see [analysis.md](analysis.md).

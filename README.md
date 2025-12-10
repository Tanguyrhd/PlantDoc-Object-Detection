# Plant Disease Classification Project

## Overview

This project uses the PlantDoc dataset to build a computer vision model that can classify plant diseases from leaf images. Early detection of plant diseases is crucial for agriculture, but it often requires lab infrastructure and expertise that isn't widely available. This project explores using machine learning to make plant disease detection more accessible.

## The Problem

Plant diseases cause significant crop losses worldwide. For example, India loses 35% of its annual crop yield due to plant diseases. The challenge is detecting these diseases early enough to prevent widespread damage, especially in areas without access to plant pathology labs.

## Project Goal

Build a machine learning model that can:

1. Take an image of a plant leaf as input
2. Identify with species it is
3. Classify whether the leaf is healthy or diseased
4. If diseased, identify the specific disease

## The Dataset

This repository contains the **PlantDoc dataset**, which includes:

- **2,598 annotated images** of plant leaves
- **13 plant species**: Apple, Bell Pepper, Blueberry, Cherry, Corn, Grape, Peach, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato
- **27 classes** including both healthy leaves and various diseases (rust, blight, mold, spots, etc.)
- Images collected from the internet and annotated with bounding boxes
- Split into training (2,355 images) and test (239 images) sets

### Raw Dataset Structure

```text
dataset/
├── TRAIN/
│   ├── *.jpg          # Training images
│   └── *.xml          # Annotations (bounding boxes)
├── TEST/
│   ├── *.jpg          # Test images
│   └── *.xml          # Annotations (bounding boxes)
├── train_labels.csv   # Training labels in CSV format
└── test_labels.csv    # Test labels in CSV format
```

### Data Format

**CSV Files** (easiest to work with):

- Columns: `filename`, `width`, `height`, `class`, `xmin`, `ymin`, `xmax`, `ymax`
- Each row represents one bounding box annotation
- Some images have multiple annotations

**XML Files**:

- PASCAL VOC format
- Contains the same information as CSV but in XML structure

### Example Classes

- Healthy leaves: `Apple leaf`, `Tomato leaf`, `Potato leaf`
- Diseased leaves: `Apple rust leaf`, `Tomato Early blight leaf`, `Potato leaf late blight`
- And many more...

## Original Research

This dataset was created for the paper "PlantDoc: A Dataset for Visual Plant Disease Detection" published at ACM CoDS-COMAD 2020.

**Paper**: [Arxiv](https://arxiv.org/abs/1911.10317) | [ACM](https://dl.acm.org/doi/10.1145/3371158.3371196)

**Authors**: Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and Nipun Batra

## License

Creative Commons Attribution 4.0 International - You're free to use this dataset for learning and research purposes.

"""
YOLO Format Conversion Utilities

This module provides functions to convert annotated image datasets into YOLO format.

YOLO Format Specification:
--------------------------
Each image has a corresponding .txt label file with one row per object:
<class_id> <x_center> <y_center> <width> <height>

Where:
- class_id: Integer class index (0-based)
- x_center, y_center: Object center coordinates (normalized 0.0-1.0 relative to image dimensions)
- width, height: Bounding box dimensions (normalized 0.0-1.0 relative to image dimensions)

Directory Structure:
-------------------
output_dir/
├── images/
│   ├── train/  # Training images (.jpg, .png, etc.)
│   └── val/    # Validation images
├── labels/
│   ├── train/  # Training labels (.txt)
│   └── val/    # Validation labels (.txt)
└── dataset.yaml  # YOLO dataset configuration

Example:
--------
For an image 'plant_001.jpg' (640x480) with a disease detection box:

labels/train/plant_001.txt:
    0 0.500000 0.625000 0.312500 0.416667

This represents:
- Class 0 (e.g., "healthy")
- Center at (320px, 300px) → normalized (0.5, 0.625)
- Box size 200x200px → normalized (0.3125, 0.4167)
"""
import logging

logger = logging.getLogger(__name__)

import shutil
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

def convert_bbox_to_yolo(row: pd.Series) -> Tuple[float, float, float, float]:
    """
    Convert bounding box to YOLO format (normalized coordinates).

    Args:
        row: DataFrame row with xmin, xmax, ymin, ymax, width, height

    Returns:
        Tuple of (x_center, y_center, bbox_width, bbox_height) in normalized coords
    """
    x_center = (row['xmin'] + row['xmax']) / 2 / row['width']
    y_center = (row['ymin'] + row['ymax']) / 2 / row['height']
    bbox_width = (row['xmax'] - row['xmin']) / row['width']
    bbox_height = (row['ymax'] - row['ymin']) / row['height']

    return x_center, y_center, bbox_width, bbox_height

def export_to_yolo(
    df: pd.DataFrame,
    source_images_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    class_mapping: Dict[str, int],
    class_column: str = 'class'
) -> Tuple[int, int]:
    """
    Export dataset to YOLO format.

    Args:
        df: DataFrame with image annotations
        source_images_dir: Source directory containing original images
        output_images_dir: Output directory for images
        output_labels_dir: Output directory for labels
        class_mapping: Dictionary mapping class names to indices
        class_column: Column name containing class labels

    Returns:
        Tuple of (exported_count, skipped_count)
    """
    exported = 0
    skipped = 0

    source_images_dir = Path(source_images_dir)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)

    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for filename, group in df.groupby("filename"):
        try:
            # Check if this is a duplicate (has _dup in name)
            if '_dup' in filename:
                # Get original filename
                original_filename = filename.split('_dup')[0] + Path(filename).suffix
                src = source_images_dir / original_filename
            else:
                src = source_images_dir / filename

            if not src.exists():
                skipped += 1
                continue

            # Copy image
            dst = output_images_dir / filename
            shutil.copy2(src, dst)

            # Create label file
            label_file = output_labels_dir / (Path(filename).stem + ".txt")
            with open(label_file, "w") as f:
                for _, row in group.iterrows():
                    cls_idx = class_mapping[row[class_column]]
                    x_c, y_c, w, h = convert_bbox_to_yolo(row)
                    f.write(f"{cls_idx} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

            exported += 1

        except Exception as e:
            logger.exception(f"Error processing {filename}: {e}")
            skipped += 1

    return exported, skipped

def create_yaml_config(
    output_dir: Path,
    class_mapping: Dict[str, int],
    train_subdir: str = 'images/train',
    val_subdir: str = 'images/val'
) -> Path:
    """
    Create YOLO dataset YAML configuration file.

    Args:
        output_dir: Base output directory
        class_mapping: Dictionary mapping class names to indices
        train_subdir: Relative path to training images
        val_subdir: Relative path to validation images

    Returns:
        Path to created YAML file
    """
    output_dir = Path(output_dir)

    yaml_content = {
        'path': str(output_dir.resolve()),
        'train': train_subdir,
        'val': val_subdir,
        'nc': len(class_mapping),
        'names': {idx: name for name, idx in class_mapping.items()}
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    logger.info(f"\n✓ YAML configuration created: {yaml_path}")
    logger.info(f"  Classes: {len(class_mapping)}")

    return yaml_path

def create_class_mapping(df: pd.DataFrame, column: str) -> Dict[str, int]:
    """
    Create class-to-index mapping (alphabetically sorted).

    Args:
        df: DataFrame
        column: Column containing class names

    Returns:
        Dictionary mapping class names to indices
    """
    unique_classes = sorted(df[column].unique())
    class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}

    logger.info(f"\nClass mapping created ({len(class_mapping)} classes):")
    for cls, idx in class_mapping.items():
        count = len(df[df[column] == cls])
        logger.info(f"  {idx}: {cls} ({count} samples)")

    return class_mapping

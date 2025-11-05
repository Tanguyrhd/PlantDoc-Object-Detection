"""
Feature Extractor
Extracts species and disease information from class labels.
"""

import re
import pandas as pd
from typing import List, Optional


class FeatureExtractor:
    """Extracts species and disease features from class labels."""

    def __init__(self, plant_species: List[str]):
        """
        Initialize feature extractor.

        Args:
            plant_species: List of plant species names
        """
        self.plant_species = plant_species

    def extract_species(self, text: str) -> Optional[str]:
        """
        Extract plant species from text.

        Args:
            text: Class label text

        Returns:
            Species name or None if not found
        """
        for plant in self.plant_species:
            if re.search(rf"\b{plant}\b", text, flags=re.IGNORECASE):
                return plant
        return None

    def extract_disease(self, text: str) -> str:
        """
        Extract disease name from text by removing species name.

        Args:
            text: Class label text

        Returns:
            Disease name or "healthy" if no disease
        """
        for plant in self.plant_species:
            text = re.sub(rf"\b{plant}\b", "", text, flags=re.IGNORECASE).strip()

        # Normalize to title case to avoid duplicates
        return text.title() if text else "healthy"

    def add_features(self, df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
        """
        Add 'species' and 'disease' columns to DataFrame.

        Args:
            df: DataFrame with 'class' column
            dataset_type: 'train' or 'test'

        Returns:
            DataFrame with added 'species' and 'disease' columns
        """
        df = df.copy()
        df['species'] = df['class'].apply(self.extract_species)
        df['disease'] = df['class'].apply(self.extract_disease)

        print(f"✓ Features extracted for the {dataset_type} dataset (species, disease)")

        return df

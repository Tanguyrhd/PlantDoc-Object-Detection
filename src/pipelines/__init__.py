"""Pipeline module for different classification tasks."""

from .base_pipeline import BasePipeline
from .binary_pipeline import BinaryPipeline
from .species_pipeline import SpeciesPipeline
from .disease_pipeline import DiseasePipeline

__all__ = ["BasePipeline", "BinaryPipeline", "SpeciesPipeline", "DiseasePipeline"]

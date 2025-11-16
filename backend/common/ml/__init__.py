"""
Machine Learning module for NANO-OS.

This module provides ML-based property prediction for materials and structures.
It includes:
- Stub implementations for property prediction (deterministic hashing)
- Model registry and version management
- Comparison utilities for ML vs simulation results
- Future integration points for real ML models (CGCNN, MEGNet, M3GNET, ALIGNN)

The stub implementation provides realistic, deterministic predictions based on
structure properties, allowing the API and database infrastructure to be
developed and tested before real ML models are integrated.
"""

from .properties import (
    predict_properties_for_structure,
    get_available_models,
    ModelInfo,
)
from .comparison import compare_ml_vs_simulation

__all__ = [
    "predict_properties_for_structure",
    "get_available_models",
    "ModelInfo",
    "compare_ml_vs_simulation",
]

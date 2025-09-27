"""
ORION Candidate Generation Module
================================

Advanced candidate generation with surrogate models and uncertainty quantification.
"""

from .candidate_generator import CandidateGenerator
from .advanced_generator import (
    GNNSurrogate,
    EnsembleSurrogate,
    UncertaintyQuantifier,
    DiversitySampler
)
from .surrogate_trainer import SurrogatePredictorTrainer

__all__ = [
    # Main generator
    "CandidateGenerator",
    
    # Advanced components
    "GNNSurrogate",
    "EnsembleSurrogate",
    "UncertaintyQuantifier",
    "DiversitySampler",
    
    # Training
    "SurrogatePredictorTrainer",
]
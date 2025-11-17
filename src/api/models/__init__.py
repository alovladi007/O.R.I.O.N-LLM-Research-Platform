"""
SQLAlchemy ORM models for NANO-OS.

Models defined here represent the core data structures for:
- User management and authentication
- Materials and atomic structures
- Simulation workflows and jobs
- Results and provenance tracking
- Vector embeddings for ML/search

All models use:
- UUID primary keys for distributed systems
- Timestamps for auditing
- Proper foreign key relationships
- JSON fields for flexible metadata
"""

from .user import User
from .material import Material
from .structure import Structure
from .structure_features import StructureFeatures
from .ml_model_registry import MLModelRegistry
from .workflow import WorkflowTemplate
from .simulation import SimulationJob, SimulationResult
from .embedding import VectorEmbedding
from .predicted_properties import PredictedProperties
from .provenance import ProvenanceRecord, EntityType, EventType
from .multiscale import (
    MesoscaleSimulationJob,
    MesoscaleSimulationResult,
    ContinuumSimulationJob,
    ContinuumSimulationResult,
    MesoscaleEngineType,
    ContinuumEngineType,
    MultiscaleJobStatus,
)
from .campaign import DesignCampaign, DesignIteration, CampaignStatus

__all__ = [
    "User",
    "Material",
    "Structure",
    "StructureFeatures",
    "MLModelRegistry",
    "WorkflowTemplate",
    "SimulationJob",
    "SimulationResult",
    "VectorEmbedding",
    "PredictedProperties",
    "ProvenanceRecord",
    "EntityType",
    "EventType",
    "MesoscaleSimulationJob",
    "MesoscaleSimulationResult",
    "ContinuumSimulationJob",
    "ContinuumSimulationResult",
    "MesoscaleEngineType",
    "ContinuumEngineType",
    "MultiscaleJobStatus",
    "DesignCampaign",
    "DesignIteration",
    "CampaignStatus",
]

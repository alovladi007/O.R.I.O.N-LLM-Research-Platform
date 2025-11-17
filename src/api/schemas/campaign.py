"""
Design campaign schemas for API requests and responses.

These schemas define the structure of campaign-related API payloads,
including configuration, requests, and responses.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# Configuration schemas

class PropertyTarget(BaseModel):
    """
    Target specification for a material property.

    Defines what value or range a property should have, along with
    importance weighting and tolerance.
    """
    value: Optional[float] = Field(
        None,
        description="Target value (for exact matching with tolerance)"
    )
    tolerance: Optional[float] = Field(
        None,
        ge=0.0,
        description="Acceptable deviation from target value"
    )
    min: Optional[float] = Field(
        None,
        description="Minimum acceptable value (for range constraints)"
    )
    max: Optional[float] = Field(
        None,
        description="Maximum acceptable value (for range constraints)"
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Importance weight (higher = more important)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "value": 2.0,
                "tolerance": 0.2,
                "weight": 1.0
            }
        }


class CampaignConstraints(BaseModel):
    """
    Constraints for structure generation.

    Defines limits and requirements for generated structures,
    such as allowed elements, size limits, dimensionality, etc.
    """
    elements: Optional[List[str]] = Field(
        None,
        description="Allowed chemical elements (e.g., ['Mo', 'W', 'S', 'Se'])"
    )
    max_atoms: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum number of atoms per structure"
    )
    min_atoms: Optional[int] = Field(
        None,
        ge=1,
        description="Minimum number of atoms per structure"
    )
    dimensionality: Optional[int] = Field(
        None,
        ge=0,
        le=3,
        description="Required dimensionality (0=molecule, 1=1D, 2=2D, 3=bulk)"
    )
    max_species: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum number of different elements"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "elements": ["Mo", "W", "S", "Se"],
                "max_atoms": 20,
                "dimensionality": 2
            }
        }


class CampaignConfig(BaseModel):
    """
    Complete configuration for a design campaign.

    Defines the optimization objectives, constraints, and strategy
    for an autonomous design campaign.
    """
    target_properties: Dict[str, PropertyTarget] = Field(
        default_factory=dict,
        description="Target properties and their specifications"
    )
    constraints: Optional[CampaignConstraints] = Field(
        None,
        description="Constraints for structure generation"
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of design iterations"
    )
    candidates_per_iteration: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of candidate structures per iteration"
    )
    evaluation_strategy: str = Field(
        default="ml_prediction",
        description="Strategy for evaluating candidates (ml_prediction, simulation, hybrid)"
    )
    generation_strategy: str = Field(
        default="random",
        description="Strategy for generating candidates (random, bayesian, genetic, rl, generative)"
    )
    convergence_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Stop if best score exceeds this threshold"
    )
    early_stopping_patience: Optional[int] = Field(
        None,
        ge=1,
        description="Stop if no improvement for this many iterations"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "target_properties": {
                    "bandgap": {
                        "value": 2.0,
                        "tolerance": 0.2,
                        "weight": 1.0
                    },
                    "formation_energy": {
                        "max": -3.0,
                        "weight": 0.5
                    }
                },
                "constraints": {
                    "elements": ["Mo", "W", "S", "Se"],
                    "max_atoms": 20,
                    "dimensionality": 2
                },
                "max_iterations": 20,
                "candidates_per_iteration": 10,
                "generation_strategy": "bayesian"
            }
        }


# Request schemas

class DesignCampaignCreate(BaseModel):
    """Schema for creating a new design campaign."""
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Campaign name"
    )
    description: Optional[str] = Field(
        None,
        description="Campaign description and goals"
    )
    config: CampaignConfig = Field(
        ...,
        description="Campaign configuration"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata (tags, notes, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "TMD Bandgap Optimization",
                "description": "Find transition metal dichalcogenides with 2eV bandgap",
                "config": {
                    "target_properties": {
                        "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0}
                    },
                    "constraints": {
                        "elements": ["Mo", "W", "S", "Se"],
                        "dimensionality": 2
                    },
                    "max_iterations": 20,
                    "candidates_per_iteration": 10
                }
            }
        }


class DesignCampaignUpdate(BaseModel):
    """Schema for updating a design campaign."""
    name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=255
    )
    description: Optional[str] = None
    config: Optional[CampaignConfig] = None
    status: Optional[str] = Field(
        None,
        description="Campaign status (CREATED, RUNNING, PAUSED, COMPLETED, FAILED, CANCELLED)"
    )
    metadata: Optional[Dict[str, Any]] = None


class CampaignStepRequest(BaseModel):
    """
    Request to run one or more iterations of a campaign.

    Used to advance the campaign by running design loop iterations.
    """
    num_iterations: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of iterations to run"
    )
    override_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Temporary config overrides for this step"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "num_iterations": 1
            }
        }


# Response schemas

class DesignCampaignResponse(BaseModel):
    """Schema for design campaign in API responses."""
    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    description: Optional[str] = None
    status: str
    config: Dict[str, Any]
    max_iterations: int
    current_iteration: int
    best_score: Optional[float] = None
    best_structure_id: Optional[uuid.UUID] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "owner_id": "660e8400-e29b-41d4-a716-446655440000",
                "name": "TMD Bandgap Optimization",
                "description": "Find 2eV bandgap materials",
                "status": "RUNNING",
                "config": {
                    "target_properties": {
                        "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0}
                    },
                    "max_iterations": 20,
                    "candidates_per_iteration": 10
                },
                "max_iterations": 20,
                "current_iteration": 5,
                "best_score": 0.87,
                "best_structure_id": "770e8400-e29b-41d4-a716-446655440000",
                "created_at": "2025-01-15T10:00:00Z",
                "updated_at": "2025-01-15T10:30:00Z",
                "started_at": "2025-01-15T10:05:00Z",
                "completed_at": None,
                "duration_seconds": None
            }
        }


class DesignIterationResponse(BaseModel):
    """Schema for design iteration in API responses."""
    id: uuid.UUID
    campaign_id: uuid.UUID
    iteration_index: int
    created_structures: List[str] = Field(default_factory=list)
    evaluated_structures: List[str] = Field(default_factory=list)
    best_score_this_iter: Optional[float] = None
    best_structure_id_this_iter: Optional[uuid.UUID] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    strategy_used: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "880e8400-e29b-41d4-a716-446655440000",
                "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
                "iteration_index": 5,
                "created_structures": [
                    "990e8400-e29b-41d4-a716-446655440000",
                    "aa0e8400-e29b-41d4-a716-446655440000"
                ],
                "evaluated_structures": [
                    "990e8400-e29b-41d4-a716-446655440000",
                    "aa0e8400-e29b-41d4-a716-446655440000"
                ],
                "best_score_this_iter": 0.87,
                "best_structure_id_this_iter": "990e8400-e29b-41d4-a716-446655440000",
                "metrics": {
                    "scores": [0.87, 0.75, 0.82],
                    "mean_score": 0.81,
                    "max_score": 0.87,
                    "improvement_from_previous": 0.05,
                    "diversity_metric": 0.45
                },
                "strategy_used": "bayesian",
                "created_at": "2025-01-15T10:25:00Z",
                "completed_at": "2025-01-15T10:30:00Z",
                "duration_seconds": 300.0
            }
        }


class CampaignStepResponse(BaseModel):
    """
    Response after running campaign iterations.

    Contains the updated campaign and list of iterations executed.
    """
    campaign: DesignCampaignResponse
    iterations: List[DesignIterationResponse]
    message: str = Field(
        ...,
        description="Human-readable status message"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "campaign": {
                    "id": "550e8400-e29b-41d4-a716-446655440000",
                    "name": "TMD Optimization",
                    "status": "RUNNING",
                    "current_iteration": 6,
                    "best_score": 0.87
                },
                "iterations": [
                    {
                        "iteration_index": 5,
                        "best_score_this_iter": 0.87,
                        "metrics": {"mean_score": 0.81}
                    }
                ],
                "message": "Completed 1 iteration. Best score: 0.87"
            }
        }


class CampaignListResponse(BaseModel):
    """Response for listing campaigns."""
    campaigns: List[DesignCampaignResponse]
    total: int
    page: int
    page_size: int

    class Config:
        json_schema_extra = {
            "example": {
                "campaigns": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "name": "TMD Optimization",
                        "status": "RUNNING",
                        "current_iteration": 5,
                        "best_score": 0.87
                    }
                ],
                "total": 1,
                "page": 1,
                "page_size": 20
            }
        }


class CampaignSummaryResponse(BaseModel):
    """
    Summary statistics for a campaign.

    Provides aggregated metrics and insights about campaign progress.
    """
    campaign_id: uuid.UUID
    total_iterations: int
    total_structures_created: int
    total_structures_evaluated: int
    best_score_overall: Optional[float] = None
    best_structure_id: Optional[uuid.UUID] = None
    score_history: List[float] = Field(
        default_factory=list,
        description="Best score at each iteration"
    )
    mean_scores: List[float] = Field(
        default_factory=list,
        description="Mean score at each iteration"
    )
    improvements: List[Optional[float]] = Field(
        default_factory=list,
        description="Score improvement at each iteration"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "campaign_id": "550e8400-e29b-41d4-a716-446655440000",
                "total_iterations": 10,
                "total_structures_created": 100,
                "total_structures_evaluated": 100,
                "best_score_overall": 0.92,
                "best_structure_id": "770e8400-e29b-41d4-a716-446655440000",
                "score_history": [0.65, 0.70, 0.75, 0.82, 0.87, 0.89, 0.90, 0.91, 0.92, 0.92],
                "mean_scores": [0.60, 0.65, 0.70, 0.75, 0.80, 0.83, 0.85, 0.87, 0.88, 0.89],
                "improvements": [None, 0.05, 0.05, 0.07, 0.05, 0.02, 0.01, 0.01, 0.01, 0.0]
            }
        }

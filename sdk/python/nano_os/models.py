"""
Data models for NANO-OS SDK.

Session 28: Python SDK and Workflow DSL
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


# Structure Models

class Structure(BaseModel):
    """Material structure."""
    id: UUID
    owner_id: UUID
    composition: str
    lattice_type: Optional[str] = None
    num_atoms: Optional[int] = None
    space_group: Optional[int] = None
    dimensionality: Optional[int] = None
    structure_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class StructureCreate(BaseModel):
    """Request to create a structure."""
    composition: str = Field(..., min_length=1)
    lattice_type: Optional[str] = None
    num_atoms: Optional[int] = None
    space_group: Optional[int] = None
    dimensionality: Optional[int] = None
    structure_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


# Job Models

class Job(BaseModel):
    """Computational job."""
    id: UUID
    owner_id: UUID
    structure_id: UUID
    job_type: str
    status: str
    priority: int
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    class Config:
        from_attributes = True


class DFTJobCreate(BaseModel):
    """Request to create a DFT job."""
    structure_id: UUID
    functional: str = "PBE"
    kpoints_density: float = 0.03
    energy_cutoff: Optional[float] = None
    is_relaxation: bool = True
    priority: int = 5
    metadata: Optional[Dict[str, Any]] = None


class MLPredictionCreate(BaseModel):
    """Request to create an ML prediction job."""
    structure_id: UUID
    model_name: str
    properties: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# Campaign Models

class Campaign(BaseModel):
    """Design campaign."""
    id: UUID
    owner_id: UUID
    name: str
    description: Optional[str] = None
    status: str
    config: Dict[str, Any]
    max_iterations: int
    current_iteration: int
    best_score: Optional[float] = None
    best_structure_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class CampaignCreate(BaseModel):
    """Request to create a campaign."""
    name: str = Field(..., min_length=1)
    description: Optional[str] = None
    config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class CampaignStepRequest(BaseModel):
    """Request to run campaign iterations."""
    num_iterations: int = Field(default=1, ge=1, le=100)
    override_config: Optional[Dict[str, Any]] = None


# Experiment Models

class Instrument(BaseModel):
    """Lab instrument."""
    id: UUID
    owner_id: UUID
    name: str
    adapter_type: str
    connection_info: Dict[str, Any]
    capabilities: Optional[List[str]] = None
    status: str
    created_at: datetime
    updated_at: datetime
    last_connected_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class InstrumentCreate(BaseModel):
    """Request to create an instrument."""
    name: str = Field(..., min_length=1)
    adapter_type: str
    connection_info: Dict[str, Any]
    capabilities: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class Experiment(BaseModel):
    """Experimental run."""
    id: UUID
    owner_id: UUID
    instrument_id: UUID
    linked_structure_id: Optional[UUID] = None
    type: str
    status: str
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    external_job_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class ExperimentCreate(BaseModel):
    """Request to create an experiment."""
    instrument_id: UUID
    linked_structure_id: Optional[UUID] = None
    type: str
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


# Property Models

class MaterialProperties(BaseModel):
    """Material properties from computation or experiment."""
    structure_id: UUID
    bandgap: Optional[float] = None
    formation_energy: Optional[float] = None
    band_structure: Optional[Dict[str, Any]] = None
    dos: Optional[Dict[str, Any]] = None
    optical_properties: Optional[Dict[str, Any]] = None
    mechanical_properties: Optional[Dict[str, Any]] = None
    source: str  # "dft", "ml_prediction", "experiment"
    metadata: Optional[Dict[str, Any]] = None


# Response Models

class PaginatedResponse(BaseModel):
    """Paginated list response."""
    items: List[Any]
    total: int
    page: int
    page_size: int


class JobStatusResponse(BaseModel):
    """Job status response."""
    id: UUID
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None
    error_message: Optional[str] = None


class CampaignSummary(BaseModel):
    """Campaign summary statistics."""
    campaign_id: UUID
    total_iterations: int
    total_structures_created: int
    total_structures_evaluated: int
    best_score_overall: Optional[float] = None
    best_structure_id: Optional[UUID] = None
    score_history: List[float] = Field(default_factory=list)
    mean_scores: List[float] = Field(default_factory=list)
    improvements: List[Optional[float]] = Field(default_factory=list)

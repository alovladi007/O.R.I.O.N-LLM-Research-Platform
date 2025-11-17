"""
Experiment and Instrument API schemas.

Session 21: Lab Integration & Experiment Management
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# ============================================================================
# Instrument Schemas
# ============================================================================

class InstrumentCreate(BaseModel):
    """Schema for creating an instrument."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    adapter_type: str = Field(..., description="MOCK, REST, OPCUA, SSH, CUSTOM")
    connection_info: Dict[str, Any] = Field(..., description="Connection configuration")
    capabilities: Optional[List[str]] = Field(default_factory=list)
    location: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class InstrumentUpdate(BaseModel):
    """Schema for updating an instrument."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    connection_info: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None
    status: Optional[str] = None
    location: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class InstrumentResponse(BaseModel):
    """Schema for instrument response."""
    id: uuid.UUID
    owner_id: uuid.UUID
    name: str
    description: Optional[str] = None
    adapter_type: str
    connection_info: Dict[str, Any]
    capabilities: List[str]
    status: str
    location: Optional[str] = None
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============================================================================
# Experiment Schemas
# ============================================================================

class ExperimentCreate(BaseModel):
    """Schema for creating an experiment."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    type: str = Field(..., description="synthesis, measurement, characterization, testing")
    instrument_id: uuid.UUID
    linked_structure_id: Optional[uuid.UUID] = None
    parameters: Dict[str, Any] = Field(..., description="Experiment parameters")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ExperimentUpdate(BaseModel):
    """Schema for updating an experiment."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ExperimentResponse(BaseModel):
    """Schema for experiment response."""
    id: uuid.UUID
    owner_id: uuid.UUID
    instrument_id: uuid.UUID
    linked_structure_id: Optional[uuid.UUID] = None
    name: str
    description: Optional[str] = None
    type: str
    status: str
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    external_job_id: Optional[str] = None
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExperimentRunRequest(BaseModel):
    """Schema for running an experiment."""
    async_execution: bool = Field(
        default=True,
        description="Run asynchronously (True) or wait for completion (False)"
    )


class ExperimentRunResponse(BaseModel):
    """Schema for experiment run response."""
    experiment_id: uuid.UUID
    status: str
    message: str
    external_job_id: Optional[str] = None

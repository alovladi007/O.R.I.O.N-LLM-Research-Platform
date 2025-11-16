"""Simulation job and result schemas for API requests and responses."""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class SimulationJobCreate(BaseModel):
    """Schema for creating a simulation job."""
    structure_id: uuid.UUID = Field(..., description="ID of the structure to simulate")
    workflow_template_id: uuid.UUID = Field(..., description="ID of the workflow template")
    name: Optional[str] = Field(None, max_length=255, description="Job name")
    description: Optional[str] = Field(None, description="Job description")
    priority: int = Field(default=5, ge=0, le=20, description="Job priority (0-20)")
    parameters: dict = Field(default_factory=dict, description="Custom parameters (override template defaults)")
    resources: Optional[dict] = Field(None, description="Resource allocation")
    metadata: Optional[dict] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "workflow_template_id": "223e4567-e89b-12d3-a456-426614174000",
                "name": "MoS2 relaxation",
                "priority": 10,
                "parameters": {"ecutwfc": 60}
            }
        }


class SimulationJobUpdate(BaseModel):
    """Schema for updating a simulation job."""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    priority: Optional[int] = Field(None, ge=0, le=20)
    status: Optional[str] = None  # Typically changed by worker, not user
    metadata: Optional[dict] = None


class SimulationJobResponse(BaseModel):
    """Schema for simulation job in API responses."""
    id: uuid.UUID
    structure_id: uuid.UUID
    workflow_template_id: uuid.UUID
    name: Optional[str] = None
    description: Optional[str] = None
    status: str
    priority: int
    engine: str
    parameters: dict = Field(default_factory=dict)
    resources: Optional[dict] = None
    progress: Optional[float] = None
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    submitted_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    worker_id: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    class Config:
        from_attributes = True


class SimulationResultResponse(BaseModel):
    """Schema for simulation result in API responses."""
    id: uuid.UUID
    simulation_job_id: uuid.UUID
    summary: dict = Field(default_factory=dict, description="High-level results")
    artifacts_path: Optional[str] = None
    artifacts: Optional[list] = None
    convergence_reached: Optional[bool] = None
    quality_score: Optional[float] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime

    class Config:
        from_attributes = True

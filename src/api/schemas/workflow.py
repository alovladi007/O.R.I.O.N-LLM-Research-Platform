"""Workflow template schemas for API requests and responses."""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class WorkflowTemplateCreate(BaseModel):
    """Schema for creating a workflow template."""
    name: str = Field(..., min_length=1, max_length=255, description="Unique workflow name")
    display_name: str = Field(..., min_length=1, max_length=255, description="Human-readable name")
    description: Optional[str] = Field(None, description="Workflow description")
    engine: str = Field(..., max_length=50, description="Simulation engine (QE, VASP, LAMMPS, MOCK)")
    engine_version: Optional[str] = Field(None, max_length=50, description="Engine version")
    category: Optional[str] = Field(None, max_length=100, description="Category (DFT, MD, Phonon, etc.)")
    default_parameters: dict = Field(default_factory=dict, description="Default parameters")
    default_resources: Optional[dict] = Field(None, description="Default resource allocation")
    is_public: bool = Field(default=True, description="Whether template is public")
    documentation_url: Optional[str] = Field(None, max_length=500)

    class Config:
        json_schema_extra = {
            "example": {
                "name": "DFT_relaxation_QE",
                "display_name": "DFT Relaxation (Quantum ESPRESSO)",
                "description": "Geometry optimization using DFT",
                "engine": "QE",
                "category": "DFT",
                "default_parameters": {
                    "ecutwfc": 50,
                    "k_points": [4, 4, 4],
                    "conv_thr": 1e-6
                }
            }
        }


class WorkflowTemplateUpdate(BaseModel):
    """Schema for updating a workflow template."""
    display_name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    engine_version: Optional[str] = None
    category: Optional[str] = None
    default_parameters: Optional[dict] = None
    default_resources: Optional[dict] = None
    is_active: Optional[bool] = None
    is_public: Optional[bool] = None
    documentation_url: Optional[str] = None


class WorkflowTemplateResponse(BaseModel):
    """Schema for workflow template in API responses."""
    id: uuid.UUID
    name: str
    display_name: str
    description: Optional[str] = None
    engine: str
    engine_version: Optional[str] = None
    category: Optional[str] = None
    default_parameters: dict = Field(default_factory=dict)
    default_resources: Optional[dict] = None
    is_active: bool
    is_public: bool
    usage_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

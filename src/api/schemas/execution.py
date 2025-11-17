"""
Pydantic schemas for execution profiles API endpoints.

Session 27: HPC and Cloud Scaling
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class ExecutionProfileCreate(BaseModel):
    """Request to create an execution profile."""
    name: str = Field(..., description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    backend_type: str = Field(..., description="Backend type: LOCAL, SLURM, SSH, CLOUD_API, CUSTOM")
    config: Dict[str, Any] = Field(..., description="Backend-specific configuration")
    default_resources: Optional[Dict[str, Any]] = Field(None, description="Default resource requests")
    is_active: bool = Field(default=True, description="Whether profile is active")
    is_default: bool = Field(default=False, description="Whether this is the default profile")
    metadata: Optional[Dict[str, Any]] = None


class ExecutionProfileUpdate(BaseModel):
    """Request to update an execution profile."""
    name: Optional[str] = None
    description: Optional[str] = None
    backend_type: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    default_resources: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    is_default: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class ExecutionProfileResponse(BaseModel):
    """Execution profile response."""
    id: UUID
    owner_id: UUID
    name: str
    description: Optional[str]
    backend_type: str
    config: Dict[str, Any]
    default_resources: Optional[Dict[str, Any]]
    is_active: bool
    is_default: bool
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

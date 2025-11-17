"""
Pydantic schemas for quantum materials API endpoints.

Session 24: Quantum Materials Vertical
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class QuantumMaterialCreate(BaseModel):
    """Request to create a quantum material."""
    structure_id: UUID = Field(..., description="Base structure ID")
    name: str = Field(..., description="Material name")
    focus: str = Field(..., description="Focus area: TOPOLOGICAL_INSULATOR, SUPERCONDUCTOR_CANDIDATE, etc.")
    description: Optional[str] = Field(None, description="Material description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="SOC strength, phases, etc.")


class QuantumMaterialUpdate(BaseModel):
    """Request to update a quantum material."""
    name: Optional[str] = None
    focus: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QuantumMaterialResponse(BaseModel):
    """Quantum material response."""
    id: UUID
    owner_id: UUID
    structure_id: UUID
    name: str
    focus: str
    description: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WannierSetupCreate(BaseModel):
    """Request to create a Wannier setup."""
    structure_id: UUID = Field(..., description="Structure ID")
    name: str = Field(..., description="Setup name")
    target_bands_range: Optional[Dict[str, Any]] = Field(None, description="Min/max band indices")
    initial_projections: Optional[Dict[str, Any]] = Field(None, description="Projection specifications")
    metadata: Optional[Dict[str, Any]] = None


class WannierSetupUpdate(BaseModel):
    """Request to update a Wannier setup."""
    name: Optional[str] = None
    target_bands_range: Optional[Dict[str, Any]] = None
    initial_projections: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class WannierSetupResponse(BaseModel):
    """Wannier setup response."""
    id: UUID
    structure_id: UUID
    name: str
    target_bands_range: Optional[Dict[str, Any]]
    initial_projections: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True

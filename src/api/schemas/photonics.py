"""
Pydantic schemas for photonics API endpoints.

Session 22: Photonics Vertical
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class PhotonicsStructureCreate(BaseModel):
    """Request to create a photonics structure."""
    base_structure_id: UUID = Field(..., description="Base structure ID")
    type: str = Field(..., description="Structure type: WAVEGUIDE, PHOTONIC_CRYSTAL, RESONATOR, COUPLER")
    geometry_params: Dict[str, Any] = Field(..., description="Geometry parameters (thickness, width, period, etc.)")
    wavelength_range: Optional[Dict[str, float]] = Field(None, description="Wavelength range {min, max} in nm")
    metadata: Optional[Dict[str, Any]] = None


class PhotonicsStructureUpdate(BaseModel):
    """Request to update a photonics structure."""
    geometry_params: Optional[Dict[str, Any]] = None
    wavelength_range: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class PhotonicsStructureResponse(BaseModel):
    """Photonics structure response."""
    id: UUID
    base_structure_id: UUID
    type: str
    geometry_params: Dict[str, Any]
    wavelength_range: Optional[Dict[str, float]]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PhotonicsSimulationJobCreate(BaseModel):
    """Request to create a photonics simulation job."""
    photonics_structure_id: UUID = Field(..., description="Photonics structure to simulate")
    engine: str = Field(default="FDTD_STUB", description="Simulation engine")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Simulation parameters")
    metadata: Optional[Dict[str, Any]] = None


class PhotonicsSimulationJobResponse(BaseModel):
    """Photonics simulation job response."""
    id: UUID
    photonics_structure_id: UUID
    engine: str
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]]
    status: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True

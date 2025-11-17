"""
Pydantic schemas for metamaterials API endpoints.

Session 25: Metamaterials Vertical
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class MetamaterialUnitCellCreate(BaseModel):
    """Request to create a metamaterial unit cell."""
    base_structure_id: UUID = Field(..., description="Base structure ID")
    name: str = Field(..., description="Unit cell name")
    type: str = Field(..., description="Type: ELECTROMAGNETIC, MECHANICAL, ACOUSTIC, THERMAL")
    lattice_vectors: Optional[Dict[str, Any]] = Field(None, description="Periodicity vectors")
    fill_pattern: Optional[Dict[str, Any]] = Field(None, description="Fill pattern description")
    target_frequency_range: Optional[Dict[str, Any]] = Field(None, description="Target frequency range")
    metadata: Optional[Dict[str, Any]] = None


class MetamaterialUnitCellUpdate(BaseModel):
    """Request to update a metamaterial unit cell."""
    name: Optional[str] = None
    type: Optional[str] = None
    lattice_vectors: Optional[Dict[str, Any]] = None
    fill_pattern: Optional[Dict[str, Any]] = None
    target_frequency_range: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class MetamaterialUnitCellResponse(BaseModel):
    """Metamaterial unit cell response."""
    id: UUID
    owner_id: UUID
    base_structure_id: UUID
    name: str
    type: str
    lattice_vectors: Optional[Dict[str, Any]]
    fill_pattern: Optional[Dict[str, Any]]
    target_frequency_range: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MetamaterialSimulationJobCreate(BaseModel):
    """Request to create a metamaterial simulation job."""
    unit_cell_id: UUID = Field(..., description="Metamaterial unit cell to simulate")
    name: str = Field(..., description="Job name")
    engine: str = Field(default="COMSOL_STUB", description="Simulation engine")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Simulation parameters")


class MetamaterialSimulationJobResponse(BaseModel):
    """Metamaterial simulation job response."""
    id: UUID
    unit_cell_id: UUID
    name: str
    status: str
    engine: str
    parameters: Dict[str, Any]
    results: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

"""
Pydantic schemas for battery materials API endpoints.

Session 23: Battery Materials Vertical
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class BatteryMaterialCreate(BaseModel):
    """Request to create a battery material."""
    structure_id: UUID = Field(..., description="Structure ID (cathode, anode, etc.)")
    name: str = Field(..., description="Material name")
    role: str = Field(..., description="Role: CATHODE, ANODE, ELECTROLYTE, SEPARATOR")
    redox_species: str = Field(..., description="Redox species (Li, Na, K, etc.)")
    reference_state_info: Optional[Dict[str, Any]] = Field(None, description="Reference state information")
    metadata: Optional[Dict[str, Any]] = None


class BatteryMaterialUpdate(BaseModel):
    """Request to update a battery material."""
    name: Optional[str] = None
    role: Optional[str] = None
    redox_species: Optional[str] = None
    reference_state_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BatteryMaterialResponse(BaseModel):
    """Battery material response."""
    id: UUID
    owner_id: UUID
    structure_id: UUID
    name: str
    role: str
    redox_species: str
    reference_state_info: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

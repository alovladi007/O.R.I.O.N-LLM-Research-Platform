"""
Pydantic schemas for Phase-Change Memory (PCM) materials API endpoints.

Session 26: PCM Vertical
"""

from typing import Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class PCMMaterialCreate(BaseModel):
    """Request to create a PCM material."""
    structure_crystalline_id: UUID = Field(..., description="Crystalline structure ID")
    structure_amorphous_id: UUID = Field(..., description="Amorphous structure ID")
    name: str = Field(..., description="Material name")
    composition: str = Field(..., description="Composition (e.g., Ge2Sb2Te5)")
    description: Optional[str] = Field(None, description="Material description")
    metadata: Optional[Dict[str, Any]] = None


class PCMMaterialUpdate(BaseModel):
    """Request to update a PCM material."""
    name: Optional[str] = None
    composition: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PCMMaterialResponse(BaseModel):
    """PCM material response."""
    id: UUID
    owner_id: UUID
    structure_crystalline_id: UUID
    structure_amorphous_id: UUID
    name: str
    composition: str
    description: Optional[str]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

"""Structure schemas for API requests and responses."""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class StructureCreate(BaseModel):
    """Schema for creating a new structure."""
    material_id: uuid.UUID = Field(..., description="ID of the parent material")
    name: Optional[str] = Field(None, max_length=255, description="Structure name")
    description: Optional[str] = Field(None, description="Structure description")
    format: str = Field(..., description="File format (CIF, POSCAR, XYZ, INTERNAL)")
    raw_text: str = Field(..., description="Raw structure file content")
    source: str = Field(default="USER_UPLOADED", description="Structure source")
    metadata: Optional[dict] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "material_id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "MoS2 1H polymorph",
                "format": "CIF",
                "raw_text": "data_MoS2\n_cell_length_a 3.16...",
                "source": "USER_UPLOADED"
            }
        }


class StructureUpdate(BaseModel):
    """Schema for updating a structure."""
    name: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    metadata: Optional[dict] = None


class StructureParseRequest(BaseModel):
    """Schema for structure parsing request."""
    text: str = Field(..., description="Raw structure file content")
    format: str = Field(..., description="File format (CIF, POSCAR, XYZ)")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Mo 1.0 1.0 1.0\nS 2.0 2.0 2.0",
                "format": "XYZ"
            }
        }


class StructureParseResponse(BaseModel):
    """Schema for parsed structure data."""
    formula: str = Field(..., description="Chemical formula")
    num_atoms: int = Field(..., description="Number of atoms")
    dimensionality: int = Field(..., description="0=molecule, 1=1D, 2=2D, 3=bulk")
    lattice: dict = Field(..., description="Lattice vectors and parameters")
    atoms: list[dict] = Field(..., description="Atomic species and positions")
    lattice_parameters: dict = Field(..., description="a, b, c, alpha, beta, gamma, volume")


class StructureResponse(BaseModel):
    """Schema for structure in API responses."""
    id: uuid.UUID
    material_id: uuid.UUID
    name: Optional[str] = None
    description: Optional[str] = None
    format: str
    source: str
    formula: Optional[str] = None
    num_atoms: Optional[int] = None
    dimensionality: Optional[int] = None
    lattice_parameters: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

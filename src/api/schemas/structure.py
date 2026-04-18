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
    symprec: float = Field(
        0.01,
        ge=0.0,
        le=1.0,
        description=(
            "Symmetry tolerance (Å) passed to pymatgen's SpacegroupAnalyzer. "
            "Default 0.01 matches the Materials Project convention. Loosen "
            "to e.g. 0.1 for DFT-relaxed structures with residual noise."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "data_Si\n_cell_length_a 5.43\n...",
                "format": "CIF",
                "symprec": 0.01,
            }
        }


class StructureParseResponse(BaseModel):
    """
    Parsed-structure payload. Symmetry fields are populated when pymatgen
    can determine a spacegroup; they default to ``None`` for molecular
    (XYZ) inputs or on symprec-too-tight failures.
    """
    formula: str = Field(..., description="Reduced chemical formula (e.g. 'SiO2').")
    num_atoms: int = Field(..., ge=1, description="Total atoms in the unit cell.")
    dimensionality: int = Field(
        ..., ge=0, le=3,
        description="0=molecule, 1=chain, 2=layer, 3=bulk.",
    )
    lattice: dict = Field(..., description="Lattice vectors and derived params.")
    atoms: list[dict] = Field(..., description="List of {species, position} atoms.")
    lattice_parameters: dict = Field(
        ..., description="a, b, c (Å), alpha, beta, gamma (deg), volume (Å³)."
    )
    space_group: Optional[str] = Field(
        None, description="International spacegroup symbol (e.g. 'Fd-3m')."
    )
    space_group_number: Optional[int] = Field(
        None, ge=1, le=230, description="International spacegroup number 1–230.",
    )
    structure_hash: str = Field(
        ..., min_length=64, max_length=64,
        description="Deterministic 64-char SHA-256 fingerprint.",
    )


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
    space_group: Optional[str] = None
    space_group_number: Optional[int] = None
    structure_hash: Optional[str] = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

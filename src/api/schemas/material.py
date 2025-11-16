"""Material schemas for API requests and responses."""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class MaterialCreate(BaseModel):
    """Schema for creating a new material."""
    name: str = Field(..., min_length=1, max_length=255, description="Material name")
    formula: str = Field(..., min_length=1, max_length=100, description="Chemical formula")
    description: Optional[str] = Field(None, description="Material description")
    tags: Optional[list[str]] = Field(default_factory=list, description="Tags for categorization")
    composition: Optional[dict] = Field(None, description="Element composition (e.g., {'Mo': 1, 'S': 2})")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")
    source: Optional[str] = Field(None, max_length=255, description="Data source")
    external_id: Optional[str] = Field(None, max_length=255, description="External database ID")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Molybdenum Disulfide",
                "formula": "MoS2",
                "description": "2D semiconductor with interesting optical properties",
                "tags": ["2D material", "semiconductor", "TMD"],
                "composition": {"Mo": 1, "S": 2},
                "source": "User Upload"
            }
        }


class MaterialUpdate(BaseModel):
    """Schema for updating a material."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    formula: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    composition: Optional[dict] = None
    metadata: Optional[dict] = None
    source: Optional[str] = None
    external_id: Optional[str] = None


class MaterialResponse(BaseModel):
    """Schema for material in API responses."""
    id: uuid.UUID
    name: str
    formula: str
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    composition: Optional[dict] = None
    metadata: dict = Field(default_factory=dict)
    source: Optional[str] = None
    external_id: Optional[str] = None
    structure_count: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MaterialListResponse(BaseModel):
    """Schema for list of materials with pagination."""
    items: list[MaterialResponse]
    total: int
    page: int
    page_size: int
    pages: int

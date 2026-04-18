"""API schemas for the Session 1.3 bulk property importer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class StructureRefSpec(BaseModel):
    """Describes how rows' structure references are resolved."""

    column: str = Field(..., description="CSV column carrying the identifier.")
    kind: str = Field(
        "uuid",
        description="'uuid' for ORION Structure.id, 'external_id' for MP/ChEMBL/etc.",
    )
    namespace: Optional[str] = Field(
        None,
        description="Only used when kind=external_id (e.g. 'materials_project').",
    )


class BulkPropertyImportRequest(BaseModel):
    """
    POST body for the synchronous property-import endpoint.

    The CSV lives in `csv_text` (inline for now; Session 2.1 adds the
    multipart/MinIO variant). The `mapping` block describes how to
    interpret the columns.
    """

    csv_text: str = Field(
        ...,
        min_length=1,
        description="Raw CSV body; first line is the header.",
    )
    mapping: Dict[str, Any] = Field(
        ...,
        description=(
            "Column/property mapping. Required keys: "
            "structure_ref (StructureRefSpec), property, value_column, unit. "
            "Optional: method, conditions."
        ),
    )


class BulkImportJobResponse(BaseModel):
    """Shape returned by POST /properties/bulk and GET /jobs/bulk-import/{id}."""

    id: UUID
    owner_id: UUID
    kind: str
    status: str
    rows_total: int
    rows_ok: int
    rows_failed: int
    mapping: Dict[str, Any]
    errors: List[Dict[str, Any]]
    error_message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True

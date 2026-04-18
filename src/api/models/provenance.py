"""
Provenance tracking model.

Shape matches the 003_add_provenance_tracking migration. Every event of
interest (job created / started / completed, prediction produced, structure
uploaded, etc.) appends one `ProvenanceRecord`. Queries build a full
ancestry graph by filtering on `(entity_type, entity_id)` and ordering by
`timestamp`.

Restored in Phase 1 / Session 1.2. The full provenance-graph walker and
reproducibility features land in Phase 12.
"""

import enum
import uuid
from datetime import datetime
from typing import Any, Dict

from sqlalchemy import DateTime, String
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class EntityType(str, enum.Enum):
    """Kinds of entities that appear in the provenance graph."""

    JOB = "JOB"
    PREDICTION = "PREDICTION"
    STRUCTURE = "STRUCTURE"
    MATERIAL = "MATERIAL"
    CAMPAIGN = "CAMPAIGN"
    ITERATION = "ITERATION"
    EXPERIMENT = "EXPERIMENT"
    WORKFLOW = "WORKFLOW"
    USER = "USER"
    MODEL = "MODEL"


class EventType(str, enum.Enum):
    """Events we record against entities."""

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    PREDICTED = "PREDICTED"
    UPLOADED = "UPLOADED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"


class ProvenanceRecord(Base):
    """
    One row per recorded event.

    Fields deliberately thin: `details` is a free-form JSON blob containing
    engine identifiers, worker info, code versions, or anything else the
    caller wants to capture. Phase 12 will introduce a typed overlay for
    common details keys.
    """

    __tablename__ = "provenance_records"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )

    entity_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
        comment="Kind of entity (JOB, PREDICTION, STRUCTURE, ...).",
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True,
        comment="Kind of event (CREATED, STARTED, COMPLETED, ...).",
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
        comment="When the event occurred (UTC).",
    )

    details: Mapped[Dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict,
        comment="Free-form event details (engine, worker, host info, etc.).",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<ProvenanceRecord(id={self.id}, {self.entity_type}:{self.entity_id}, "
            f"event={self.event_type}, t={self.timestamp.isoformat() if self.timestamp else None})>"
        )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "entity_type": self.entity_type,
            "entity_id": str(self.entity_id),
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "details": self.details or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

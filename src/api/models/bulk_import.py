"""
BulkImportJob — tracks bulk ingestion of properties / structures / etc.

Phase 1 / Session 1.3 uses this for property imports; Session 1.5 will
extend usage to structure seed data. Rows are stored as a manifest
referencing the original payload (in MinIO once Session 2.1 lands) plus
a rolling per-row error log.
"""

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class BulkImportKind(str, enum.Enum):
    """Which kind of entity the job is ingesting."""

    PROPERTY = "PROPERTY"
    STRUCTURE = "STRUCTURE"


class BulkImportStatus(str, enum.Enum):
    """Lifecycle state."""

    PENDING = "PENDING"        # created but not started
    RUNNING = "RUNNING"        # worker is processing rows
    COMPLETED = "COMPLETED"    # done — may have partial errors
    FAILED = "FAILED"          # hard failure before per-row processing
    CANCELLED = "CANCELLED"    # user-requested halt


class BulkImportJob(Base):
    """
    A single bulk-import request, with rolling progress + per-row errors.

    `errors` is the append-only log of rejected rows:

        [
          {"row": 42, "reason": "unit 'Pa' not accepted for bandgap",
           "raw": {"mp_id": "mp-1234", "e_form": 0.12}},
          ...
        ]

    Keep `errors` bounded; Session 1.3 caps at `ERROR_LOG_CAP` rows so a
    10k-row input with a bad schema doesn't blow out a row.
    """

    __tablename__ = "bulk_import_jobs"

    ERROR_LOG_CAP: int = 500  # class constant, enforced by the importer

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )

    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    kind: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        default=BulkImportKind.PROPERTY.value,
        comment="PROPERTY | STRUCTURE",
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        default=BulkImportStatus.PENDING.value,
    )

    # Mapping spec supplied by the user. Shape documented in
    # src.api.schemas.bulk_import.BulkPropertyImportRequest.
    mapping: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )

    # Source artefact reference. For Session 1.3 the full CSV text lives
    # here inline; Session 2.1 moves it to a MinIO path.
    source_kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default="inline",
        comment="inline | minio | url",
    )
    source_payload: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Raw CSV when source_kind=inline; MinIO key when minio.",
    )

    # Progress counters.
    rows_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_ok: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rows_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    errors: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list,
        comment="Capped list of {row, reason, raw} — see ERROR_LOG_CAP.",
    )

    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Top-level error message for FAILED jobs (schema error etc.).",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow,
        index=True,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    owner: Mapped["User"] = relationship(  # noqa: F821
        "User", foreign_keys=[owner_id], lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<BulkImportJob(id={self.id}, kind={self.kind}, "
            f"status={self.status}, ok={self.rows_ok}/{self.rows_total})>"
        )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "owner_id": str(self.owner_id),
            "kind": self.kind,
            "status": self.status,
            "mapping": self.mapping,
            "rows_total": self.rows_total,
            "rows_ok": self.rows_ok,
            "rows_failed": self.rows_failed,
            "errors": self.errors or [],
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

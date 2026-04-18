"""
Experiment and Instrument models (lab integration).

No existing Alembic migration covered these — the legacy `src/experiments/`
code path that would have created them was deleted in Session 0.2. This
module defines the Python-side shape that the agent / experiment routers
already expect; a corresponding Alembic migration is created in this same
Session 1.2 (`009_add_experiments_and_extra_metadata`).

Phase 9 wires the lab-control real behavior on top of these tables;
until then the instrument adapter defaults to `MOCK`.
"""

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class InstrumentStatus(str, enum.Enum):
    """Operational status of a physical or simulated instrument."""

    IDLE = "IDLE"
    BUSY = "BUSY"
    ERROR = "ERROR"
    OFFLINE = "OFFLINE"
    MAINTENANCE = "MAINTENANCE"


class ExperimentStatus(str, enum.Enum):
    """Lifecycle state of an experiment run."""

    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Instrument(Base):
    """A lab instrument or its mock/simulation adapter."""

    __tablename__ = "instruments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    adapter_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="MOCK",
        comment="Adapter: MOCK | REST | OPCUA | SSH | CUSTOM.",
    )
    connection_info: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Adapter-specific connection details (URL, creds ref, etc.).",
    )
    capabilities: Mapped[List[str]] = mapped_column(
        ARRAY(String(100)), nullable=False, default=list,
        comment="List of capability tags (xrd, sem, afm, raman, xps, ...).",
    )

    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=InstrumentStatus.IDLE.value, index=True,
    )
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSONB, nullable=True, default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    experiments: Mapped[List["ExperimentRun"]] = relationship(
        "ExperimentRun",
        back_populates="instrument",
        cascade="all, delete-orphan",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return f"<Instrument(id={self.id}, name={self.name!r}, status={self.status})>"


class ExperimentRun(Base):
    """One execution of an experiment recipe on a specific instrument."""

    __tablename__ = "experiment_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    linked_structure_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("structures.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment="synthesis | measurement | characterization | testing",
    )

    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=ExperimentStatus.PENDING.value, index=True,
    )
    parameters: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )
    results: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    external_job_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Optional external tracking ID (LIMS / vendor system).",
    )

    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSONB, nullable=True, default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    instrument: Mapped[Instrument] = relationship(
        "Instrument", back_populates="experiments", lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<ExperimentRun(id={self.id}, name={self.name!r}, "
            f"status={self.status}, type={self.type})>"
        )

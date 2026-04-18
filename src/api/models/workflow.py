"""
WorkflowTemplate model — reusable parameter set for a simulation engine.

Shape matches the 001_initial_schema migration (`workflow_templates` table).
A WorkflowTemplate is a named, versioned bundle of:

- the engine to invoke (mock / qe / lammps / continuum / mesoscale)
- default parameters the user can override at job submission
- default resource asks (cores, memory, walltime)
- optional JSON Schema for parameter validation

Restored in Phase 1 / Session 1.2 — this file previously didn't exist, which
was blocking every import of `src.api.models`.
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class WorkflowTemplate(Base):
    """
    Versioned description of a reusable simulation workflow.

    A SimulationJob references a template and layers its own parameter
    overrides on top of `default_parameters`. Templates are owned globally
    (no owner_id) by design — they're engine definitions, not user data.
    Seed data in Phase 1 / Session 1.5 will populate the common ones
    (QE relax, QE static, LAMMPS NVT, etc.).
    """

    __tablename__ = "workflow_templates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True, unique=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    engine: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Engine name: mock, qe, lammps, continuum, mesoscale, ...",
    )
    engine_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    category: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        index=True,
        comment="Grouping: relax, static, md, bands, ...",
    )

    default_parameters: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict,
        comment="Engine-specific parameter defaults; overridable per-job.",
    )
    default_resources: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True,
        comment="Default resource ask: cores, memory_gb, walltime_minutes.",
    )

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    parameter_schema: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True,
        comment="Optional JSON Schema validating user-supplied parameters.",
    )
    documentation_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    usage_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Reverse side of the SimulationJob.workflow_template relationship.
    simulation_jobs: Mapped[List["SimulationJob"]] = relationship(  # noqa: F821
        "SimulationJob",
        back_populates="workflow_template",
        lazy="noload",  # templates rarely care about their jobs at load time
    )

    def __repr__(self) -> str:
        return (
            f"<WorkflowTemplate(id={self.id}, name={self.name!r}, "
            f"engine={self.engine!r}, category={self.category!r})>"
        )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "category": self.category,
            "default_parameters": self.default_parameters,
            "default_resources": self.default_resources,
            "is_active": self.is_active,
            "is_public": self.is_public,
            "parameter_schema": self.parameter_schema,
            "documentation_url": self.documentation_url,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

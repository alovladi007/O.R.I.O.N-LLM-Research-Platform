"""Workflow-run models — one row per DAG execution (Session 2.4).

Two tables:

- ``workflow_runs`` — one row per submitted spec. Stores the raw spec
  (JSONB) + aggregate status + workflow.json manifest path once done.
- ``workflow_run_steps`` — one row per resolved step. Status + link to
  the underlying SimulationJob (1:1).

The runtime tick task reads + writes these rows. The API router also
reads them to produce aggregate status.

These tables are independent of ``workflow_templates`` (the CRUD
router from earlier sessions), which continues to serve reusable
single-step parameter bundles.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class WorkflowRunStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            WorkflowRunStatus.COMPLETED,
            WorkflowRunStatus.FAILED,
            WorkflowRunStatus.CANCELLED,
        )


class WorkflowStepStatus(str, enum.Enum):
    PENDING = "PENDING"
    DISPATCHABLE = "DISPATCHABLE"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    SKIPPED = "SKIPPED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            WorkflowStepStatus.COMPLETED,
            WorkflowStepStatus.FAILED,
            WorkflowStepStatus.CANCELLED,
            WorkflowStepStatus.SKIPPED,
        )


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=WorkflowRunStatus.PENDING.value,
        index=True,
    )

    # The raw spec the caller submitted. We keep it verbatim so a later
    # "rerun" path doesn't need the client to resend.
    spec: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Aggregated manifest (workflow.json) written after terminal.
    manifest: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    steps: Mapped[List["WorkflowRunStep"]] = relationship(
        "WorkflowRunStep",
        back_populates="workflow_run",
        cascade="all, delete-orphan",
        order_by="WorkflowRunStep.topo_index",
    )

    def __repr__(self) -> str:
        return f"<WorkflowRun(id={self.id}, name={self.name!r}, status={self.status})>"


class WorkflowRunStep(Base):
    __tablename__ = "workflow_run_steps"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    workflow_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=WorkflowStepStatus.PENDING.value,
        index=True,
    )
    topo_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # The step's spec as a dict (structure_id, inputs, depends_on).
    spec: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)

    # Cached outputs (populated after the underlying SimulationJob completes)
    outputs: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    # 1:1 link to the SimulationJob that executes this step.
    simulation_job_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulation_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    workflow_run: Mapped["WorkflowRun"] = relationship(
        "WorkflowRun", back_populates="steps"
    )

    def __repr__(self) -> str:
        return (
            f"<WorkflowRunStep(step_id={self.step_id!r}, "
            f"kind={self.kind!r}, status={self.status})>"
        )

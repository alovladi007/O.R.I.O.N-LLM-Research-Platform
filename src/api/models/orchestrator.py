"""
Orchestrator models — the AGI control plane tables.

Shape matches the 008_add_orchestrator migration. Three tables:

- `orchestrator_state` — singleton-per-name configuration + rolling stats.
- `orchestrator_runs` — append-only log of each orchestrator step.
- `agent_commands` — append-only log of LLM-agent commands executed.

Restored in Phase 1 / Session 1.2. Session 7.3 (agent loop) fills in
behavior on top of these tables.
"""

import enum
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..database import Base


class OrchestratorMode(str, enum.Enum):
    """How the orchestrator decides when to act."""

    MANUAL = "MANUAL"
    SCHEDULED = "SCHEDULED"
    CONTINUOUS = "CONTINUOUS"
    PAUSED = "PAUSED"


class OrchestratorState(Base):
    """Singleton-per-name configuration + running stats for the orchestrator."""

    __tablename__ = "orchestrator_state"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )

    name: Mapped[str] = mapped_column(
        Text, nullable=False, unique=True, index=True,
        comment="Unique orchestrator name (use 'default' for the single-tenant case).",
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    mode: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default=OrchestratorMode.MANUAL.value,
        comment="Operation mode: MANUAL, SCHEDULED, CONTINUOUS, PAUSED.",
    )

    config: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Business rules: budgets, thresholds, priorities.",
    )

    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_training_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_experiment_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    run_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_simulations_launched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_experiments_launched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_trainings_launched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    stats: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSONB, nullable=True, default=dict,
    )

    runs: Mapped[List["OrchestratorRun"]] = relationship(
        "OrchestratorRun",
        back_populates="orchestrator",
        cascade="all, delete-orphan",
        order_by="OrchestratorRun.started_at.desc()",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return f"<OrchestratorState(name={self.name!r}, mode={self.mode}, run_count={self.run_count})>"


class OrchestratorRun(Base):
    """One executed orchestrator step — append-only."""

    __tablename__ = "orchestrator_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )

    orchestrator_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("orchestrator_state.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    actions: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Actions taken: campaigns advanced, simulations/experiments launched.",
    )

    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    stats_before: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    stats_after: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    triggered_by: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, index=True,
        comment="What triggered this run: schedule, manual, api, agent.",
    )
    trigger_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow,
    )

    orchestrator: Mapped[OrchestratorState] = relationship(
        "OrchestratorState", back_populates="runs", lazy="selectin",
    )

    def __repr__(self) -> str:
        return (
            f"<OrchestratorRun(id={self.id}, orchestrator_id={self.orchestrator_id}, "
            f"success={self.success})>"
        )


class AgentCommand(Base):
    """Append-only record of an LLM-agent command handled by the API."""

    __tablename__ = "agent_commands"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )

    agent_id: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, index=True,
        comment="Identifier for the agent / LLM that issued the command.",
    )
    command_type: Mapped[str] = mapped_column(
        Text, nullable=False, index=True,
        comment="Kind of command (create_campaign, advance_campaign, ...).",
    )
    payload: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, nullable=False, comment="Command payload / request data.",
    )

    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow, index=True,
    )

    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        "metadata", JSONB, nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<AgentCommand(id={self.id}, agent_id={self.agent_id!r}, "
            f"type={self.command_type}, success={self.success})>"
        )

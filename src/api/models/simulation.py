"""
Simulation job and result models.

SimulationJob: Represents a queued or running simulation job
SimulationResult: Stores the output and results of a completed job

These models track the full lifecycle of a simulation from submission
to completion, including status, parameters, and results.
"""

import uuid
from datetime import datetime
from typing import Optional
import enum

from sqlalchemy import String, DateTime, Text, Integer, Enum, ForeignKey, JSON, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class JobStatus(str, enum.Enum):
    """Simulation job status."""
    PENDING = "PENDING"       # Created but not queued yet
    QUEUED = "QUEUED"         # Queued for execution
    RUNNING = "RUNNING"       # Currently running
    COMPLETED = "COMPLETED"   # Finished successfully
    FAILED = "FAILED"         # Failed with error
    CANCELLED = "CANCELLED"   # Cancelled by user
    TIMEOUT = "TIMEOUT"       # Exceeded time limit


class JobKind(str, enum.Enum):
    """
    Semantic kind of work the job is doing, independent of the engine.
    Added in Session 1.4. The existing ``engine`` column says *which*
    physics package (qe / lammps / mock / ...); ``kind`` says *what*
    the job is attempting.
    """
    MOCK_STATIC = "mock_static"
    DFT_RELAX = "dft_relax"
    DFT_STATIC = "dft_static"
    DFT_BANDS = "dft_bands"
    DFT_DOS = "dft_dos"
    MD_NVT = "md_nvt"
    MD_NPT = "md_npt"
    MD_NVE = "md_nve"
    CONTINUUM_ELASTIC = "continuum_elastic"
    CONTINUUM_THERMAL = "continuum_thermal"
    MESOSCALE_KMC = "mesoscale_kmc"
    ML_TRAIN = "ml_train"
    ML_INFER = "ml_infer"
    BO_SUGGEST = "bo_suggest"
    AL_QUERY = "al_query"
    IMPORT = "import"
    EXPORT = "export"
    AGENT_STEP = "agent_step"


# State machine: keys are the source state, values are the set of legal
# destination states. COMPLETED / FAILED / CANCELLED / TIMEOUT are terminal
# (empty transition set).
_LEGAL_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.QUEUED, JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.RUNNING: {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMEOUT,
    },
    # Terminal states: nothing legal after.
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
    JobStatus.TIMEOUT: set(),
}


class IllegalJobTransitionError(Exception):
    """Raised when :meth:`SimulationJob.transition_to` is called with an illegal target."""

    def __init__(self, current: JobStatus, target: JobStatus):
        self.current = current
        self.target = target
        super().__init__(
            f"Illegal job transition {current.value} → {target.value}. "
            f"Legal from {current.value}: "
            f"{sorted(s.value for s in _LEGAL_TRANSITIONS.get(current, set()))}"
        )


def check_transition(current: "JobStatus", target: "JobStatus") -> "JobStatus":
    """
    Validate a state transition and return the normalized target enum.

    Pure function (no model instance needed). Raises
    :class:`IllegalJobTransitionError` on illegal moves. Lets unit tests
    exercise the state machine without constructing SQLAlchemy objects.
    """
    if isinstance(current, str):
        current = JobStatus(current)
    if isinstance(target, str):
        target = JobStatus(target)
    allowed = _LEGAL_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise IllegalJobTransitionError(current=current, target=target)
    return target


def is_terminal_status(status: "JobStatus") -> bool:
    """Whether `status` is one of the four terminal states."""
    if isinstance(status, str):
        status = JobStatus(status)
    return status in (
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMEOUT,
    )


class JobPriority(int, enum.Enum):
    """Job priority levels."""
    LOW = 0
    NORMAL = 5
    HIGH = 10
    URGENT = 20


class SimulationJob(Base):
    """
    Simulation job model.

    Represents a single simulation job (DFT, MD, etc.) submitted by a user.
    Jobs are processed by the worker service and produce results.

    Future sessions will add:
    - Job scheduling and priority queues
    - Resource usage tracking
    - Job dependencies (run job B after job A completes)
    - Automatic retry logic
    - Real-time job monitoring via WebSocket
    """
    __tablename__ = "simulation_jobs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Owner (multi-tenancy)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who owns this record"
    )

    # Foreign keys
    structure_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("structures.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    workflow_template_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("workflow_templates.id"),
        nullable=False,
        index=True
    )

    # Job metadata
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Status and priority
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus, native_enum=False),
        nullable=False,
        default=JobStatus.PENDING,
        index=True
    )
    priority: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=JobPriority.NORMAL.value,
        index=True
    )

    # Engine and parameters
    engine: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Resolved engine for this job (from template)"
    )
    # Semantic kind of work — independent of the engine. Session 1.4.
    # Nullable for backwards compatibility with pre-1.4 rows; the router
    # requires it on new submissions via the schema.
    kind: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Semantic kind: dft_relax, md_nvt, ml_train, ... (see JobKind).",
    )
    parameters: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Job parameters (merged template defaults + user overrides)"
    )

    # Resource allocation
    resources: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Allocated resources (cores, memory, GPU, walltime)"
    )

    # Timing
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Progress tracking
    progress: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Job progress (0.0 to 1.0)"
    )
    current_step: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Current step in workflow"
    )

    # Error handling
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of retry attempts"
    )

    # Worker info
    worker_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="ID of worker processing this job"
    )
    worker_hostname: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Hostname of worker machine"
    )

    # Celery task ID (for tracking async tasks)
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True
    )

    # Metadata (see notes in material.py on the attribute rename).
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=dict
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # Relationships
    structure: Mapped["Structure"] = relationship(
        "Structure",
        back_populates="simulation_jobs",
        lazy="selectin"
    )
    workflow_template: Mapped["WorkflowTemplate"] = relationship(
        "WorkflowTemplate",
        back_populates="simulation_jobs",
        lazy="selectin"
    )
    results: Mapped[Optional["SimulationResult"]] = relationship(
        "SimulationResult",
        back_populates="simulation_job",
        uselist=False,
        cascade="all, delete-orphan"
    )

    owner: Mapped["User"] = relationship("User", foreign_keys=[owner_id])

    def __repr__(self) -> str:
        return f"<SimulationJob(id={self.id}, status={self.status}, engine={self.engine})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        duration = None
        if self.started_at and self.finished_at:
            duration = (self.finished_at - self.started_at).total_seconds()

        return {
            "id": str(self.id),
            "structure_id": str(self.structure_id),
            "workflow_template_id": str(self.workflow_template_id),
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority,
            "engine": self.engine,
            "parameters": self.parameters,
            "resources": self.resources,
            "progress": self.progress,
            "current_step": self.current_step,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "duration_seconds": duration,
            "worker_id": self.worker_id,
            "metadata": self.extra_metadata,
        }

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state (completed, failed, or cancelled)."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT]

    @property
    def is_running_or_queued(self) -> bool:
        """Check if job is currently active."""
        return self.status in [JobStatus.QUEUED, JobStatus.RUNNING]

    # -----------------------------------------------------------------
    # State machine (Session 1.4)
    # -----------------------------------------------------------------
    def transition_to(
        self,
        target: "JobStatus",
        *,
        error_message: Optional[str] = None,
        set_started: bool = False,
        set_finished: bool = False,
    ) -> "JobStatus":
        """
        Attempt to move the job to ``target``. Raises
        :class:`IllegalJobTransitionError` when the move violates the
        state machine defined in :data:`_LEGAL_TRANSITIONS`.

        Side effects on success:

        - ``self.status`` = target
        - ``self.updated_at`` = now
        - optionally ``self.started_at`` / ``self.finished_at`` per flags
        - ``self.error_message`` set when provided (typically on FAILED /
          CANCELLED / TIMEOUT)

        Delegates the legality check to :func:`check_transition` so the
        logic is unit-testable without constructing SQLAlchemy instances.
        """
        normalized = check_transition(self.status, target)
        now = datetime.utcnow()
        self.status = normalized
        self.updated_at = now
        if set_started and self.started_at is None:
            self.started_at = now
        if set_finished and self.finished_at is None:
            self.finished_at = now
        if error_message is not None:
            self.error_message = error_message
        return normalized


class SimulationResult(Base):
    """
    Simulation result model.

    Stores the output and results from a completed simulation job.
    Results include summary metrics and paths to artifact files.

    Future sessions will add:
    - Automatic property extraction
    - Result comparison and analysis
    - Result caching for ML training
    - Result visualization metadata
    """
    __tablename__ = "simulation_results"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Owner (multi-tenancy)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who owns this record"
    )

    # Foreign key (one-to-one with SimulationJob)
    simulation_job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulation_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )

    # High-level summary (e.g., final energy, band gap, etc.)
    # Example: {"energy": -123.45, "band_gap": 1.2, "magnetization": 0.5}
    summary: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Key results and metrics"
    )

    # Artifact storage
    # Option 1: Path to files (MinIO, S3, local filesystem)
    artifacts_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Path to result artifacts (output files, plots, etc.)"
    )

    # Option 2: List of artifact metadata
    # Example: [{"name": "output.txt", "path": "s3://...", "size": 1024}, ...]
    artifacts: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        comment="List of result artifacts with metadata"
    )

    # Detailed results (for small results, store directly)
    # Example: trajectory data, DOS, band structure points
    detailed_results: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Detailed results data (if not too large)"
    )

    # Quality metrics
    convergence_reached: Mapped[Optional[bool]] = mapped_column(
        nullable=True,
        comment="Whether calculation converged"
    )
    quality_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Quality score (0-1, higher is better)"
    )

    # Metadata (see notes in material.py on the attribute rename).
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, default=dict
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )

    # Relationships
    simulation_job: Mapped["SimulationJob"] = relationship(
        "SimulationJob",
        back_populates="results",
        lazy="selectin"
    )

    owner: Mapped["User"] = relationship("User", foreign_keys=[owner_id])

    def __repr__(self) -> str:
        return f"<SimulationResult(id={self.id}, job_id={self.simulation_job_id})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "simulation_job_id": str(self.simulation_job_id),
            "summary": self.summary,
            "artifacts_path": self.artifacts_path,
            "artifacts": self.artifacts,
            "convergence_reached": self.convergence_reached,
            "quality_score": self.quality_score,
            "metadata": self.extra_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

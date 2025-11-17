"""
Multi-scale simulation models.

This module defines models for mesoscale and continuum simulations that bridge
atomistic and macroscopic scales. These simulations can use results from
atomistic simulations as inputs.

Models:
    - MesoscaleSimulationJob: Phase field, Monte Carlo, KMC simulations
    - MesoscaleSimulationResult: Microstructure metrics and outputs
    - ContinuumSimulationJob: FEM, FVM, BEM simulations
    - ContinuumSimulationResult: Effective properties and solution fields

Session 11: Multi-scale simulation infrastructure
"""

import uuid
from datetime import datetime
from typing import Optional
import enum

from sqlalchemy import String, DateTime, Integer, Enum, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class MesoscaleEngineType(str, enum.Enum):
    """Mesoscale simulation engine types."""
    PHASE_FIELD = "PHASE_FIELD"  # Phase field modeling for microstructure evolution
    MONTE_CARLO = "MONTE_CARLO"  # Monte Carlo for grain growth, phase transitions
    KMC = "KMC"                   # Kinetic Monte Carlo for diffusion, nucleation


class ContinuumEngineType(str, enum.Enum):
    """Continuum simulation engine types."""
    FEM = "FEM"  # Finite Element Method
    FVM = "FVM"  # Finite Volume Method
    BEM = "BEM"  # Boundary Element Method


class MultiscaleJobStatus(str, enum.Enum):
    """Multi-scale simulation job status."""
    PENDING = "PENDING"       # Created but not queued yet
    QUEUED = "QUEUED"         # Queued for execution
    RUNNING = "RUNNING"       # Currently running
    COMPLETED = "COMPLETED"   # Finished successfully
    FAILED = "FAILED"         # Failed with error
    CANCELLED = "CANCELLED"   # Cancelled by user


class MesoscaleSimulationJob(Base):
    """
    Mesoscale simulation job model.

    Represents simulations at the mesoscale (micrometers to millimeters),
    including phase field modeling, Monte Carlo simulations for grain growth,
    and kinetic Monte Carlo for diffusion processes.

    These simulations can use parameters derived from atomistic simulations
    (e.g., interface energies, diffusion coefficients) to model larger-scale
    microstructure evolution.
    """
    __tablename__ = "mesoscale_simulation_jobs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign keys
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who submitted this mesoscale simulation"
    )

    parent_atomic_job_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulation_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Optional parent atomistic simulation providing input parameters"
    )

    # Job metadata
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Descriptive name for this mesoscale simulation"
    )

    # Engine configuration
    engine_type: Mapped[MesoscaleEngineType] = mapped_column(
        Enum(MesoscaleEngineType, native_enum=False),
        nullable=False,
        index=True,
        comment="Type of mesoscale simulation engine"
    )

    # Status
    status: Mapped[MultiscaleJobStatus] = mapped_column(
        Enum(MultiscaleJobStatus, native_enum=False),
        nullable=False,
        default=MultiscaleJobStatus.PENDING,
        index=True,
        comment="Current status of the mesoscale simulation"
    )

    # Parameters
    parameters: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Simulation parameters (domain size, timesteps, material properties, etc.)"
    )

    # Timing
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Celery task ID (for tracking async tasks)
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
        comment="ID of the Celery task running this simulation"
    )

    # Relationships
    owner: Mapped["User"] = relationship(
        "User",
        foreign_keys=[owner_id],
        lazy="selectin"
    )

    parent_atomic_job: Mapped[Optional["SimulationJob"]] = relationship(
        "SimulationJob",
        foreign_keys=[parent_atomic_job_id],
        lazy="selectin"
    )

    results: Mapped[Optional["MesoscaleSimulationResult"]] = relationship(
        "MesoscaleSimulationResult",
        back_populates="job",
        uselist=False,
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<MesoscaleSimulationJob(id={self.id}, engine={self.engine_type}, status={self.status})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        duration = None
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()

        return {
            "id": str(self.id),
            "owner_id": str(self.owner_id),
            "parent_atomic_job_id": str(self.parent_atomic_job_id) if self.parent_atomic_job_id else None,
            "name": self.name,
            "engine_type": self.engine_type.value,
            "status": self.status.value,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": duration,
        }


class MesoscaleSimulationResult(Base):
    """
    Mesoscale simulation result model.

    Stores results from mesoscale simulations including microstructure metrics
    (grain size distributions, phase fractions), evolved field data, and
    output file references.
    """
    __tablename__ = "mesoscale_simulation_results"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign key (one-to-one with MesoscaleSimulationJob)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mesoscale_simulation_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )

    # Microstructure metrics
    microstructure_metrics: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Microstructure analysis: grain size, phase fractions, grain boundary density, etc."
    )

    # Domain information
    domain_size: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Simulation domain dimensions: [nx, ny, nz] or [lx, ly, lz]"
    )

    # Simulation metadata
    timesteps: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of timesteps simulated"
    )

    # Output files
    output_files: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="List of output file paths (VTK, HDF5, images, etc.)"
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )

    # Relationships
    job: Mapped["MesoscaleSimulationJob"] = relationship(
        "MesoscaleSimulationJob",
        back_populates="results",
        lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<MesoscaleSimulationResult(id={self.id}, job_id={self.job_id})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "microstructure_metrics": self.microstructure_metrics,
            "domain_size": self.domain_size,
            "timesteps": self.timesteps,
            "output_files": self.output_files,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ContinuumSimulationJob(Base):
    """
    Continuum simulation job model.

    Represents simulations at the continuum scale (millimeters to meters),
    including finite element analysis (FEM), finite volume methods (FVM),
    and boundary element methods (BEM).

    These simulations can use effective properties from mesoscale simulations
    (e.g., homogenized elastic moduli, thermal conductivity) to model
    macroscopic behavior.
    """
    __tablename__ = "continuum_simulation_jobs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign keys
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who submitted this continuum simulation"
    )

    parent_mesoscale_job_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mesoscale_simulation_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Optional parent mesoscale simulation providing material properties"
    )

    # Job metadata
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Descriptive name for this continuum simulation"
    )

    # Engine configuration
    engine_type: Mapped[ContinuumEngineType] = mapped_column(
        Enum(ContinuumEngineType, native_enum=False),
        nullable=False,
        index=True,
        comment="Type of continuum simulation engine"
    )

    # Status
    status: Mapped[MultiscaleJobStatus] = mapped_column(
        Enum(MultiscaleJobStatus, native_enum=False),
        nullable=False,
        default=MultiscaleJobStatus.PENDING,
        index=True,
        comment="Current status of the continuum simulation"
    )

    # Parameters
    parameters: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Simulation parameters (mesh, boundary conditions, material properties, etc.)"
    )

    # Timing
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Celery task ID (for tracking async tasks)
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
        comment="ID of the Celery task running this simulation"
    )

    # Relationships
    owner: Mapped["User"] = relationship(
        "User",
        foreign_keys=[owner_id],
        lazy="selectin"
    )

    parent_mesoscale_job: Mapped[Optional["MesoscaleSimulationJob"]] = relationship(
        "MesoscaleSimulationJob",
        foreign_keys=[parent_mesoscale_job_id],
        lazy="selectin"
    )

    results: Mapped[Optional["ContinuumSimulationResult"]] = relationship(
        "ContinuumSimulationResult",
        back_populates="job",
        uselist=False,
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ContinuumSimulationJob(id={self.id}, engine={self.engine_type}, status={self.status})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        duration = None
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()

        return {
            "id": str(self.id),
            "owner_id": str(self.owner_id),
            "parent_mesoscale_job_id": str(self.parent_mesoscale_job_id) if self.parent_mesoscale_job_id else None,
            "name": self.name,
            "engine_type": self.engine_type.value,
            "status": self.status.value,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": duration,
        }


class ContinuumSimulationResult(Base):
    """
    Continuum simulation result model.

    Stores results from continuum simulations including effective material
    properties, mesh information, solution fields (displacement, stress,
    temperature), and output file references.
    """
    __tablename__ = "continuum_simulation_results"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign key (one-to-one with ContinuumSimulationJob)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("continuum_simulation_jobs.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True
    )

    # Effective properties
    effective_properties: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Homogenized material properties: elastic moduli, thermal conductivity, etc."
    )

    # Mesh information
    mesh_info: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Mesh metadata: number of nodes, elements, element types, etc."
    )

    # Solution fields
    solution_fields: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Summary of solution fields: displacement, stress, temperature, etc."
    )

    # Output files
    output_files: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        comment="List of output file paths (VTK, ParaView, result databases, etc.)"
    )

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )

    # Relationships
    job: Mapped["ContinuumSimulationJob"] = relationship(
        "ContinuumSimulationJob",
        back_populates="results",
        lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<ContinuumSimulationResult(id={self.id}, job_id={self.job_id})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "job_id": str(self.job_id),
            "effective_properties": self.effective_properties,
            "mesh_info": self.mesh_info,
            "solution_fields": self.solution_fields,
            "output_files": self.output_files,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

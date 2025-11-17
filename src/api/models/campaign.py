"""
Design Campaign models for AI-driven materials discovery loops.

DesignCampaign: Represents an autonomous design loop optimizing materials
DesignIteration: Tracks each iteration of the design loop (candidates, scores, etc.)

These models enable AI agents to:
1. Define target properties and constraints
2. Generate and evaluate candidate structures
3. Iteratively improve designs through ML-guided search
4. Track progress and best discoveries
"""

import uuid
from datetime import datetime
from typing import Optional, List
import enum

from sqlalchemy import String, DateTime, Text, Integer, Enum, ForeignKey, JSON, Float, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class CampaignStatus(str, enum.Enum):
    """Design campaign status."""
    CREATED = "CREATED"       # Campaign created but not started
    RUNNING = "RUNNING"       # Currently running iterations
    PAUSED = "PAUSED"         # Paused by user
    COMPLETED = "COMPLETED"   # Reached max iterations or convergence
    FAILED = "FAILED"         # Failed with error
    CANCELLED = "CANCELLED"   # Cancelled by user


class DesignCampaign(Base):
    """
    Design campaign model for AI-driven materials discovery.

    Represents an autonomous design loop that:
    1. Generates candidate structures based on current knowledge
    2. Evaluates candidates using ML predictions
    3. Selects best candidates for further exploration
    4. Iterates until convergence or max iterations

    AI Agent Integration:
    ---------------------
    AI agents can interact with campaigns through:

    1. CREATE: Define optimization goals
       - Target properties (e.g., bandgap=2.0 eV, formation_energy < -3.0)
       - Constraints (e.g., elements=['Mo', 'S'], max_atoms=20)
       - Strategy (random, bayesian, genetic, reinforcement_learning)

    2. RUN: Execute design iterations
       - Agent monitors progress through iteration metrics
       - Agent can pause/resume based on discoveries
       - Agent can adjust strategy mid-campaign

    3. ANALYZE: Examine results and learn
       - Review best structures found
       - Analyze what strategies worked
       - Use insights to inform future campaigns

    Future Enhancements:
    --------------------
    - Multi-objective optimization (Pareto fronts)
    - Transfer learning from previous campaigns
    - Active learning strategies
    - Reinforcement learning integration
    - Generative models for structure creation
    - Uncertainty-guided exploration
    """
    __tablename__ = "design_campaigns"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign key to owner
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who created this campaign"
    )

    # Basic information
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Campaign name"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Campaign description and goals"
    )

    # Status
    status: Mapped[CampaignStatus] = mapped_column(
        Enum(CampaignStatus, native_enum=False),
        nullable=False,
        default=CampaignStatus.CREATED,
        index=True,
        comment="Current campaign status"
    )

    # Configuration (JSON)
    # Example:
    # {
    #   "target_properties": {
    #     "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0},
    #     "formation_energy": {"max": -3.0, "weight": 0.5}
    #   },
    #   "constraints": {
    #     "elements": ["Mo", "S", "Se"],
    #     "max_atoms": 20,
    #     "dimensionality": 2
    #   },
    #   "strategy": "bayesian_optimization",
    #   "acquisition_function": "expected_improvement"
    # }
    config: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Campaign configuration (targets, constraints, strategy)"
    )

    # Iteration tracking
    max_iterations: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
        comment="Maximum number of design iterations"
    )
    current_iteration: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Current iteration number (0-based)"
    )

    # Best results tracking
    best_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        index=True,
        comment="Best score achieved so far (0-1, higher is better)"
    )
    best_structure_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("structures.id", ondelete="SET NULL"),
        nullable=True,
        comment="ID of best structure found"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When first iteration started"
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When campaign completed or was stopped"
    )

    # Metadata for extensibility
    metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=dict,
        comment="Additional metadata (tags, notes, etc.)"
    )

    # Relationships
    owner: Mapped["User"] = relationship(
        "User",
        back_populates="design_campaigns",
        lazy="selectin"
    )
    iterations: Mapped[List["DesignIteration"]] = relationship(
        "DesignIteration",
        back_populates="campaign",
        cascade="all, delete-orphan",
        order_by="DesignIteration.iteration_index"
    )
    best_structure: Mapped[Optional["Structure"]] = relationship(
        "Structure",
        foreign_keys=[best_structure_id],
        lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<DesignCampaign(id={self.id}, name={self.name}, status={self.status}, iter={self.current_iteration}/{self.max_iterations})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        duration = None
        if self.started_at and self.completed_at:
            duration = (self.completed_at - self.started_at).total_seconds()

        return {
            "id": str(self.id),
            "owner_id": str(self.owner_id),
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "config": self.config,
            "max_iterations": self.max_iterations,
            "current_iteration": self.current_iteration,
            "best_score": self.best_score,
            "best_structure_id": str(self.best_structure_id) if self.best_structure_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": duration,
            "metadata": self.metadata,
        }

    @property
    def is_active(self) -> bool:
        """Check if campaign is actively running."""
        return self.status == CampaignStatus.RUNNING

    @property
    def is_terminal(self) -> bool:
        """Check if campaign is in a terminal state."""
        return self.status in [
            CampaignStatus.COMPLETED,
            CampaignStatus.FAILED,
            CampaignStatus.CANCELLED
        ]

    @property
    def progress_fraction(self) -> float:
        """Get campaign progress as fraction (0-1)."""
        if self.max_iterations <= 0:
            return 0.0
        return min(1.0, self.current_iteration / self.max_iterations)


class DesignIteration(Base):
    """
    Design iteration model - tracks one iteration of a design campaign.

    Each iteration represents one cycle of:
    1. Generate candidate structures
    2. Evaluate candidates (ML predictions)
    3. Score against targets
    4. Select best candidates

    AI Agent Learning:
    ------------------
    Agents can analyze iterations to learn:
    - Which generation strategies work best
    - How scores improve over time
    - Diversity of explored space
    - Exploitation vs exploration balance

    The metrics field stores rich information for learning:
    {
        "scores": [0.8, 0.75, 0.9, ...],  # All candidate scores
        "mean_score": 0.82,
        "max_score": 0.9,
        "min_score": 0.75,
        "std_score": 0.06,
        "improvement_from_previous": 0.05,
        "diversity_metric": 0.45,  # How different are candidates
        "coverage": 0.33,  # How much of design space explored
        "novelty_scores": [0.2, 0.8, 0.3, ...]  # How novel each candidate is
    }

    Future Analytics:
    -----------------
    - Correlation analysis (which features predict success)
    - Learning curves (score vs iteration)
    - Exploration heatmaps
    - Feature importance ranking
    """
    __tablename__ = "design_iterations"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign key to campaign
    campaign_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("design_campaigns.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Iteration index (0-based)
    iteration_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
        comment="Iteration number within campaign (0-based)"
    )

    # Structure tracking
    # Note: Using ARRAY of UUID requires PostgreSQL
    # These are stored as strings for JSON serialization
    created_structures: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        default=list,
        comment="List of structure IDs created this iteration"
    )
    evaluated_structures: Mapped[Optional[list]] = mapped_column(
        JSON,
        nullable=True,
        default=list,
        comment="List of structure IDs evaluated (may include previous structures)"
    )

    # Best result this iteration
    best_score_this_iter: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Best score achieved in this iteration"
    )
    best_structure_id_this_iter: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("structures.id", ondelete="SET NULL"),
        nullable=True,
        comment="Best structure found in this iteration"
    )

    # Metrics (JSON)
    # Rich data for analysis and learning
    metrics: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=dict,
        comment="Iteration metrics (scores, diversity, improvements, etc.)"
    )

    # Strategy used for this iteration
    strategy_used: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Strategy used: random, bayesian, genetic, rl, etc."
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When iteration completed"
    )

    # Metadata
    metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=dict,
        comment="Additional iteration metadata"
    )

    # Relationships
    campaign: Mapped["DesignCampaign"] = relationship(
        "DesignCampaign",
        back_populates="iterations",
        lazy="selectin"
    )
    best_structure: Mapped[Optional["Structure"]] = relationship(
        "Structure",
        foreign_keys=[best_structure_id_this_iter],
        lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<DesignIteration(campaign_id={self.campaign_id}, index={self.iteration_index}, best_score={self.best_score_this_iter})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        duration = None
        if self.created_at and self.completed_at:
            duration = (self.completed_at - self.created_at).total_seconds()

        return {
            "id": str(self.id),
            "campaign_id": str(self.campaign_id),
            "iteration_index": self.iteration_index,
            "created_structures": self.created_structures or [],
            "evaluated_structures": self.evaluated_structures or [],
            "best_score_this_iter": self.best_score_this_iter,
            "best_structure_id_this_iter": str(self.best_structure_id_this_iter) if self.best_structure_id_this_iter else None,
            "metrics": self.metrics or {},
            "strategy_used": self.strategy_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": duration,
            "metadata": self.metadata or {},
        }

    @property
    def num_candidates_created(self) -> int:
        """Get number of structures created in this iteration."""
        return len(self.created_structures) if self.created_structures else 0

    @property
    def num_candidates_evaluated(self) -> int:
        """Get number of structures evaluated in this iteration."""
        return len(self.evaluated_structures) if self.evaluated_structures else 0

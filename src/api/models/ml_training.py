"""
ML Training Job Models
=======================

Models for tracking ML model training jobs in the database.
These jobs are executed by Celery workers and train GNN models
for property prediction.
"""

import uuid
from datetime import datetime
from typing import Optional
from enum import Enum

from sqlalchemy import String, DateTime, Text, JSON, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class TrainingStatus(str, Enum):
    """Status of ML training job."""
    PENDING = "PENDING"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class MLTrainingJob(Base):
    """
    ML Model Training Job.

    Tracks the training of machine learning models for property prediction.
    Each job trains a specific model architecture on a dataset targeting
    a specific property.
    """
    __tablename__ = "ml_training_jobs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Owner
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Job metadata
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable job name"
    )

    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Job description"
    )

    # Training configuration
    model_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Model architecture (CGCNN, ALIGNN, RandomForest, etc.)"
    )

    target_property: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Target property to predict (bandgap, formation_energy, etc.)"
    )

    training_config: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Training hyperparameters and configuration"
    )

    # Dataset info
    dataset_config: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Dataset configuration (filters, split, etc.)"
    )

    # Status
    status: Mapped[TrainingStatus] = mapped_column(
        SQLEnum(TrainingStatus),
        nullable=False,
        default=TrainingStatus.PENDING,
        index=True
    )

    # Progress tracking
    current_epoch: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Current training epoch"
    )

    total_epochs: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        comment="Total training epochs"
    )

    progress: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Progress percentage (0.0 - 1.0)"
    )

    # Celery task tracking
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True
    )

    # Results
    final_metrics: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Final training and validation metrics"
    )

    best_val_loss: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        comment="Best validation loss achieved"
    )

    model_registry_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ml_model_registry.id"),
        nullable=True,
        comment="ID of registered model (if training succeeded)"
    )

    # Error tracking
    error_message: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True
    )

    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True
    )

    # Relationships
    owner: Mapped["User"] = relationship("User", back_populates="ml_training_jobs")
    registered_model: Mapped[Optional["MLModelRegistry"]] = relationship(
        "MLModelRegistry",
        foreign_keys=[model_registry_id]
    )

    def __repr__(self):
        return f"<MLTrainingJob(id={self.id}, name={self.name}, status={self.status.value})>"

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
            "model_type": self.model_type,
            "target_property": self.target_property,
            "training_config": self.training_config,
            "dataset_config": self.dataset_config,
            "status": self.status.value,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "progress": self.progress,
            "celery_task_id": self.celery_task_id,
            "final_metrics": self.final_metrics,
            "best_val_loss": self.best_val_loss,
            "model_registry_id": str(self.model_registry_id) if self.model_registry_id else None,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": duration,
        }

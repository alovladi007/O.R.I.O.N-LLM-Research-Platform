"""
ML Model Registry for tracking trained models.

Session 16: Model Training Infrastructure
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Text, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class MLModelRegistry(Base):
    """
    Registry of trained ML models.

    Tracks all ML models (GNNs, traditional ML, etc.) that have been trained
    or deployed in NANO-OS, including their metadata, checkpoints, and metrics.

    Session 16: Model Training
    """
    __tablename__ = "ml_model_registry"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Model identification
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique model name (e.g., 'cgcnn_bandgap_v2')"
    )

    version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Model version (e.g., '1.0.0', '2.1.3')"
    )

    # Target property
    target: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Target property (bandgap, formation_energy, etc.)"
    )

    # Model metadata
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable description"
    )

    model_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="CGCNN",
        comment="Model architecture type (CGCNN, ALIGNN, RandomForest, etc.)"
    )

    # Checkpoint and artifacts
    checkpoint_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Path to model checkpoint file (.pth, .pkl, etc.)"
    )

    artifacts_path: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Path to additional artifacts (scalers, preprocessors, etc.)"
    )

    # Training configuration
    training_config: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Hyperparameters and training config"
    )

    # Performance metrics
    metrics: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Training and validation metrics (MSE, MAE, R², etc.)"
    )

    # Dataset info
    dataset_info: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Information about training dataset"
    )

    # Status flags
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        comment="Whether this model is active for inference"
    )

    is_system_provided: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if provided by system, False if user-trained"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        index=True
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    def __repr__(self):
        return f"<MLModelRegistry(name={self.name}, version={self.version}, target={self.target})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "target": self.target,
            "description": self.description,
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "dataset_info": self.dataset_info,
            "is_active": self.is_active,
            "is_system_provided": self.is_system_provided,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

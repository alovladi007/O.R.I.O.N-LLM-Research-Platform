"""
ML Interatomic Potential model for tracking machine learning force fields.

Session 18: ML Interatomic Potentials & Hybrid DFT-ML Workflows
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, DateTime, Text, JSON, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class MLPotential(Base):
    """
    Registry of ML interatomic potentials.

    Tracks ML-based force fields (e.g., SNAP, NequIP, MACE) trained from
    DFT data for use in molecular dynamics simulations.

    Session 18: Hybrid DFT-ML Workflows
    """
    __tablename__ = "ml_potentials"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Potential identification
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        comment="Unique potential name (e.g., 'snap_silicon_v1', 'nequip_water_v2')"
    )

    version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Potential version (e.g., '1.0.0', '2.1.3')"
    )

    # Potential metadata
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Human-readable description"
    )

    descriptor_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Descriptor/architecture type (SNAP, SOAP, NequIP, MACE, etc.)"
    )

    # Training data
    training_data_source: Mapped[List[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        comment="List of simulation job IDs used for training data"
    )

    training_dataset_info: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Information about training dataset (num_structures, energy_range, etc.)"
    )

    # Potential files
    path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Path to potential files directory"
    )

    files: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Dictionary of potential files (coefficients, network weights, etc.)"
    )

    # Training configuration
    training_config: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Training hyperparameters and configuration"
    )

    # Performance metrics
    metrics: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Validation metrics (energy RMSE, force MAE, etc.)"
    )

    # Applicability
    elements: Mapped[List[str]] = mapped_column(
        ARRAY(String),
        nullable=False,
        comment="Chemical elements covered by this potential"
    )

    temperature_range: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Valid temperature range (min, max) in K"
    )

    pressure_range: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Valid pressure range (min, max) in GPa"
    )

    # Status flags
    is_active: Mapped[bool] = mapped_column(
        String,
        nullable=False,
        default=True,
        comment="Whether this potential is active for use"
    )

    is_validated: Mapped[bool] = mapped_column(
        String,
        nullable=False,
        default=False,
        comment="Whether potential has been validated on test set"
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
        return f"<MLPotential(name={self.name}, version={self.version}, descriptor={self.descriptor_type})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "descriptor_type": self.descriptor_type,
            "training_data_source": self.training_data_source,
            "training_dataset_info": self.training_dataset_info,
            "path": self.path,
            "files": self.files,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "elements": self.elements,
            "temperature_range": self.temperature_range,
            "pressure_range": self.pressure_range,
            "is_active": self.is_active,
            "is_validated": self.is_validated,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

"""
Predicted properties model for ML-based material property predictions.

This model stores the results of ML property predictions, including:
- Predicted properties (bandgap, formation energy, stability, etc.)
- Confidence scores for each prediction
- Model name and version for reproducibility
- Metadata for additional information

The model supports:
- Multiple predictions per structure (different models, versions)
- Caching of predictions to avoid redundant computation
- Tracking of model provenance
- Comparison with simulation results (via comparison utilities)
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, ForeignKey, JSON, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class PredictedProperties(Base):
    """
    ML-predicted material properties model.

    Stores predictions from machine learning models for material properties.
    Each prediction is associated with a specific structure and model version.

    Multiple predictions can exist for the same structure:
    - Different ML models (STUB, CGCNN, ALIGNN, etc.)
    - Different model versions
    - Different prediction timestamps

    This allows:
    - Model comparison and benchmarking
    - Tracking prediction improvements over time
    - Ensemble predictions from multiple models
    - A/B testing of model versions

    Future enhancements:
    - Uncertainty quantification
    - Active learning integration
    - Automatic recomputation when models are updated
    - Prediction explanation/interpretability metadata
    """
    __tablename__ = "predicted_properties"

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

    # Foreign key to structure
    structure_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("structures.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Structure for which properties were predicted"
    )

    # Model information
    model_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
        comment="Name of ML model (STUB, CGCNN, MEGNET, M3GNET, ALIGNN, etc.)"
    )
    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="Version of the ML model"
    )

    # Predicted properties
    # Example: {
    #   "bandgap": 2.5,
    #   "formation_energy": -3.2,
    #   "stability_score": 0.85,
    #   "elastic_moduli": {"bulk": 150, "shear": 80},
    #   ...
    # }
    properties: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Predicted material properties"
    )

    # Confidence scores for each property
    # Example: {
    #   "bandgap": 0.95,
    #   "formation_energy": 0.92,
    #   "stability_score": 0.88,
    #   ...
    # }
    confidence_scores: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Confidence/uncertainty for each predicted property"
    )

    # Additional metadata
    # Example: {
    #   "inference_time_ms": 123.45,
    #   "gpu_used": true,
    #   "ensemble_size": 5,
    #   "input_features": ["composition", "structure", "volume"],
    #   ...
    # }
    metadata: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        default=dict,
        comment="Additional metadata about the prediction"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
        comment="When the prediction was made"
    )

    # Relationships
    structure: Mapped["Structure"] = relationship(
        "Structure",
        backref="predicted_properties",
        lazy="selectin"
    )

    owner: Mapped["User"] = relationship("User", foreign_keys=[owner_id])

    # Composite index for efficient queries
    __table_args__ = (
        # Find latest prediction for a structure + model combination
        Index(
            "ix_predicted_properties_structure_model",
            "structure_id",
            "model_name",
            "created_at"
        ),
        # Find all predictions by model
        Index(
            "ix_predicted_properties_model",
            "model_name",
            "model_version",
            "created_at"
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<PredictedProperties(id={self.id}, "
            f"structure_id={self.structure_id}, "
            f"model={self.model_name}:{self.model_version})>"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "structure_id": str(self.structure_id),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "properties": self.properties,
            "confidence_scores": self.confidence_scores,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def get_property(self, property_name: str) -> Optional[float]:
        """
        Get a specific predicted property value.

        Args:
            property_name: Name of the property (e.g., "bandgap")

        Returns:
            Property value or None if not found
        """
        return self.properties.get(property_name)

    def get_confidence(self, property_name: str) -> Optional[float]:
        """
        Get confidence score for a specific property.

        Args:
            property_name: Name of the property (e.g., "bandgap")

        Returns:
            Confidence score (0-1) or None if not found
        """
        return self.confidence_scores.get(property_name)

    @property
    def average_confidence(self) -> float:
        """
        Calculate average confidence across all predicted properties.

        Returns:
            Average confidence score (0-1)
        """
        if not self.confidence_scores:
            return 0.0

        scores = [s for s in self.confidence_scores.values() if isinstance(s, (int, float))]
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    @property
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """
        Check if predictions have high confidence.

        Args:
            threshold: Minimum average confidence (default: 0.8)

        Returns:
            True if average confidence exceeds threshold
        """
        return self.average_confidence >= threshold

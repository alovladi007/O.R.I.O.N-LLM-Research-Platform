"""
StructureFeatures model for ML feature caching.

Stores pre-computed graph and scalar features for structures,
avoiding recomputation for GNN inference and training.

Session 14: ML Infrastructure
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import String, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class StructureFeatures(Base):
    """
    Cached ML features for a structure.

    Stores both graph representation (for GNNs) and scalar features
    to avoid recomputing them for every inference or training run.

    Session 14: ML Infrastructure
    """
    __tablename__ = "structure_features"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Foreign key to structure (unique constraint ensures 1-to-1)
    structure_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("structures.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # One features entry per structure
        index=True
    )

    # Graph representation (for GNN models like CGCNN)
    # JSON format:
    # {
    #   "atom_features": [...],
    #   "neighbor_lists": {...},
    #   "bond_distances": {...},
    #   "num_atoms": N,
    #   "num_edges": E,
    #   "lattice_vectors": [[...], [...], [...]],
    #   "volume": V,
    #   "formula": "...",
    #   "dimensionality": D
    # }
    graph_repr: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Graph representation for GNN input"
    )

    # Scalar features (for traditional ML or feature analysis)
    # JSON format:
    # {
    #   "elements": [...],
    #   "element_fractions": {...},
    #   "num_elements": N,
    #   "avg_atomic_mass": M,
    #   "avg_electronegativity": E,
    #   "avg_atomic_radius": R,
    #   "volume_per_atom": V,
    #   "density": D,
    #   "space_filling": S
    # }
    scalar_features: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        comment="Scalar features for traditional ML"
    )

    # Feature extraction parameters (for reproducibility)
    extraction_params: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Parameters used during extraction (cutoff_radius, etc.)"
    )

    # Version tracking (in case feature extraction changes)
    feature_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="1.0.0",
        comment="Version of feature extraction code"
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )

    # Relationships
    structure: Mapped["Structure"] = relationship(
        "Structure",
        back_populates="features",
        lazy="joined"
    )

    def __repr__(self):
        return f"<StructureFeatures(structure_id={self.structure_id}, version={self.feature_version})>"

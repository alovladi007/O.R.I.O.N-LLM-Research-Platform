"""
Material model for storing material metadata.

A Material represents a chemical compound or element that can have
multiple atomic structures (crystal structures, molecular geometries, etc.).

Examples:
- MoS2 (material) can have multiple structures: 1H, 1T, 2H polymorphs
- Water (H2O) can have ice, liquid, gas structures
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, DateTime, Text, ARRAY, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class Material(Base):
    """
    Material model representing a chemical compound.

    A material can have multiple associated structures representing
    different configurations, polymorphs, or geometries.

    Future sessions will add:
    - Material properties (band gap, formation energy, etc.)
    - Experimental data integration
    - Literature references and provenance
    - Material families and classifications
    """
    __tablename__ = "materials"

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

    # Basic information
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    formula: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Classification and tags
    tags: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String(50)),
        nullable=True,
        default=list
    )

    # Chemical composition (e.g., {"Mo": 1, "S": 2} for MoS2)
    composition: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Metadata (flexible JSON for future properties)
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, default=dict)

    # Source information
    source: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Where this material came from (e.g., 'Materials Project', 'User Upload', 'Generated')"
    )
    external_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="ID in external database (e.g., mp-id from Materials Project)"
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

    # Soft delete
    deleted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True
    )

    # Relationships
    structures: Mapped[List["Structure"]] = relationship(
        "Structure",
        back_populates="material",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    owner: Mapped["User"] = relationship("User", foreign_keys=[owner_id])

    # Future: created_by relationship to User
    # created_by_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    # created_by: Mapped[Optional["User"]] = relationship(back_populates="materials")

    def __repr__(self) -> str:
        return f"<Material(id={self.id}, formula={self.formula}, name={self.name})>"

    @property
    def structure_count(self) -> int:
        """Number of associated structures."""
        return len(self.structures) if self.structures else 0

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "formula": self.formula,
            "description": self.description,
            "tags": self.tags or [],
            "composition": self.composition,
            "metadata": self.metadata,
            "source": self.source,
            "external_id": self.external_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "structure_count": self.structure_count,
        }

"""
Structure model for atomic structures.

Represents the 3D atomic arrangement of a material, including:
- Lattice vectors (unit cell)
- Atomic positions and species
- Original file format and content

Supports multiple formats: CIF, POSCAR, XYZ, and internal JSON representation.
"""

import uuid
from datetime import datetime
from typing import Optional, List
import enum

from sqlalchemy import String, DateTime, Text, Integer, Enum, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID

from ..database import Base


class StructureFormat(str, enum.Enum):
    """Supported structure file formats."""
    CIF = "CIF"           # Crystallographic Information File
    POSCAR = "POSCAR"     # VASP POSCAR/CONTCAR
    XYZ = "XYZ"           # XYZ coordinate file
    INTERNAL = "INTERNAL" # Internal JSON representation


class StructureSource(str, enum.Enum):
    """Source of the structure."""
    USER_UPLOADED = "USER_UPLOADED"   # Uploaded by user
    GENERATED = "GENERATED"            # Generated (e.g., from prototype)
    EXTERNAL_DB = "EXTERNAL_DB"        # From external database
    OPTIMIZED = "OPTIMIZED"            # Result of geometry optimization
    MD_SNAPSHOT = "MD_SNAPSHOT"        # Snapshot from MD trajectory


class Structure(Base):
    """
    Atomic structure model.

    Stores both the original file representation and a normalized
    internal representation for consistent processing.

    Future sessions will add:
    - Structure similarity search
    - Automatic property calculation (volume, density, etc.)
    - Structure visualization metadata
    - Trajectory support for MD simulations
    """
    __tablename__ = "structures"

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

    # Foreign key to material
    material_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("materials.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Metadata
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Format and source
    format: Mapped[StructureFormat] = mapped_column(
        Enum(StructureFormat, native_enum=False),
        nullable=False,
        default=StructureFormat.INTERNAL
    )
    source: Mapped[StructureSource] = mapped_column(
        Enum(StructureSource, native_enum=False),
        nullable=False,
        default=StructureSource.USER_UPLOADED
    )

    # Original file content (as uploaded)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Normalized representation (Session 3 will populate this)
    # Lattice: 3x3 matrix of lattice vectors (in Angstroms)
    # [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]]
    lattice: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="Lattice vectors and parameters"
    )

    # Alternative representation for lattice vectors (used by feature extractor)
    lattice_vectors: Mapped[Optional[List[List[float]]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Lattice vectors as 3x3 matrix"
    )

    # Atomic data: [{"species": "Mo", "position": [x, y, z], "charge": 0, ...}, ...]
    # Positions in fractional coordinates (0-1)
    atoms: Mapped[Optional[List[dict]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Atomic species and positions"
    )

    # Dimensionality: 0=molecule, 1=1D, 2=2D, 3=bulk
    dimensionality: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="0=molecule, 1=1D, 2=2D, 3=bulk"
    )

    # Computed properties (will be populated by parsers)
    num_atoms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    formula: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)

    # Lattice parameters (for easy filtering/searching)
    a: Mapped[Optional[float]] = mapped_column(nullable=True, comment="Lattice parameter a (Å)")
    b: Mapped[Optional[float]] = mapped_column(nullable=True, comment="Lattice parameter b (Å)")
    c: Mapped[Optional[float]] = mapped_column(nullable=True, comment="Lattice parameter c (Å)")
    alpha: Mapped[Optional[float]] = mapped_column(nullable=True, comment="Angle alpha (degrees)")
    beta: Mapped[Optional[float]] = mapped_column(nullable=True, comment="Angle beta (degrees)")
    gamma: Mapped[Optional[float]] = mapped_column(nullable=True, comment="Angle gamma (degrees)")
    volume: Mapped[Optional[float]] = mapped_column(nullable=True, comment="Unit cell volume (Å³)")

    # Additional metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, default=dict)

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

    # Relationships
    material: Mapped["Material"] = relationship(
        "Material",
        back_populates="structures",
        lazy="selectin"
    )

    simulation_jobs: Mapped[List["SimulationJob"]] = relationship(
        "SimulationJob",
        back_populates="structure",
        cascade="all, delete-orphan"
    )

    # Session 14: ML features
    features: Mapped[Optional["StructureFeatures"]] = relationship(
        "StructureFeatures",
        back_populates="structure",
        uselist=False,  # One-to-one relationship
        cascade="all, delete-orphan"
    )

    owner: Mapped["User"] = relationship("User", foreign_keys=[owner_id])

    def __repr__(self) -> str:
        return f"<Structure(id={self.id}, formula={self.formula}, format={self.format})>"

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "material_id": str(self.material_id),
            "name": self.name,
            "description": self.description,
            "format": self.format.value,
            "source": self.source.value,
            "formula": self.formula,
            "num_atoms": self.num_atoms,
            "dimensionality": self.dimensionality,
            "lattice_parameters": {
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "volume": self.volume,
            },
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

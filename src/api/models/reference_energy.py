"""Reference elemental energies for formation-energy calculations.

Session 3.4 introduces this table. It's the piece that turns ORION's
absolute DFT energies into comparable formation energies.

Why this exists
---------------

The total energy of a crystal from DFT is meaningful only *relative to
a reference state*. For formation energies, the reference is the
ground-state elemental crystal at the same functional + pseudo family.
Without consistent references, cross-comparing materials is
meaningless — even a 0.1 eV/atom shift in the Si reference propagates
into every SiO2 or SiC formation energy.

This table stores one row per (element, functional, pseudo_family),
populated by the Session 3.4 calibration workflow. The formation-energy
calculator reads from it and raises if a reference is missing rather
than silently substituting zero.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class ReferenceEnergy(Base):
    """One reference energy per (element, functional, pseudo_family)."""

    __tablename__ = "reference_energies"
    __table_args__ = (
        UniqueConstraint(
            "element", "functional", "pseudo_family",
            name="uq_reference_energy_triple",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )

    # The key: (element, functional, pseudo_family).
    element: Mapped[str] = mapped_column(
        String(8), nullable=False, index=True,
        comment="Periodic-table symbol — Si, Cu, Na, etc.",
    )
    functional: Mapped[str] = mapped_column(
        String(32), nullable=False, index=True,
        comment="XC functional: PBE, SCAN, HSE06, ...",
    )
    pseudo_family: Mapped[str] = mapped_column(
        String(64), nullable=False, index=True,
        comment=(
            "Pseudopotential library: SSSP_efficiency_1.3.0, "
            "SSSP_precision_1.3.0, Dojo_v0.4, etc. Specific version "
            "matters — cutoffs and valence configurations change."
        ),
    )

    # The value.
    energy_per_atom_ev: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="Total energy / n_atoms of the elemental ground state.",
    )
    n_atoms_in_reference_cell: Mapped[int] = mapped_column(
        Integer, nullable=False,
        comment="How many atoms in the cell used to compute the reference.",
    )

    # Provenance: which job produced this value.
    source_job_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("simulation_jobs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="SimulationJob that produced this reference (null for hand-seeded).",
    )

    # Structure used for the reference calculation (prototype crystal
    # structure — e.g. diamond-cubic for Si, FCC for Al, BCC for Na).
    reference_prototype: Mapped[str] = mapped_column(
        String(32), nullable=False, default="auto",
        comment=(
            "Prototype label: 'diamond_cubic', 'fcc', 'bcc', 'hcp', 'rocksalt'. "
            "'auto' means the caller didn't specify."
        ),
    )

    # Optional: lattice constant of the relaxed reference cell.
    relaxed_a_ang: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Relaxed conventional lattice constant (Å), when applicable.",
    )

    # Free-form metadata for post-hoc analysis: SCF params, k-mesh, cutoffs.
    extra_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True, default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False,
        default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<ReferenceEnergy({self.element}/{self.functional}/"
            f"{self.pseudo_family}: {self.energy_per_atom_ev:.3f} eV/atom)>"
        )

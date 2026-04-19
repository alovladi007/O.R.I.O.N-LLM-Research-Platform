"""Output schemas for job kinds.

Session 2.2 only defines ``MockStaticOutput``; the rest of the
``JobKind`` enum fills in during Phase 3 (DFT), Phase 4 (MD), etc.

Design notes
------------

- Pure pydantic v2. No SQLAlchemy imports, so the worker can validate a
  run's outputs without importing the API package.
- ``schema_version`` is stored on every output instance so that a future
  schema bump can co-exist with older rows in the DB.
- Units are fixed and documented next to each field. The roadmap's
  Session 2.2 spec fixes energies in eV and forces in eV/Å, matching
  the wider ORION convention.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


MOCK_STATIC_OUTPUT_SCHEMA_VERSION = "mock_static.v1"


class MockForceEntry(BaseModel):
    """One atom's force vector in eV/Å, ordered like the input structure."""

    model_config = ConfigDict(extra="forbid")

    species: str = Field(..., description="Atomic symbol (e.g. 'Si', 'Cl').")
    fx: float = Field(..., description="Force x-component (eV/Å).")
    fy: float = Field(..., description="Force y-component (eV/Å).")
    fz: float = Field(..., description="Force z-component (eV/Å).")


class MockTrajectoryFrame(BaseModel):
    """One frame in the synthetic trajectory."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, description="Zero-based frame index.")
    positions: List[List[float]] = Field(
        ...,
        description="Cartesian coordinates per atom (Å). Shape: n_atoms × 3.",
    )


class MockStaticOutput(BaseModel):
    """Output of the mock static calculation.

    Matches the roadmap Session 2.2 spec:

    - ``energy_ev`` = -sum(Z_i) * 1.5 + 0.01 * deterministic_noise
    - ``forces`` = per-atom zero vector + Gaussian noise at σ ≈ 0.05 eV/Å
    - ``trajectory`` = 10 frames with small displacements
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default=MOCK_STATIC_OUTPUT_SCHEMA_VERSION,
        description="Fixed schema version tag. Bump when the shape changes.",
    )
    engine: str = Field(default="mock", description="Engine name.")
    n_atoms: int = Field(..., ge=1, description="Atom count used in the run.")
    formula: str = Field(..., description="Chemical formula of the input structure.")

    energy_ev: float = Field(..., description="Total energy in eV.")
    energy_per_atom_ev: float = Field(..., description="Energy / n_atoms, eV.")

    max_abs_force_ev_per_ang: float = Field(
        ..., description="max |force| across all atoms (eV/Å)."
    )
    forces: List[MockForceEntry] = Field(
        ..., description="Per-atom force triples (eV/Å)."
    )

    trajectory_n_frames: int = Field(
        ..., ge=1, description="Number of frames in the trajectory.xyz artifact."
    )
    trajectory: List[MockTrajectoryFrame] = Field(
        ...,
        description=(
            "Inline copy of the trajectory frames. Duplicates the "
            "trajectory.xyz artifact in the run dir so clients that only "
            "read outputs_json still get frame positions."
        ),
    )

    # Bookkeeping
    converged: bool = Field(default=True, description="Always True for the mock engine.")
    seed: int = Field(
        ...,
        description="Seed used for the per-run noise draws; derived from structure id.",
    )

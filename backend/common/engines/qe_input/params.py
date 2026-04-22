"""Pydantic schema for the parameter dict driving QE input generation.

Defaults are the PBE solid-state screening defaults that SSSP's
publications recommend. Every field is overridable by the user.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


CalculationKind = Literal["scf", "nscf", "bands", "relax", "vc-relax", "md"]


class QEInputParams(BaseModel):
    """Parameters that control the generated pw.x input file.

    Only the commonly-set fields are here. Raw ``extra_namelist``
    dicts exist for the long tail (rare flags like
    ``assume_isolated``, ``london_*`` DFT-D3 knobs, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    calculation: CalculationKind = Field(
        default="scf",
        description="pw.x calculation keyword.",
    )
    prefix: str = Field(default="orion", max_length=32)
    outdir: str = Field(
        default="./tmp",
        description="Relative to the run directory the backend will cd into.",
    )
    disk_io: Literal["high", "medium", "low", "none"] = "low"
    verbosity: Literal["high", "low"] = "low"

    # Cutoffs — when None, registry picks from max(UPF suggestions).
    ecutwfc: Optional[float] = Field(
        default=None,
        gt=0,
        description="Wavefunction cutoff in Ry.",
    )
    ecutrho: Optional[float] = Field(
        default=None,
        gt=0,
        description="Charge density cutoff in Ry. Defaults to 8*ecutwfc for USPP/PAW.",
    )
    dual: float = Field(
        default=8.0,
        gt=1.0,
        description=(
            "ecutrho / ecutwfc ratio used when ecutrho is not set. "
            "8 is correct for USPP/PAW, 4 for norm-conserving."
        ),
    )

    # SCF convergence
    conv_thr: float = Field(default=1.0e-8, gt=0)
    mixing_beta: float = Field(default=0.4, gt=0, le=1.0)
    electron_maxstep: int = Field(default=200, ge=1, le=10_000)

    # Smearing (metals) — set smearing=None for insulators
    occupations: Literal["smearing", "fixed", "tetrahedra"] = "smearing"
    smearing: Optional[Literal["gaussian", "gauss", "mp", "methfessel-paxton", "mv", "marzari-vanderbilt", "fd", "fermi-dirac"]] = "gauss"
    degauss: float = Field(default=0.01, ge=0, description="Smearing width in Ry.")

    # Spin
    spin_polarized: bool = False
    tot_magnetization: Optional[float] = None

    # Forces / stress
    tstress: bool = True
    tforces: bool = True

    # Relaxation convergence (only for relax / vc-relax)
    forc_conv_thr: float = Field(default=1.0e-4, gt=0)
    press_conv_thr: float = Field(default=0.5, gt=0, description="kbar")

    # K-point grid (None → derive from structure via kgrid_from_structure)
    kpoints: Optional[tuple[int, int, int]] = None
    kpoints_shift: tuple[int, int, int] = Field(default=(0, 0, 0))
    kspacing: float = Field(
        default=0.25,
        gt=0,
        description="Target k-point spacing in 1/Å used when kpoints=None.",
    )

    # Escape hatches for niche flags. Merged verbatim into their
    # respective namelists, after the structured fields.
    extra_control: dict = Field(default_factory=dict)
    extra_system: dict = Field(default_factory=dict)
    extra_electrons: dict = Field(default_factory=dict)

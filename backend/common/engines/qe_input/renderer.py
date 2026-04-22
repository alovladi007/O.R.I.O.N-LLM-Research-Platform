"""Render a pw.x input file from a structure + QEInputParams.

The output of this module is a deterministic string that ``pw.x`` can
parse. We don't invoke QE here; Session 3.2 does that.

Structure convention
--------------------

We accept either:

- A pymatgen ``Structure`` (``lattice.matrix``, ``species``, ``frac_coords``).
- A dict with ``lattice`` (3×3 list), ``species`` (list[str]), and
  ``frac_coords`` (list[list[float]]). Useful for tests and for the
  Celery task which may not want to rehydrate a pymatgen object.

Both convert into a common ``_NormalizedStructure`` internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .params import QEInputParams
from .registry import PseudopotentialRegistry, UnknownElementError


@dataclass(frozen=True)
class KPointGrid:
    """Monkhorst-Pack grid (nk1, nk2, nk3) with optional shift."""

    nk1: int
    nk2: int
    nk3: int
    sk1: int = 0
    sk2: int = 0
    sk3: int = 0


@dataclass
class RenderedInput:
    """Return of :func:`generate_pw_input`."""

    text: str                     # the .in file contents
    input_filename: str           # suggested filename (prefix.in)
    pseudo_files: List[str]       # UPF basenames needed at runtime
    ecutwfc_ry: float
    ecutrho_ry: float
    kpoints: KPointGrid


# ---------------------------------------------------------------------------
# Structure normalization
# ---------------------------------------------------------------------------


@dataclass
class _NormalizedStructure:
    lattice: List[List[float]]         # 3x3 Å
    species: List[str]                 # per-atom element symbols
    frac_coords: List[List[float]]     # per-atom fractional (crystal) coords

    def unique_species(self) -> List[str]:
        """Ordered, deduplicated species list (first-appearance order)."""
        seen = {}
        out: List[str] = []
        for s in self.species:
            if s not in seen:
                seen[s] = True
                out.append(s)
        return out


def _normalize_structure(structure: Any) -> _NormalizedStructure:
    # pymatgen Structure
    if hasattr(structure, "lattice") and hasattr(structure, "frac_coords"):
        lattice = [list(row) for row in structure.lattice.matrix]
        species = [str(s) for s in structure.species]
        coords = [list(c) for c in structure.frac_coords]
        return _NormalizedStructure(lattice, species, coords)

    if isinstance(structure, dict):
        if "lattice" in structure and "species" in structure and "frac_coords" in structure:
            return _NormalizedStructure(
                lattice=[list(r) for r in structure["lattice"]],
                species=[str(s) for s in structure["species"]],
                frac_coords=[list(c) for c in structure["frac_coords"]],
            )

    raise ValueError(
        "generate_pw_input: structure must be a pymatgen Structure or a dict "
        "with keys {lattice, species, frac_coords}."
    )


# ---------------------------------------------------------------------------
# K-point heuristic
# ---------------------------------------------------------------------------


def kgrid_from_structure(
    lattice: Sequence[Sequence[float]],
    *,
    kspacing: float = 0.25,
    min_per_axis: int = 1,
) -> KPointGrid:
    """Compute a Monkhorst-Pack grid from the reciprocal lattice.

    Uses the "spacing in 1/Å" convention: ``n_i = ceil(|b_i| / kspacing)``
    where ``b_i`` is the *i*-th reciprocal lattice vector. This matches
    VASP/ASE's common convention and is close to what SSSP
    benchmarks use.

    Returns a :class:`KPointGrid` with shift (0,0,0) by default.
    """
    a1 = lattice[0]
    a2 = lattice[1]
    a3 = lattice[2]

    def cross(u, v):
        return [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    def norm(v):
        return math.sqrt(dot(v, v))

    volume = dot(a1, cross(a2, a3))
    if abs(volume) < 1e-10:
        raise ValueError("lattice is singular (zero volume)")

    b1 = [2 * math.pi * x / volume for x in cross(a2, a3)]
    b2 = [2 * math.pi * x / volume for x in cross(a3, a1)]
    b3 = [2 * math.pi * x / volume for x in cross(a1, a2)]

    nk = [
        max(min_per_axis, int(math.ceil(norm(b) / kspacing)))
        for b in (b1, b2, b3)
    ]
    return KPointGrid(nk1=nk[0], nk2=nk[1], nk3=nk[2])


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def generate_pw_input(
    structure: Any,
    params: QEInputParams,
    registry: PseudopotentialRegistry,
    *,
    pseudo_dir: Optional[str] = None,
) -> RenderedInput:
    """Build the pw.x input file text.

    Parameters
    ----------
    structure
        pymatgen ``Structure`` or dict (see module docstring).
    params
        :class:`QEInputParams`. Cutoffs may be ``None`` — we derive
        them from the registry's recommended values.
    registry
        Pseudopotential registry for the library on disk.
    pseudo_dir
        Path written into ``&CONTROL/pseudo_dir``. Defaults to the
        registry's own path.
    """
    ns = _normalize_structure(structure)
    if not ns.species:
        raise ValueError("structure has no atoms")

    unique = ns.unique_species()

    # Resolve pseudo entries up front so we fail fast on unknown
    # elements, and so cutoff inference has all data it needs.
    pseudos = []
    for el in unique:
        try:
            pseudos.append(registry.get(el))
        except UnknownElementError:
            raise

    # Cutoffs
    ecutwfc = params.ecutwfc or registry.recommended_ecutwfc(unique)
    ecutrho = params.ecutrho or max(
        registry.recommended_ecutrho(unique),
        params.dual * ecutwfc,
    )

    # K-points
    if params.kpoints is not None:
        kg = KPointGrid(
            nk1=params.kpoints[0], nk2=params.kpoints[1], nk3=params.kpoints[2],
            sk1=params.kpoints_shift[0], sk2=params.kpoints_shift[1], sk3=params.kpoints_shift[2],
        )
    else:
        kg = kgrid_from_structure(ns.lattice, kspacing=params.kspacing)
        kg = KPointGrid(
            nk1=kg.nk1, nk2=kg.nk2, nk3=kg.nk3,
            sk1=params.kpoints_shift[0], sk2=params.kpoints_shift[1], sk3=params.kpoints_shift[2],
        )

    dir_for_pseudos = pseudo_dir or str(registry.pseudo_dir)

    lines: List[str] = []

    # &CONTROL
    lines.append("&CONTROL")
    lines.append(f"  calculation = '{params.calculation}'")
    lines.append(f"  prefix = '{params.prefix}'")
    lines.append(f"  outdir = '{params.outdir}'")
    lines.append(f"  pseudo_dir = '{dir_for_pseudos}'")
    lines.append(f"  verbosity = '{params.verbosity}'")
    lines.append(f"  disk_io = '{params.disk_io}'")
    lines.append(f"  tstress = .{str(params.tstress).lower()}.")
    lines.append(f"  tprnfor = .{str(params.tforces).lower()}.")
    if params.calculation in ("relax", "vc-relax", "md"):
        lines.append(f"  forc_conv_thr = {_fortran_float(params.forc_conv_thr)}")
    if params.calculation == "vc-relax":
        lines.append(f"  press_conv_thr = {params.press_conv_thr}")
    for k, v in params.extra_control.items():
        lines.append(f"  {k} = {_fortran_value(v)}")
    lines.append("/")
    lines.append("")

    # &SYSTEM
    lines.append("&SYSTEM")
    lines.append("  ibrav = 0")
    lines.append(f"  nat = {len(ns.species)}")
    lines.append(f"  ntyp = {len(unique)}")
    lines.append(f"  ecutwfc = {ecutwfc:.1f}")
    lines.append(f"  ecutrho = {ecutrho:.1f}")
    if params.occupations == "smearing":
        lines.append("  occupations = 'smearing'")
        lines.append(f"  smearing = '{params.smearing}'")
        lines.append(f"  degauss = {_fortran_float(params.degauss)}")
    elif params.occupations == "fixed":
        lines.append("  occupations = 'fixed'")
    elif params.occupations == "tetrahedra":
        lines.append("  occupations = 'tetrahedra'")
    if params.spin_polarized:
        lines.append("  nspin = 2")
        if params.tot_magnetization is not None:
            lines.append(f"  tot_magnetization = {params.tot_magnetization}")
    for k, v in params.extra_system.items():
        lines.append(f"  {k} = {_fortran_value(v)}")
    lines.append("/")
    lines.append("")

    # &ELECTRONS
    lines.append("&ELECTRONS")
    lines.append(f"  conv_thr = {_fortran_float(params.conv_thr)}")
    lines.append(f"  mixing_beta = {params.mixing_beta}")
    lines.append(f"  electron_maxstep = {params.electron_maxstep}")
    for k, v in params.extra_electrons.items():
        lines.append(f"  {k} = {_fortran_value(v)}")
    lines.append("/")
    lines.append("")

    # &IONS / &CELL (relaxation)
    if params.calculation in ("relax", "vc-relax", "md"):
        lines.append("&IONS")
        lines.append("  ion_dynamics = 'bfgs'")
        lines.append("/")
        lines.append("")
    if params.calculation == "vc-relax":
        lines.append("&CELL")
        lines.append("  cell_dynamics = 'bfgs'")
        lines.append("/")
        lines.append("")

    # ATOMIC_SPECIES
    lines.append("ATOMIC_SPECIES")
    for entry in pseudos:
        lines.append(f"  {entry.element:<3s} {entry.mass:>10.4f}  {entry.filename}")
    lines.append("")

    # CELL_PARAMETERS (ibrav=0 requires them *before* ATOMIC_POSITIONS in strict mode
    # but pw.x accepts either order; we put cell first for clarity).
    lines.append("CELL_PARAMETERS angstrom")
    for row in ns.lattice:
        lines.append(f"  {row[0]:>14.8f} {row[1]:>14.8f} {row[2]:>14.8f}")
    lines.append("")

    # ATOMIC_POSITIONS
    lines.append("ATOMIC_POSITIONS crystal")
    for sym, pos in zip(ns.species, ns.frac_coords):
        lines.append(
            f"  {sym:<3s} {pos[0]:>14.8f} {pos[1]:>14.8f} {pos[2]:>14.8f}"
        )
    lines.append("")

    # K_POINTS
    lines.append("K_POINTS automatic")
    lines.append(
        f"  {kg.nk1} {kg.nk2} {kg.nk3}  {kg.sk1} {kg.sk2} {kg.sk3}"
    )
    lines.append("")

    return RenderedInput(
        text="\n".join(lines),
        input_filename=f"{params.prefix}.in",
        pseudo_files=[p.filename for p in pseudos],
        ecutwfc_ry=ecutwfc,
        ecutrho_ry=ecutrho,
        kpoints=kg,
    )


# ---------------------------------------------------------------------------
# Fortran-flavored value formatters
# ---------------------------------------------------------------------------


def _fortran_float(x: float) -> str:
    """Render a float in the 1.0d-8 form QE prefers for small numbers."""
    if x == 0:
        return "0.0d0"
    abs_x = abs(x)
    if 1e-3 <= abs_x < 1e3:
        return f"{x:.6g}"
    # Scientific → Fortran d-exponent.
    s = f"{x:.6e}"
    mantissa, exp = s.split("e")
    return f"{mantissa}d{int(exp)}"


def _fortran_value(v: Any) -> str:
    """Render an arbitrary Python value for a Fortran namelist."""
    if isinstance(v, bool):
        return ".true." if v else ".false."
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        return _fortran_float(v)
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)

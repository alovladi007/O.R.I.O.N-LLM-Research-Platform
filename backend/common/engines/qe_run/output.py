"""Parse ``pw.x`` output files into structured Python.

What we extract (vs. pymatgen's PWOutput):

- **Total energy** — the ``!    total energy`` marker and the running
  SCF energies. We report both the final value and the full history.
- **Forces** — per-atom Cartesian force vector (eV/Å, converted from
  Ry/Bohr).
- **Stress tensor** — 3×3, in GPa (converted from kbar).
- **Convergence** — SCF iteration count, final SCF accuracy,
  whether ``convergence has been achieved`` was printed.
- **Timings** — wall time in seconds if present.
- **Error markers** — common ``%%%`` panic messages, e.g. charge
  density instability, wrong atomic species, SCF non-convergence.

Units
-----

QE prints in Ry / Bohr / kbar. We convert to ORION conventions
(eV / Å / GPa) on parse. All floats are in the converted units unless
the attribute name carries the original unit (``_ry``, ``_bohr``).
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Physical constants for unit conversion
# ---------------------------------------------------------------------------

RY_TO_EV = 13.605693122994       # CODATA 2018
BOHR_TO_ANG = 0.529177210903
RY_PER_BOHR_TO_EV_PER_ANG = RY_TO_EV / BOHR_TO_ANG  # ≈ 25.7112
KBAR_TO_GPA = 0.1


class PWOutputParseError(ValueError):
    """Raised when the output file is malformed in a way we can't recover from."""


class ConvergenceStatus(str, enum.Enum):
    CONVERGED = "converged"
    UNCONVERGED = "unconverged"              # SCF itself failed to converge (fatal)
    BFGS_UNCONVERGED = "bfgs_unconverged"    # SCF converged but geometry relax
                                             # stalled. Final energy usable.
    UNKNOWN = "unknown"          # parse couldn't tell — treat as failure
    ERRORED = "errored"          # explicit error marker in output


# ---------------------------------------------------------------------------
# Lightweight result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ParsedEnergy:
    """Total energy in eV + the SCF history."""

    total_ev: float
    total_ry: float
    scf_history_ry: List[float] = field(default_factory=list)

    @property
    def scf_history_ev(self) -> List[float]:
        return [e * RY_TO_EV for e in self.scf_history_ry]


@dataclass
class ParsedForce:
    """One atom's force (eV/Å) — species + three components."""

    atom_index: int     # zero-based
    species: str
    fx: float
    fy: float
    fz: float

    def max_component(self) -> float:
        return max(abs(self.fx), abs(self.fy), abs(self.fz))


@dataclass
class ParsedStress:
    """Stress tensor (GPa) as a 3×3 list-of-lists + derived pressure."""

    tensor_gpa: List[List[float]]

    @property
    def pressure_gpa(self) -> float:
        """Hydrostatic pressure = -trace/3."""
        t = self.tensor_gpa
        return -(t[0][0] + t[1][1] + t[2][2]) / 3.0


@dataclass
class RelaxedStructure:
    """Relaxed geometry extracted from a ``vc-relax`` or ``relax`` run.

    Units: lattice in Å, Cartesian positions in Å, species as symbols.
    When the run was SCF-only, all fields are ``None``.
    """

    lattice_ang: List[List[float]]
    species: List[str]
    cart_coords_ang: List[List[float]]
    alat_bohr: Optional[float] = None

    @property
    def a_lattice_const_ang(self) -> float:
        """|a1| — the first lattice vector's length in Å."""
        a = self.lattice_ang[0]
        return (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5

    @property
    def volume_ang3(self) -> float:
        a, b, c = self.lattice_ang
        # |a · (b × c)|
        bxc = (
            b[1] * c[2] - b[2] * c[1],
            b[2] * c[0] - b[0] * c[2],
            b[0] * c[1] - b[1] * c[0],
        )
        return abs(a[0] * bxc[0] + a[1] * bxc[1] + a[2] * bxc[2])


@dataclass
class BandKpoint:
    """One k-point's eigenvalues (all in eV, shifted so VBM=0 when possible)."""

    index: int
    kx: float
    ky: float
    kz: float
    eigenvalues_ev: List[float]


@dataclass
class ParsedBands:
    """Band-structure block from a bands-mode pw.x run."""

    n_kpoints: int
    n_bands: int
    kpoints: List[BandKpoint]
    # Fermi energy if printed by a preceding SCF (we often merge two outputs).
    fermi_energy_ev: Optional[float] = None


@dataclass
class ParsedDOS:
    """Density-of-states block — header fields + (E, total, integrated) rows."""

    fermi_energy_ev: Optional[float]
    energies_ev: List[float]
    dos: List[float]
    idos: List[float]    # integrated DOS

    def integrate_up_to(self, energy_ev: float) -> float:
        """Trapezoidal integral of the DOS up to *energy_ev*.

        Useful for the electron-count sanity check (∫DOS up to E_F should
        equal total electrons ±2%).
        """
        if not self.energies_ev:
            return 0.0
        total = 0.0
        prev_e = self.energies_ev[0]
        prev_d = self.dos[0]
        for e, d in zip(self.energies_ev[1:], self.dos[1:]):
            if e > energy_ev:
                break
            total += 0.5 * (d + prev_d) * (e - prev_e)
            prev_e, prev_d = e, d
        return total


@dataclass
class ParsedPhononsGamma:
    """Γ-point phonon frequencies from a ph.x calculation.

    Units: cm⁻¹ (QE's native) and THz (derived).
    """

    frequencies_cm1: List[float]

    @property
    def frequencies_thz(self) -> List[float]:
        # 1 cm⁻¹ ≈ 0.02998 THz
        return [f * 0.029979245800 for f in self.frequencies_cm1]

    @property
    def n_imaginary(self) -> int:
        """How many modes are imaginary (QE prints them as negative)."""
        return sum(1 for f in self.frequencies_cm1 if f < -1.0)


@dataclass
class PWOutput:
    """Structured view of one ``pw.x`` run's output file."""

    energy: Optional[ParsedEnergy]
    forces: List[ParsedForce]
    stress: Optional[ParsedStress]
    convergence: ConvergenceStatus
    n_scf_iterations: Optional[int]
    scf_accuracy_ry: Optional[float]
    wall_time_seconds: Optional[float]
    error_messages: List[str]
    # Raw quantities for debugging / future features.
    n_atoms: Optional[int] = None
    n_species: Optional[int] = None
    n_electrons: Optional[float] = None
    # Session 3.3 additions
    relaxed: Optional["RelaxedStructure"] = None
    bands: Optional["ParsedBands"] = None
    fermi_energy_ev: Optional[float] = None

    def as_dict(self) -> dict:
        d = {
            "energy_ev": self.energy.total_ev if self.energy else None,
            "energy_ry": self.energy.total_ry if self.energy else None,
            "scf_history_ev": self.energy.scf_history_ev if self.energy else [],
            "forces": [
                {"index": f.atom_index, "species": f.species, "fx": f.fx, "fy": f.fy, "fz": f.fz}
                for f in self.forces
            ],
            "max_force_ev_per_ang": self.max_abs_force(),
            "stress_gpa": self.stress.tensor_gpa if self.stress else None,
            "pressure_gpa": self.stress.pressure_gpa if self.stress else None,
            "convergence": self.convergence.value,
            "n_scf_iterations": self.n_scf_iterations,
            "scf_accuracy_ry": self.scf_accuracy_ry,
            "wall_time_seconds": self.wall_time_seconds,
            "n_atoms": self.n_atoms,
            "n_species": self.n_species,
            "n_electrons": self.n_electrons,
            "error_messages": self.error_messages,
            "fermi_energy_ev": self.fermi_energy_ev,
        }
        if self.relaxed is not None:
            d["relaxed"] = {
                "lattice_ang": self.relaxed.lattice_ang,
                "species": self.relaxed.species,
                "cart_coords_ang": self.relaxed.cart_coords_ang,
                "a_lattice_const_ang": self.relaxed.a_lattice_const_ang,
                "volume_ang3": self.relaxed.volume_ang3,
            }
        if self.bands is not None:
            d["bands"] = {
                "n_kpoints": self.bands.n_kpoints,
                "n_bands": self.bands.n_bands,
                "fermi_energy_ev": self.bands.fermi_energy_ev,
                "kpoints": [
                    {"index": k.index, "kx": k.kx, "ky": k.ky, "kz": k.kz,
                     "eigenvalues_ev": k.eigenvalues_ev}
                    for k in self.bands.kpoints
                ],
            }
        return d

    def max_abs_force(self) -> Optional[float]:
        if not self.forces:
            return None
        return max(f.max_component() for f in self.forces)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------


# Final "!    total energy              =    -15.83 Ry"
_FINAL_ENERGY_RE = re.compile(r"^!\s+total energy\s*=\s*([-\d.Ee+]+)\s*Ry", re.MULTILINE)

# Per-SCF "total energy              =    -15.83 Ry" (also picks up final).
_SCF_ENERGY_RE = re.compile(r"^\s+total energy\s*=\s*([-\d.Ee+]+)\s*Ry", re.MULTILINE)

# "convergence has been achieved in   6 iterations"
_SCF_CONVERGED_RE = re.compile(
    r"convergence has been achieved in\s+(\d+)\s+iterations", re.IGNORECASE,
)

# "     scf convergence threshold =   1.0E-08"   (just a sanity read)
_SCF_THRESHOLD_RE = re.compile(
    r"scf\s+convergence\s+threshold\s*=\s*([-\d.Ee+]+)", re.IGNORECASE,
)

# "     estimated scf accuracy    <     0.00000008 Ry"
_SCF_ACCURACY_RE = re.compile(
    r"estimated scf accuracy\s*<\s*([-\d.Ee+]+)\s*Ry",
)

# Two kinds of "NOT achieved" pw.x emits:
#
# 1. SCF non-convergence — fatal: "convergence NOT achieved after N
#    iterations: stopping". This means the electronic loop diverged
#    and no trustworthy energy exists.
# 2. BFGS non-convergence — often NOT fatal: "bfgs failed after X scf
#    cycles and Y bfgs steps, convergence not achieved". Here each SCF
#    did converge; only the geometry-relaxation loop hit the step
#    limit. The final energy is usable as an approximate minimum.
#
# Different status codes for each so the task layer can decide.
_SCF_UNCONV_RE = re.compile(
    r"convergence\s+NOT\s+achieved\s+after\s+\d+\s+iterations:\s+stopping",
    re.IGNORECASE,
)
_BFGS_UNCONV_RE = re.compile(
    r"bfgs\s+failed\s+after.*convergence\s+not\s+achieved", re.IGNORECASE,
)

# Atomic counts from the preamble
_NAT_RE = re.compile(r"number of atoms/cell\s*=\s*(\d+)")
_NTYP_RE = re.compile(r"number of atomic types\s*=\s*(\d+)")
_NELEC_RE = re.compile(r"number of electrons\s*=\s*([-\d.]+)")

# Wall time: "     PWSCF        :     0.30s CPU      0.40s WALL"
# pw.x emits the final PWSCF timer in two styles:
#   "PWSCF        :     0.30s CPU      0.40s WALL"            (short runs)
#   "PWSCF        :   5m56.46s CPU   6m21.02s WALL"            (longer)
#   "PWSCF        :  1h10m 5.12s CPU  1h11m20.5s WALL"         (very long)
# We accept all three and convert to seconds.
_WALL_RE = re.compile(
    r"PWSCF\s*:\s*"
    r"(?:\d+h\s*)?(?:\d+m\s*)?[\d.]+s\s+CPU\s+"
    r"(?:(\d+)h\s*)?(?:(\d+)m\s*)?([\d.]+)s\s+WALL",
    re.IGNORECASE,
)

# Forces block starts with "Forces acting on atoms (cartesian axes, Ry/au):"
# then atom lines "atom    1 type  1   force =    0.001  0.002  0.003"
_FORCES_START_RE = re.compile(
    r"Forces acting on atoms.*Ry/au", re.IGNORECASE,
)
_FORCE_LINE_RE = re.compile(
    r"^\s*atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*"
    r"([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)",
)

# Stress block:
#   total   stress  (Ry/bohr**3)                   (kbar)     P=    3.21
#           0.00001234 -0.00000056  0.00000000     1.81   -0.08   0.00
_STRESS_START_RE = re.compile(
    r"total\s+stress.*\(kbar\).*P\s*=\s*([-\d.Ee+]+)", re.IGNORECASE,
)
# Lines of 6 floats: 3 Ry/bohr³ then 3 kbar.
_STRESS_LINE_RE = re.compile(
    r"^\s*[-\d.Ee+]+\s+[-\d.Ee+]+\s+[-\d.Ee+]+\s+"
    r"([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*$"
)

# Panic markers — "%%%%" banners mean pw.x bailed.
_ERROR_BANNER_RE = re.compile(r"^\s*%%%+", re.MULTILINE)
_ERROR_TEXT_RE = re.compile(
    r"^\s*Error in routine\s+(.+)$", re.MULTILINE | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Relaxation output (vc-relax / relax)
# ---------------------------------------------------------------------------


# "lattice parameter (alat)  =      10.2000  a.u."
_ALAT_RE = re.compile(r"lattice parameter \(alat\)\s*=\s*([-\d.Ee+]+)\s*a\.u\.")

# Preamble crystal axes, used when CELL_PARAMETERS is absent:
#   a(1) = (   1.000000   0.000000   0.000000 )
# All three ai vectors follow each other. Units are alat.
_PREAMBLE_AXES_RE = re.compile(
    r"a\(1\)\s*=\s*\(\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\)\s*\n"
    r"\s*a\(2\)\s*=\s*\(\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\)\s*\n"
    r"\s*a\(3\)\s*=\s*\(\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\)"
)
# "CELL_PARAMETERS (alat= 10.20)" followed by 3 lines of 3 floats
_CELL_PARAMS_ALAT_RE = re.compile(
    r"CELL_PARAMETERS \(alat=\s*([-\d.Ee+]+)\)\s*\n"
    r"\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\n"
    r"\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\n"
    r"\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)"
)
# "CELL_PARAMETERS (angstrom)" variant (vc-relax sometimes prints angstrom)
_CELL_PARAMS_ANG_RE = re.compile(
    r"CELL_PARAMETERS \(angstrom\)\s*\n"
    r"\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\n"
    r"\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\n"
    r"\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)"
)
# "ATOMIC_POSITIONS (crystal|angstrom|alat)" followed by N atom lines
_ATOMIC_POSITIONS_BLOCK_RE = re.compile(
    r"ATOMIC_POSITIONS \((\w+)\)\s*\n((?:\s*[A-Z][a-z]?\s+[-\d.Ee+]+\s+[-\d.Ee+]+\s+[-\d.Ee+]+[^\n]*\n)+)"
)
_POSITION_LINE_RE = re.compile(
    r"\s*([A-Z][a-z]?)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)"
)


# ---------------------------------------------------------------------------
# Bands (non-SCF output)
# ---------------------------------------------------------------------------


# "          k = 0.0000 0.0000 0.0000 (  469 PWs)   bands (ev):"
_KPOINT_HEADER_RE = re.compile(
    r"^\s+k\s*=\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)\s+([-\d.Ee+]+)\s*\([^)]+\)\s*bands\s*\(ev\):",
    re.MULTILINE,
)
# Eigenvalues follow the k header, 8 per line typically.
_EIGENVALUE_LINE_RE = re.compile(r"^\s+([-\d.Ee+\s]+)\s*$", re.MULTILINE)
# "     the Fermi energy is     6.0000 ev"
_FERMI_ENERGY_RE = re.compile(r"the Fermi energy is\s+([-\d.Ee+]+)\s*ev", re.IGNORECASE)
# "     highest occupied, lowest unoccupied level (ev):    6.0000  6.8000"
_HOMO_LUMO_RE = re.compile(
    r"highest occupied,\s+lowest unoccupied level\s*\(ev\):\s*([-\d.Ee+]+)\s+([-\d.Ee+]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Phonons (ph.x output)
# ---------------------------------------------------------------------------


# Γ-point block in ph.x output:
#   "     freq (    1) =       0.000 [THz] =       0.000 [cm-1]"
# We match the cm-1 column.
_PH_FREQ_RE = re.compile(
    r"freq\s*\(\s*\d+\s*\)\s*=.*?\[THz\]\s*=\s*([-\d.Ee+]+)\s*\[cm-1\]",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_pw_output(source: str | Path) -> PWOutput:
    """Parse a ``pw.x`` output file (or its text) into :class:`PWOutput`.

    Parameters
    ----------
    source
        Path to the output file, or the output text itself.

    Returns
    -------
    PWOutput
        Structured result. Missing sections are ``None``/empty rather
        than raised — a crashed pw.x still gives us useful partial
        information (e.g. error_messages) to persist on the job row.
    """
    text = _load_text(source)
    if not text.strip():
        raise PWOutputParseError("empty output")

    energy = _parse_energy(text)
    forces = _parse_forces(text)
    stress = _parse_stress(text)
    convergence = _parse_convergence(text)
    n_scf = _maybe_int(_SCF_CONVERGED_RE, text)
    scf_acc = _maybe_float(_SCF_ACCURACY_RE, text)
    wall = _parse_wall_seconds(text)
    errors = _collect_errors(text)
    nat = _maybe_int(_NAT_RE, text)
    ntyp = _maybe_int(_NTYP_RE, text)
    nelec = _maybe_float(_NELEC_RE, text)
    # Session 3.3 additions
    relaxed = _parse_relaxed(text)
    bands = _parse_bands(text)
    fermi_ev = _maybe_float(_FERMI_ENERGY_RE, text)

    return PWOutput(
        energy=energy,
        forces=forces,
        stress=stress,
        convergence=convergence,
        n_scf_iterations=n_scf,
        scf_accuracy_ry=scf_acc,
        wall_time_seconds=wall,
        error_messages=errors,
        n_atoms=nat,
        n_species=ntyp,
        relaxed=relaxed,
        bands=bands,
        fermi_energy_ev=fermi_ev,
        n_electrons=nelec,
    )


# ---------------------------------------------------------------------------
# Section extractors
# ---------------------------------------------------------------------------


def _load_text(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8", errors="replace")
    # Heuristic: a non-empty path-sized single-line string with no
    # newlines — treat as a path. Anything else is text.
    if (
        isinstance(source, str)
        and source.strip()
        and "\n" not in source
        and len(source) < 4096
    ):
        candidate = Path(source)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8", errors="replace")
    return source


def _parse_energy(text: str) -> Optional[ParsedEnergy]:
    final = _FINAL_ENERGY_RE.search(text)
    if final is None:
        return None
    total_ry = float(final.group(1))
    history = [float(m.group(1)) for m in _SCF_ENERGY_RE.finditer(text)]
    return ParsedEnergy(
        total_ev=total_ry * RY_TO_EV,
        total_ry=total_ry,
        scf_history_ry=history,
    )


def _parse_forces(text: str) -> List[ParsedForce]:
    # Find the *last* "Forces acting on atoms" block (pw.x prints it
    # once per SCF cycle in relax runs; we want the last one).
    starts = list(_FORCES_START_RE.finditer(text))
    if not starts:
        return []
    block_start = starts[-1].end()
    block = text[block_start:block_start + 20_000]  # plenty for any real cell

    out: List[ParsedForce] = []
    for line in block.splitlines():
        if line.strip().startswith("The non-local contrib"):
            break  # section ends
        m = _FORCE_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1  # pw.x is 1-based
        fx_ry = float(m.group(2)) * RY_PER_BOHR_TO_EV_PER_ANG
        fy_ry = float(m.group(3)) * RY_PER_BOHR_TO_EV_PER_ANG
        fz_ry = float(m.group(4)) * RY_PER_BOHR_TO_EV_PER_ANG
        out.append(ParsedForce(
            atom_index=idx,
            species="?",       # filled in by the runner when it has the input structure
            fx=fx_ry, fy=fy_ry, fz=fz_ry,
        ))
    return out


def _parse_stress(text: str) -> Optional[ParsedStress]:
    m = _STRESS_START_RE.search(text)
    if m is None:
        return None
    block = text[m.end():m.end() + 2000]
    rows: List[List[float]] = []
    for line in block.splitlines():
        if len(rows) == 3:
            break
        mm = _STRESS_LINE_RE.match(line)
        if not mm:
            continue
        row_kbar = [float(mm.group(i)) for i in (1, 2, 3)]
        rows.append([v * KBAR_TO_GPA for v in row_kbar])
    if len(rows) != 3:
        return None
    return ParsedStress(tensor_gpa=rows)


def _parse_convergence(text: str) -> ConvergenceStatus:
    # Fatal SCF divergence trumps everything — the energy is untrustworthy.
    if _SCF_UNCONV_RE.search(text):
        return ConvergenceStatus.UNCONVERGED
    # BFGS (geometry-relax) hit its step limit but individual SCFs
    # converged. Energy is approximate but usable; the caller gets a
    # distinct status so they can decide whether to accept it.
    if _BFGS_UNCONV_RE.search(text):
        return ConvergenceStatus.BFGS_UNCONVERGED
    if _SCF_CONVERGED_RE.search(text):
        return ConvergenceStatus.CONVERGED
    if _ERROR_BANNER_RE.search(text):
        return ConvergenceStatus.ERRORED
    return ConvergenceStatus.UNKNOWN


def _parse_wall_seconds(text: str) -> Optional[float]:
    """Extract PWSCF wall time. Accepts h/m/s forms; converts to seconds."""
    m = _WALL_RE.search(text)
    if m is None:
        return None
    h = int(m.group(1)) if m.group(1) else 0
    mm = int(m.group(2)) if m.group(2) else 0
    try:
        s = float(m.group(3))
    except (TypeError, ValueError):
        return None
    return h * 3600.0 + mm * 60.0 + s


def _collect_errors(text: str) -> List[str]:
    out: List[str] = []
    for m in _ERROR_TEXT_RE.finditer(text):
        out.append(m.group(1).strip())
    return out


def _maybe_int(pattern: re.Pattern, text: str) -> Optional[int]:
    m = pattern.search(text)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _maybe_float(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.search(text)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------


def _parse_relaxed(text: str) -> Optional[RelaxedStructure]:
    """Extract the final relaxed lattice + positions from a vc-relax / relax run.

    pw.x emits a final ``CELL_PARAMETERS`` block (for vc-relax only) and
    a final ``ATOMIC_POSITIONS`` block after ``Begin final coordinates``.
    For plain ``relax`` (fixed cell), the lattice stays as the input
    ``alat`` value — we reuse the initial ``alat`` + the identity cell
    shape in that case.
    """
    if "Begin final coordinates" not in text:
        return None

    # Lattice: look for the LAST CELL_PARAMETERS block (vc-relax writes
    # several as it iterates).
    lat_ang: Optional[List[List[float]]] = None
    alat_bohr: Optional[float] = None

    ang_matches = list(_CELL_PARAMS_ANG_RE.finditer(text))
    alat_matches = list(_CELL_PARAMS_ALAT_RE.finditer(text))

    # Prefer the angstrom variant when available; else convert alat units.
    if ang_matches:
        m = ang_matches[-1]
        lat_ang = [
            [float(m.group(i)) for i in (1, 2, 3)],
            [float(m.group(i)) for i in (4, 5, 6)],
            [float(m.group(i)) for i in (7, 8, 9)],
        ]
    elif alat_matches:
        m = alat_matches[-1]
        alat_bohr = float(m.group(1))
        scale_ang = alat_bohr * BOHR_TO_ANG
        lat_ang = [
            [float(m.group(i)) * scale_ang for i in (2, 3, 4)],
            [float(m.group(i)) * scale_ang for i in (5, 6, 7)],
            [float(m.group(i)) * scale_ang for i in (8, 9, 10)],
        ]
    else:
        # Plain relax: cell is unchanged. Pull it from the preamble's
        # "crystal axes" block, which prints a_i in units of alat.
        alat_m = _ALAT_RE.search(text)
        axes_m = _PREAMBLE_AXES_RE.search(text)
        if alat_m is None or axes_m is None:
            return None
        alat_bohr = float(alat_m.group(1))
        scale_ang = alat_bohr * BOHR_TO_ANG
        lat_ang = [
            [float(axes_m.group(i)) * scale_ang for i in (1, 2, 3)],
            [float(axes_m.group(i)) * scale_ang for i in (4, 5, 6)],
            [float(axes_m.group(i)) * scale_ang for i in (7, 8, 9)],
        ]

    # Positions: last ATOMIC_POSITIONS block after "Begin final coordinates".
    final_block_start = text.index("Begin final coordinates")
    pos_matches = list(_ATOMIC_POSITIONS_BLOCK_RE.finditer(text, final_block_start))
    if not pos_matches:
        return None
    pos_match = pos_matches[-1]
    unit = pos_match.group(1).lower()
    raw = pos_match.group(2)

    species: List[str] = []
    coords: List[List[float]] = []
    for line in raw.splitlines():
        pm = _POSITION_LINE_RE.match(line)
        if not pm:
            continue
        species.append(pm.group(1))
        c = [float(pm.group(i)) for i in (2, 3, 4)]
        if unit == "crystal":
            # Convert fractional → Cartesian using lat_ang.
            cx = c[0] * lat_ang[0][0] + c[1] * lat_ang[1][0] + c[2] * lat_ang[2][0]
            cy = c[0] * lat_ang[0][1] + c[1] * lat_ang[1][1] + c[2] * lat_ang[2][1]
            cz = c[0] * lat_ang[0][2] + c[1] * lat_ang[1][2] + c[2] * lat_ang[2][2]
            coords.append([cx, cy, cz])
        elif unit == "angstrom":
            coords.append(c)
        elif unit == "alat":
            # Already in alat-relative Cartesian; convert to Å.
            scale = (alat_bohr or 1.0) * BOHR_TO_ANG
            coords.append([v * scale for v in c])
        elif unit == "bohr":
            coords.append([v * BOHR_TO_ANG for v in c])
        else:
            coords.append(c)

    if not species:
        return None
    return RelaxedStructure(
        lattice_ang=lat_ang,
        species=species,
        cart_coords_ang=coords,
        alat_bohr=alat_bohr,
    )


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------


def _parse_bands(text: str) -> Optional[ParsedBands]:
    """Extract band eigenvalues from a bands-mode or nscf pw.x run.

    Bands mode prints ``k = kx ky kz (N PWs)  bands (ev):`` followed by
    free-form eigenvalue rows, then a blank line. We stream through each
    header match and collect eigenvalues until the next header or until
    a non-eigenvalue line.
    """
    headers = list(_KPOINT_HEADER_RE.finditer(text))
    if not headers:
        return None

    kpoints: List[BandKpoint] = []
    for idx, m in enumerate(headers):
        kx, ky, kz = (float(m.group(i)) for i in (1, 2, 3))
        block_start = m.end()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
        block = text[block_start:block_end]
        eigs: List[float] = []
        for raw_line in block.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                if eigs:
                    # Blank line after at least one eigenvalue row ends the block.
                    break
                continue
            # Reject section markers / transition lines.
            if re.match(r"^[A-Za-z]", stripped):
                break
            parts = stripped.split()
            try:
                eigs.extend(float(p) for p in parts)
            except ValueError:
                break
        if eigs:
            kpoints.append(BandKpoint(
                index=idx, kx=kx, ky=ky, kz=kz, eigenvalues_ev=eigs,
            ))

    if not kpoints:
        return None

    n_bands = max((len(k.eigenvalues_ev) for k in kpoints), default=0)
    fermi = _maybe_float(_FERMI_ENERGY_RE, text)
    return ParsedBands(
        n_kpoints=len(kpoints),
        n_bands=n_bands,
        kpoints=kpoints,
        fermi_energy_ev=fermi,
    )


# ---------------------------------------------------------------------------
# DOS (dos.x output)
# ---------------------------------------------------------------------------


def parse_dos_output(dos_dat_path: Path | str) -> ParsedDOS:
    """Parse ``<prefix>.dos`` written by ``dos.x``.

    Format (QE 6.x/7.x):

        #  E (eV)   dos(E)     Int dos(E) EFermi =    6.0000 eV
        -8.000  0.0000  0.0000
        ...

    Multi-column variants (spin-polarised) also exist; we take the last
    two columns + the energy column so non-spin and spin-up/dn both work.
    """
    path = Path(dos_dat_path)
    if not path.is_file():
        raise PWOutputParseError(f"dos file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")

    fermi_ev: Optional[float] = None
    energies: List[float] = []
    dos: List[float] = []
    idos: List[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m = re.search(r"EFermi\s*=\s*([-\d.Ee+]+)\s*eV", stripped)
            if m:
                try:
                    fermi_ev = float(m.group(1))
                except ValueError:
                    pass
            continue
        parts = stripped.split()
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            continue
        if len(nums) < 3:
            continue
        energies.append(nums[0])
        dos.append(nums[-2])
        idos.append(nums[-1])

    if not energies:
        raise PWOutputParseError(f"no DOS rows parsed from {path}")
    return ParsedDOS(
        fermi_energy_ev=fermi_ev,
        energies_ev=energies,
        dos=dos,
        idos=idos,
    )


# ---------------------------------------------------------------------------
# Phonons (ph.x output)
# ---------------------------------------------------------------------------


def parse_ph_output(source: str | Path) -> ParsedPhononsGamma:
    """Parse ``ph.x`` output for Γ-point frequencies.

    ``ph.x`` prints frequencies as::

        freq (  1) =       0.000 [THz] =       0.000 [cm-1]
        freq (  2) =       0.000 [THz] =       0.000 [cm-1]

    For a supercell with N atoms you get 3N modes; the first three are
    acoustic (≈0 cm⁻¹). Imaginary modes print as negative.
    """
    text = _load_text(source)
    if not text.strip():
        raise PWOutputParseError("empty ph.x output")
    freqs = [float(m.group(1)) for m in _PH_FREQ_RE.finditer(text)]
    if not freqs:
        raise PWOutputParseError("no phonon frequencies found")
    return ParsedPhononsGamma(frequencies_cm1=freqs)

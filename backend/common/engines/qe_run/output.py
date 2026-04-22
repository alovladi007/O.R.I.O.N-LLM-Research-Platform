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
    UNCONVERGED = "unconverged"
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

    def as_dict(self) -> dict:
        return {
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
        }

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

# "     convergence NOT achieved after 200 iterations: stopping"
_SCF_UNCONV_RE = re.compile(
    r"convergence\s+NOT\s+achieved", re.IGNORECASE,
)

# Atomic counts from the preamble
_NAT_RE = re.compile(r"number of atoms/cell\s*=\s*(\d+)")
_NTYP_RE = re.compile(r"number of atomic types\s*=\s*(\d+)")
_NELEC_RE = re.compile(r"number of electrons\s*=\s*([-\d.]+)")

# Wall time: "     PWSCF        :     0.30s CPU      0.40s WALL"
_WALL_RE = re.compile(r"PWSCF\s*:\s*[\d.]+s\s+CPU\s+([\d.]+)s\s+WALL", re.IGNORECASE)

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
    wall = _maybe_float(_WALL_RE, text)
    errors = _collect_errors(text)
    nat = _maybe_int(_NAT_RE, text)
    ntyp = _maybe_int(_NTYP_RE, text)
    nelec = _maybe_float(_NELEC_RE, text)

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
    if _SCF_UNCONV_RE.search(text):
        return ConvergenceStatus.UNCONVERGED
    if _SCF_CONVERGED_RE.search(text):
        return ConvergenceStatus.CONVERGED
    if _ERROR_BANNER_RE.search(text):
        return ConvergenceStatus.ERRORED
    return ConvergenceStatus.UNKNOWN


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

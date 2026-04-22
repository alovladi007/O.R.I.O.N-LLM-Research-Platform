"""Pseudopotential registry — scan a SSSP directory, expose per-element metadata.

One instance covers one pseudopotential library on disk. For ORION's
Phase 3 default that's ``~/orion/pseudos/SSSP_1.3.0_PBE_efficiency/``.

Each UPF file's header carries ``wfc_cutoff`` and ``rho_cutoff``
attributes as strings (scientific notation, Ry). Registry parses these
at build time and caches them — we read every UPF once at startup.

Filename → element mapping
--------------------------

SSSP's filenames follow no single convention. We parse the element by
splitting on `.` or `_` and taking the first alphabetic token that's
a known periodic-table symbol, in capitalization-preserving order.
Example filenames and the element we extract:

    Si.pbe-n-rrkjus_psl.1.0.0.UPF      → Si
    Ag_ONCV_PBE-1.0.oncvpsp.upf        → Ag
    Fe.pbe-spn-kjpaw_psl.0.2.1.UPF     → Fe
    Bi_pbe_v1.uspp.F.UPF               → Bi
    Ce.paw.z_12.atompaw.wentzcovitch.v1.2.upf → Ce
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# Atomic masses (amu) for ATOMIC_SPECIES block. Subset that covers
# what SSSP ships. We don't need all 118 — elements not in the
# registry raise UnknownElementError anyway.
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "Be": 9.012, "B": 10.811,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.086, "P": 30.974,
    "S": 32.065, "Cl": 35.453, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.64, "As": 74.922, "Se": 78.96, "Br": 79.904,
    "Kr": 83.798, "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224,
    "Nb": 92.906, "Mo": 95.94, "Tc": 98.0, "Ru": 101.07, "Rh": 102.906,
    "Pd": 106.42, "Ag": 107.868, "Cd": 112.411, "In": 114.818, "Sn": 118.710,
    "Sb": 121.760, "Te": 127.60, "I": 126.904, "Xe": 131.293, "Cs": 132.905,
    "Ba": 137.327, "La": 138.905, "Ce": 140.116, "Pr": 140.908, "Nd": 144.242,
    "Pm": 145.0, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.925,
    "Dy": 162.500, "Ho": 164.930, "Er": 167.259, "Tm": 168.934, "Yb": 173.04,
    "Lu": 174.967, "Hf": 178.49, "Ta": 180.948, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Au": 196.967, "Hg": 200.59,
    "Tl": 204.383, "Pb": 207.2, "Bi": 208.980, "Po": 209.0, "At": 210.0,
    "Rn": 222.0, "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.038,
    "Pa": 231.036, "U": 238.029, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
    "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0, "Fm": 257.0,
}

# All valid periodic-table symbols, longest-first for greedy matching.
_VALID_ELEMENTS = sorted(ATOMIC_MASSES.keys(), key=len, reverse=True)

_WFC_RE = re.compile(r'wfc_cutoff="([^"]+)"')
_RHO_RE = re.compile(r'rho_cutoff="([^"]+)"')
_SUGGEST_WFC_RE = re.compile(
    r"Suggested\s+minimum\s+cutoff\s+for\s+wavefunctions:\s*([\d.]+)", re.IGNORECASE,
)
_SUGGEST_RHO_RE = re.compile(
    r"Suggested\s+minimum\s+cutoff\s+for\s+charge\s+density:\s*([\d.]+)", re.IGNORECASE,
)


class UnknownElementError(KeyError):
    """Element has no matching UPF in the registry."""


class UPFFileNotFoundError(FileNotFoundError):
    """Registry pointed at a dir that doesn't exist or is empty."""


@dataclass
class PseudoEntry:
    """One element's pseudopotential + recommended cutoffs."""

    element: str
    filename: str        # bare filename, not a full path
    mass: float          # amu, from ATOMIC_MASSES
    ecutwfc_ry: Optional[float] = None  # from UPF header or docstring
    ecutrho_ry: Optional[float] = None


def _element_from_filename(filename: str) -> Optional[str]:
    """Split on ``.`` and ``_`` and return the first token that's an element.

    Returns ``None`` if no token matches. Tries exact-case match first
    (canonical pslibrary convention), then title-case (GBRV-style
    ``na_pbe_v1.4.uspp.F.UPF``). We don't try upper-case because that
    would match absurd tokens like "A" in filenames.
    """
    base = filename.rsplit(".", 1)[0] if filename.lower().endswith((".upf", ".upf2")) else filename
    tokens = re.split(r"[._]", base)
    for tok in tokens:
        # Exact match (Si, Fe, etc.)
        if tok in _VALID_ELEMENTS:
            return tok
        # Title-case variant for lowercase filenames (na, cl, fe).
        candidate = tok.capitalize() if tok else ""
        if candidate and candidate in _VALID_ELEMENTS and len(tok) <= 2:
            return candidate
    return None


def _parse_cutoffs(upf_path: Path) -> tuple[Optional[float], Optional[float]]:
    """Return (ecutwfc_ry, ecutrho_ry) parsed from a UPF file's header.

    Looks at the first 100 lines — UPF v2 puts cutoffs in an XML
    attribute near the top; UPF v1 puts them in the human-readable
    ``Suggested minimum cutoff ...`` lines. We try v2 first and fall
    back to v1.
    """
    try:
        with upf_path.open("r", encoding="utf-8", errors="replace") as f:
            head_lines = []
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                head_lines.append(line)
            head = "".join(head_lines)
    except OSError:
        return None, None

    m_wfc = _WFC_RE.search(head)
    m_rho = _RHO_RE.search(head)
    if m_wfc and m_rho:
        try:
            return float(m_wfc.group(1)), float(m_rho.group(1))
        except ValueError:
            pass

    m_wfc = _SUGGEST_WFC_RE.search(head)
    m_rho = _SUGGEST_RHO_RE.search(head)
    wfc = float(m_wfc.group(1)) if m_wfc else None
    rho = float(m_rho.group(1)) if m_rho else None
    return wfc, rho


class PseudopotentialRegistry:
    """Element → PseudoEntry mapping backed by one directory of UPF files."""

    def __init__(self, pseudo_dir: Path | str):
        self.pseudo_dir = Path(pseudo_dir).expanduser()
        if not self.pseudo_dir.is_dir():
            raise UPFFileNotFoundError(
                f"pseudo_dir {self.pseudo_dir} is not a directory"
            )
        self._entries: Dict[str, PseudoEntry] = {}
        self._scan()
        if not self._entries:
            raise UPFFileNotFoundError(
                f"no UPF files found under {self.pseudo_dir}"
            )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def __contains__(self, element: str) -> bool:
        return element in self._entries

    def __iter__(self):
        return iter(self._entries)

    def get(self, element: str) -> PseudoEntry:
        try:
            return self._entries[element]
        except KeyError as exc:
            available = ", ".join(sorted(self._entries))
            raise UnknownElementError(
                f"no pseudopotential for element {element!r}. "
                f"Available: {available}"
            ) from exc

    def elements(self) -> List[str]:
        return sorted(self._entries)

    def recommended_ecutwfc(self, elements: Iterable[str]) -> float:
        """Return ``max(wfc_cutoff)`` across the requested elements.

        Elements without a parsed cutoff contribute the SSSP-wide
        default of 40 Ry (conservative). Elements not in the registry
        raise :class:`UnknownElementError`.
        """
        elements = list(elements)
        if not elements:
            raise ValueError("recommended_ecutwfc: empty element list")
        values: List[float] = []
        for e in elements:
            entry = self.get(e)
            values.append(entry.ecutwfc_ry if entry.ecutwfc_ry else 40.0)
        return max(values)

    def recommended_ecutrho(self, elements: Iterable[str]) -> float:
        elements = list(elements)
        if not elements:
            raise ValueError("recommended_ecutrho: empty element list")
        values: List[float] = []
        for e in elements:
            entry = self.get(e)
            values.append(entry.ecutrho_ry if entry.ecutrho_ry else 320.0)
        return max(values)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _scan(self) -> None:
        for path in sorted(self.pseudo_dir.iterdir()):
            if path.name.startswith("."):
                continue
            if not path.is_file():
                continue
            if not path.name.lower().endswith((".upf", ".upf2")):
                continue
            element = _element_from_filename(path.name)
            if element is None:
                logger.debug("skip unmatched upf %s", path.name)
                continue
            # If multiple UPFs for one element, keep the first seen
            # (sorted iteration ⇒ reproducible choice).
            if element in self._entries:
                continue
            wfc, rho = _parse_cutoffs(path)
            mass = ATOMIC_MASSES.get(element, 0.0)
            self._entries[element] = PseudoEntry(
                element=element,
                filename=path.name,
                mass=mass,
                ecutwfc_ry=wfc,
                ecutrho_ry=rho,
            )

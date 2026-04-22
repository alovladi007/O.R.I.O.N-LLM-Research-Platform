"""Elemental reference crystal builder.

One ground-state prototype per element. These numbers come from
experimental / well-converged DFT values in the literature — they're
starting points for vc-relax. Calibration re-relaxes them at ORION's
functional + pseudo family, and *that* relaxed energy is what lands
in the reference_energies table.

Covered prototypes (phase 3.4 scope)
------------------------------------

- **diamond_cubic** — Si, Ge, C (graphite has lower ground-state
  energy for carbon, but diamond is the usual PBE reference).
- **fcc** — Al, Cu, Ag, Au, Ni, Pb, Pt, Pd, Sr, Ca.
- **bcc** — Fe (at 0 K PBE prefers FCC, but BCC is experimental), Li,
  Na, K, Rb, Cs, Cr, V, Mo, W.
- **hcp** — Mg, Ti, Zn, Co, Zr, Ru.
- **rocksalt** — (reserved for compound refs later — not an
  elemental prototype).

Not covered here
----------------

Diatomic gases (H2, N2, O2, F2, Cl2): these need molecule-in-vacuum
calculations with a large box + Γ-only sampling, which is a different
workflow than the bulk vc-relax we have. The calibration CLI errors
for these elements and points the user at Session 3.4b.

Other complex ground states (α-Mn, graphite for C, α-S8): not
covered this session. The rule: if it's not cubic with a small cell,
we either use a close-enough prototype or skip it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

Prototype = Literal[
    "diamond_cubic", "fcc", "bcc", "hcp", "simple_cubic",
]


class UnsupportedElement(ValueError):
    """Raised for elements whose ground state ORION's calibration can't handle yet."""


# Diatomic gases — need molecule-in-vacuum, not bulk. Refuse for now.
DIATOMIC_GAS_ELEMENTS = frozenset({"H", "N", "O", "F", "Cl", "Br", "I"})

# Elements that work only as monoatomic solids (noble gases) — also
# different workflow (Van der Waals matters). Refuse for now.
ATOMIC_SOLID_ONLY = frozenset({"He", "Ne", "Ar", "Kr", "Xe", "Rn"})


# Experimental lattice constants (Å) for common elements. These seed
# the vc-relax; the relaxed value is what gets stored.
_LATTICE_CONSTANTS_ANG: Dict[str, float] = {
    # diamond-cubic
    "C": 3.567, "Si": 5.431, "Ge": 5.658,
    # fcc
    "Al": 4.046, "Cu": 3.615, "Ag": 4.085, "Au": 4.078,
    "Ni": 3.524, "Pb": 4.950, "Pt": 3.924, "Pd": 3.891,
    "Sr": 6.084, "Ca": 5.588,
    # bcc
    "Fe": 2.866, "Li": 3.491, "Na": 4.290, "K": 5.328,
    "Rb": 5.585, "Cs": 6.141, "Cr": 2.880, "V": 3.024,
    "Mo": 3.147, "W": 3.165,
    # hcp (c/a ≈ 1.633 ideal); we encode as (a, c) pair below
    "Mg": 3.209, "Ti": 2.951, "Zn": 2.665, "Co": 2.507,
    "Zr": 3.232, "Ru": 2.706,
}

# c-axis for HCP elements. Ideal c/a = sqrt(8/3) ≈ 1.633.
_HCP_C_AXIS: Dict[str, float] = {
    "Mg": 5.211, "Ti": 4.686, "Zn": 4.947, "Co": 4.070,
    "Zr": 5.147, "Ru": 4.282,
}


_PROTOTYPE_FOR_ELEMENT: Dict[str, Prototype] = {
    # diamond-cubic
    "C": "diamond_cubic", "Si": "diamond_cubic", "Ge": "diamond_cubic",
    # fcc
    "Al": "fcc", "Cu": "fcc", "Ag": "fcc", "Au": "fcc", "Ni": "fcc",
    "Pb": "fcc", "Pt": "fcc", "Pd": "fcc", "Sr": "fcc", "Ca": "fcc",
    # bcc
    "Fe": "bcc", "Li": "bcc", "Na": "bcc", "K": "bcc", "Rb": "bcc",
    "Cs": "bcc", "Cr": "bcc", "V": "bcc", "Mo": "bcc", "W": "bcc",
    # hcp
    "Mg": "hcp", "Ti": "hcp", "Zn": "hcp", "Co": "hcp", "Zr": "hcp",
    "Ru": "hcp",
}


@dataclass(frozen=True)
class ElementalReferenceCell:
    """A concrete elemental reference cell ready to feed into the QE input gen."""

    element: str
    prototype: Prototype
    lattice_ang: List[List[float]]
    species: List[str]
    frac_coords: List[List[float]]
    n_atoms: int
    # Convenience: what's the conventional lattice constant? For
    # diamond_cubic and fcc this is the conventional cube edge. For
    # bcc same. For hcp we encode the a-axis.
    a_conv_ang: float

    def as_qe_struct(self) -> Dict[str, Any]:
        """Shape expected by backend.common.engines.qe_input.generate_pw_input."""
        return {
            "lattice": [list(row) for row in self.lattice_ang],
            "species": list(self.species),
            "frac_coords": [list(c) for c in self.frac_coords],
        }


def _fcc_primitive_lattice(a: float) -> List[List[float]]:
    """Primitive FCC cell: a = a_conv / 2 * (1,1,0) etc."""
    half = a / 2.0
    return [
        [0.0, half, half],
        [half, 0.0, half],
        [half, half, 0.0],
    ]


def _bcc_primitive_lattice(a: float) -> List[List[float]]:
    """Primitive BCC cell: a = a_conv / 2 * (-1,1,1) etc."""
    half = a / 2.0
    return [
        [-half, half, half],
        [half, -half, half],
        [half, half, -half],
    ]


def _hcp_lattice(a: float, c: float) -> List[List[float]]:
    """HCP conventional cell (2 atoms)."""
    return [
        [a, 0.0, 0.0],
        [-a / 2.0, a * math.sqrt(3) / 2.0, 0.0],
        [0.0, 0.0, c],
    ]


def build_elemental_reference_cell(
    element: str,
    *,
    a_override: Optional[float] = None,
) -> ElementalReferenceCell:
    """Return a starter cell for *element*'s ground-state prototype.

    Parameters
    ----------
    element
        Periodic-table symbol. Case-sensitive (``Si``, not ``si``).
    a_override
        Optional lattice-constant override in Å. When ``None``, uses
        the experimental value from ``_LATTICE_CONSTANTS_ANG``.

    Raises
    ------
    UnsupportedElement
        For diatomic gases, noble gases, and anything not in
        ``_PROTOTYPE_FOR_ELEMENT``.
    """
    if element in DIATOMIC_GAS_ELEMENTS:
        raise UnsupportedElement(
            f"Element {element!r} is a diatomic gas at STP — its reference "
            "energy requires a molecule-in-vacuum calculation, which ORION "
            "doesn't support yet (Session 3.4b). Until then, formation "
            "energies for compounds containing this element can't be "
            "computed directly. Workaround: supply a hand-computed "
            "reference via the API or CLI."
        )
    if element in ATOMIC_SOLID_ONLY:
        raise UnsupportedElement(
            f"Element {element!r} is a noble-gas atomic solid — binding is "
            "Van der Waals, which PBE doesn't capture. Use DFT-D3 or a "
            "stored literature value."
        )

    prototype = _PROTOTYPE_FOR_ELEMENT.get(element)
    if prototype is None:
        raise UnsupportedElement(
            f"No ground-state prototype registered for element {element!r}. "
            f"Add to backend.common.calibration.references if this is needed."
        )
    a = a_override if a_override is not None else _LATTICE_CONSTANTS_ANG[element]

    if prototype == "diamond_cubic":
        # Primitive FCC with 2-atom basis (standard diamond primitive).
        lattice = _fcc_primitive_lattice(a)
        species = [element, element]
        frac_coords = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
        n_atoms = 2
    elif prototype == "fcc":
        lattice = _fcc_primitive_lattice(a)
        species = [element]
        frac_coords = [[0.0, 0.0, 0.0]]
        n_atoms = 1
    elif prototype == "bcc":
        lattice = _bcc_primitive_lattice(a)
        species = [element]
        frac_coords = [[0.0, 0.0, 0.0]]
        n_atoms = 1
    elif prototype == "hcp":
        c = _HCP_C_AXIS[element]
        lattice = _hcp_lattice(a, c)
        species = [element, element]
        # Standard HCP fractional positions.
        frac_coords = [[0.0, 0.0, 0.0], [1 / 3, 2 / 3, 0.5]]
        n_atoms = 2
    else:  # pragma: no cover — simple_cubic reserved
        raise UnsupportedElement(f"Prototype {prototype} not yet implemented.")

    return ElementalReferenceCell(
        element=element,
        prototype=prototype,
        lattice_ang=lattice,
        species=species,
        frac_coords=frac_coords,
        n_atoms=n_atoms,
        a_conv_ang=a,
    )


def supported_elements() -> List[str]:
    """Alphabetised list of elements the builder can produce a cell for."""
    return sorted(_PROTOTYPE_FOR_ELEMENT)

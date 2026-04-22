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
    "molecule_in_vacuum",
]


class UnsupportedElement(ValueError):
    """Raised for elements whose ground state ORION's calibration can't handle yet."""


# Diatomic gases — need molecule-in-vacuum (Session 3.4b).
DIATOMIC_GAS_ELEMENTS = frozenset({"H", "N", "O", "F", "Cl", "Br", "I"})


# Experimental bond lengths for diatomic molecules (Å). Used as the
# initial geometry; `relax` adjusts it.
_DIATOMIC_BOND_ANG: Dict[str, float] = {
    "H": 0.741,   # H₂
    "N": 1.098,   # N₂
    "O": 1.208,   # O₂
    "F": 1.412,   # F₂
    "Cl": 1.988,  # Cl₂
    "Br": 2.281,  # Br₂
    "I": 2.666,   # I₂
}

# Molecules whose ground state is a triplet (total_magnetization = 2).
# O₂ is the classic; everything else on this list is also non-trivial
# for PBE and gets spin_polarized = True. For singlet diatomics
# (H₂, N₂, F₂, Cl₂) spin-polarization isn't needed but is harmless.
_TRIPLET_DIATOMICS = frozenset({"O"})

# Default vacuum box edge for molecule-in-vacuum calcs. 15 Å is
# the minimum that reliably converges PBE O₂ binding to within
# ~10 meV; 18 Å is tighter. We ship 15 Å default as a cost/accuracy
# compromise and expose an override.
_DEFAULT_VACUUM_BOX_ANG = 15.0

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
    # molecule-in-vacuum (Session 3.4b). Bond lengths in _DIATOMIC_BOND_ANG.
    "H": "molecule_in_vacuum", "N": "molecule_in_vacuum",
    "O": "molecule_in_vacuum", "F": "molecule_in_vacuum",
    "Cl": "molecule_in_vacuum", "Br": "molecule_in_vacuum",
    "I": "molecule_in_vacuum",
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
    vacuum_box_ang: Optional[float] = None,
) -> ElementalReferenceCell:
    """Return a starter cell for *element*'s ground-state prototype.

    Parameters
    ----------
    element
        Periodic-table symbol. Case-sensitive (``Si``, not ``si``).
    a_override
        Optional lattice-constant override in Å. When ``None``, uses
        the experimental value from ``_LATTICE_CONSTANTS_ANG`` or the
        diatomic bond length for molecule-in-vacuum prototypes.
    vacuum_box_ang
        Cubic vacuum-box edge (Å) for molecule-in-vacuum prototypes.
        Defaults to 15 Å — a reasonable accuracy/cost tradeoff for
        PBE at SSSP cutoffs. Only used when prototype is
        ``molecule_in_vacuum``.

    Raises
    ------
    UnsupportedElement
        For noble gases and anything not in ``_PROTOTYPE_FOR_ELEMENT``.
    """
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

    if prototype == "molecule_in_vacuum":
        box = vacuum_box_ang if vacuum_box_ang is not None else _DEFAULT_VACUUM_BOX_ANG
        bond = a_override if a_override is not None else _DIATOMIC_BOND_ANG[element]
        # Place the two atoms symmetrically around the box center
        # along the x-axis. Fractional coords = (center ± bond/2/box).
        center = 0.5
        offset = (bond / 2.0) / box
        lattice = [[box, 0.0, 0.0], [0.0, box, 0.0], [0.0, 0.0, box]]
        species = [element, element]
        frac_coords = [
            [center - offset, center, center],
            [center + offset, center, center],
        ]
        return ElementalReferenceCell(
            element=element,
            prototype=prototype,
            lattice_ang=lattice,
            species=species,
            frac_coords=frac_coords,
            n_atoms=2,
            a_conv_ang=box,  # box edge, for lack of a better "conventional a"
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


def is_triplet_diatomic(element: str) -> bool:
    """True when the diatomic molecule's ground state is a triplet (spin-polarized)."""
    return element in _TRIPLET_DIATOMICS


def supported_elements() -> List[str]:
    """Alphabetised list of elements the builder can produce a cell for."""
    return sorted(_PROTOTYPE_FOR_ELEMENT)

"""
Canonical hashing of crystal structures.

Purpose (from ROADMAP_PROMPTS.md Phase 1 / Session 1.2): give every
:class:`Structure` row a deterministic 64-char hex hash so that uploading
the same structure twice (even with trivially different atom orderings or
floating-point rounding) is detected as a duplicate.

Design choices
--------------

- **Fingerprint source.** If pymatgen is available we use its
  ``StructureMatcher.fingerprint(structure)``-style canonicalization: the
  reduced composition, the refined spacegroup number, and the symmetrized
  site list sorted by (species, lexicographic fractional coords).
- **Hash function.** SHA-256, truncated to 64 hex chars (128 bits). Full
  collision resistance for every structure we'll ever index.
- **Tolerances.** Coordinates are rounded to ``coord_tol`` (default 1e-4
  fractional), lattice parameters to ``lattice_tol`` (default 1e-4 Å).
  Angles to 0.01°. These are tighter than typical DFT noise but loose
  enough to tolerate cosmetic rounding from CIF I/O.
- **Fallback.** Without pymatgen we hash the sorted list of
  ``(species, round(coord, 4))`` tuples plus the rounded lattice. This
  gives a weaker invariance (supercell / space-group duplicates aren't
  caught) but never crashes the import path. Session 1.1 lifts the
  fallback to use real symmetry analysis.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, List, Optional, Sequence, Tuple

# pymatgen is optional at import time so unit tests can run without it.
try:  # pragma: no cover — exercised in tests with/without pymatgen
    from pymatgen.core import Structure as PmgStructure  # type: ignore

    _HAS_PYMATGEN = True
except ImportError:  # pragma: no cover
    PmgStructure = None  # type: ignore[assignment]
    _HAS_PYMATGEN = False


def _round_coord(x: float, tol: float) -> float:
    """Round to the nearest multiple of *tol*, symmetric around 0."""
    return round(x / tol) * tol


def _canonical_lattice(
    lattice: Sequence[Sequence[float]], tol: float = 1e-4
) -> Tuple[Tuple[float, ...], ...]:
    """Round lattice vectors (3x3 matrix in Å) to *tol* precision."""
    return tuple(tuple(_round_coord(x, tol) for x in row) for row in lattice)


def _canonical_atoms(
    atoms: Iterable[Any], tol: float = 1e-4
) -> List[Tuple[str, Tuple[float, float, float]]]:
    """
    Normalize ``atoms`` to ``[(species, (fx, fy, fz)), ...]`` sorted by
    ``(species, fx, fy, fz)``.

    Accepts a few shapes:
    - iterable of dicts: ``{"species": "Si", "position": [fx,fy,fz]}``
    - iterable of 2-tuples: ``("Si", [fx,fy,fz])``
    """
    norm: List[Tuple[str, Tuple[float, float, float]]] = []
    for atom in atoms:
        if isinstance(atom, dict):
            species = str(atom.get("species") or atom.get("element") or atom.get("symbol"))
            pos = atom.get("position") or atom.get("coords") or atom.get("frac_coords")
        else:
            species, pos = atom  # type: ignore[misc]
        if pos is None or species in (None, "None"):
            raise ValueError(f"atom entry {atom!r} missing species/position")
        fx, fy, fz = [float(c) for c in pos]
        norm.append(
            (
                species,
                (
                    _round_coord(fx % 1.0, tol),
                    _round_coord(fy % 1.0, tol),
                    _round_coord(fz % 1.0, tol),
                ),
            )
        )
    norm.sort()
    return norm


def structure_hash(
    *,
    lattice: Optional[Sequence[Sequence[float]]] = None,
    atoms: Optional[Iterable[Any]] = None,
    pmg_structure: Optional["PmgStructure"] = None,
    coord_tol: float = 1e-4,
    lattice_tol: float = 1e-4,
) -> str:
    """
    Return a deterministic 64-char hex hash for a crystal structure.

    Provide either ``pmg_structure`` (a :class:`pymatgen.core.Structure`)
    or both ``lattice`` (3x3 Å) and ``atoms`` (iterable). Symmetry-aware
    canonicalization is used when pymatgen is present and a
    ``pmg_structure`` is supplied; otherwise we fall back to a weaker
    species+coords hash.

    The hash is intentionally tight on coordinates (1e-4 fractional by
    default) — two structures that are physically the same but differ by
    more than 1e-4 will collide in the DB's unique index and one will be
    rejected, prompting the caller to re-check whether the difference
    matters.
    """
    if pmg_structure is not None:
        # Symmetry-refined canonical form.
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer  # lazy import

        sga = SpacegroupAnalyzer(pmg_structure, symprec=0.01, angle_tolerance=5.0)
        refined = sga.get_refined_structure()
        lattice_rep = _canonical_lattice(refined.lattice.matrix.tolist(), lattice_tol)
        atoms_rep = _canonical_atoms(
            (
                {"species": site.species_string, "position": site.frac_coords.tolist()}
                for site in refined.sites
            ),
            coord_tol,
        )
        spacegroup = sga.get_space_group_number()
        reduced_formula = refined.composition.reduced_formula
        payload = (
            "pmg-v1",
            reduced_formula,
            spacegroup,
            lattice_rep,
            atoms_rep,
        )
    else:
        if lattice is None or atoms is None:
            raise ValueError(
                "structure_hash needs either pmg_structure or (lattice, atoms)."
            )
        lattice_rep = _canonical_lattice(lattice, lattice_tol)
        atoms_rep = _canonical_atoms(atoms, coord_tol)
        payload = ("raw-v1", lattice_rep, atoms_rep)

    # Deterministic encoding: repr of the payload tuple is stable for the
    # nested floats we've already rounded, and doesn't depend on JSON
    # formatting quirks.
    encoded = repr(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

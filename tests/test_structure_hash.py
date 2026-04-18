"""
Tests for backend.common.structures.hashing.

The invariant contract (Session 1.2b):

- Atom-order shouldn't change the hash.
- Sub-tolerance perturbations (default 1e-4 fractional) shouldn't change it.
- Super-tolerance perturbations SHOULD change it.
- Different species SHOULD differ.
- Different lattice SHOULD differ.

Pymatgen-path tests are marked and skipped when pymatgen is absent.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def shash():
    from backend.common.structures.hashing import structure_hash
    return structure_hash


@pytest.fixture
def si_lattice():
    return [[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]]


@pytest.fixture
def si_atoms():
    return [
        ("Si", [0.0, 0.0, 0.0]),
        ("Si", [0.25, 0.25, 0.25]),
    ]


class TestRawHash:
    def test_reorder_invariance(self, shash, si_lattice, si_atoms):
        h1 = shash(lattice=si_lattice, atoms=si_atoms)
        h2 = shash(lattice=si_lattice, atoms=list(reversed(si_atoms)))
        assert h1 == h2

    def test_sub_tolerance_invariance(self, shash, si_lattice, si_atoms):
        """Perturb one coord by 1e-5 (below default 1e-4 tol) — hash holds."""
        h1 = shash(lattice=si_lattice, atoms=si_atoms)
        atoms_perturbed = [
            ("Si", [0.00001, 0.0, 0.0]),
            ("Si", [0.25, 0.25, 0.25]),
        ]
        h2 = shash(lattice=si_lattice, atoms=atoms_perturbed)
        assert h1 == h2

    def test_super_tolerance_differs(self, shash, si_lattice, si_atoms):
        """Perturb by 1e-3 (10x tol) — hash must change."""
        h1 = shash(lattice=si_lattice, atoms=si_atoms)
        atoms_perturbed = [
            ("Si", [0.001, 0.0, 0.0]),
            ("Si", [0.25, 0.25, 0.25]),
        ]
        h2 = shash(lattice=si_lattice, atoms=atoms_perturbed)
        assert h1 != h2

    def test_species_differs(self, shash, si_lattice, si_atoms):
        h1 = shash(lattice=si_lattice, atoms=si_atoms)
        ge_atoms = [
            ("Ge", [0.0, 0.0, 0.0]),
            ("Ge", [0.25, 0.25, 0.25]),
        ]
        h2 = shash(lattice=si_lattice, atoms=ge_atoms)
        assert h1 != h2

    def test_lattice_differs(self, shash, si_atoms):
        small = [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]]
        big = [[5.44, 0, 0], [0, 5.44, 0], [0, 0, 5.44]]
        assert shash(lattice=small, atoms=si_atoms) != shash(lattice=big, atoms=si_atoms)

    def test_hash_shape(self, shash, si_lattice, si_atoms):
        h = shash(lattice=si_lattice, atoms=si_atoms)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_needs_lattice_or_pmg(self, shash, si_atoms):
        with pytest.raises(ValueError):
            shash(atoms=si_atoms)  # no lattice, no pmg_structure


class TestDictAtomShape:
    def test_accepts_dict_with_position(self, shash, si_lattice):
        atoms = [
            {"species": "Si", "position": [0.0, 0.0, 0.0]},
            {"species": "Si", "position": [0.25, 0.25, 0.25]},
        ]
        h = shash(lattice=si_lattice, atoms=atoms)
        assert len(h) == 64

    def test_dict_and_tuple_give_same_hash(self, shash, si_lattice):
        atoms_tup = [("Si", [0.0, 0.0, 0.0]), ("Si", [0.25, 0.25, 0.25])]
        atoms_dict = [
            {"species": "Si", "position": [0.0, 0.0, 0.0]},
            {"species": "Si", "position": [0.25, 0.25, 0.25]},
        ]
        assert shash(lattice=si_lattice, atoms=atoms_tup) == shash(
            lattice=si_lattice, atoms=atoms_dict
        )


class TestPymatgenPath:
    """Real symmetry-aware hashing via pymatgen."""

    def test_pmg_hash_for_silicon(self, shash):
        pymatgen = pytest.importorskip("pymatgen")
        from pymatgen.core import Lattice, Structure

        lattice = Lattice.cubic(5.43)
        s = Structure(
            lattice,
            ["Si", "Si"],
            [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        )
        h = shash(pmg_structure=s)
        assert len(h) == 64

"""
Comprehensive tests for the structure parsing module.

Tests cover:
- Parsing of CIF, POSCAR, and XYZ formats
- Conversion between formats
- Error handling and validation
- Edge cases and malformed inputs
"""

import pytest
from backend.common.structures import (
    parse_structure,
    to_cif,
    to_poscar,
    to_xyz,
    InternalStructureModel,
    StructureFormat,
)
from src.api.exceptions import ParsingError


# ========== Test Data ==========


# Silicon diamond structure in CIF format
SILICON_CIF = """data_Si
_cell_length_a    5.46873
_cell_length_b    5.46873
_cell_length_c    5.46873
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'F d -3 m'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 0.00000 0.00000 0.00000
Si2 0.25000 0.25000 0.25000
"""

# Silicon diamond structure in POSCAR format
SILICON_POSCAR = """Si2
1.0
   5.46873000   0.00000000   0.00000000
   0.00000000   5.46873000   0.00000000
   0.00000000   0.00000000   5.46873000
Si
2
direct
   0.00000000   0.00000000   0.00000000
   0.25000000   0.25000000   0.25000000
"""

# Water molecule in XYZ format
WATER_XYZ = """3
Water molecule
O   0.000000   0.000000   0.119262
H   0.000000   0.763239  -0.477047
H   0.000000  -0.763239  -0.477047
"""

# Graphene sheet in CIF format (2D structure)
GRAPHENE_CIF = """data_graphene
_cell_length_a    2.46
_cell_length_b    2.46
_cell_length_c    20.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 120.0
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 0.0 0.0 0.5
C2 0.333333 0.666667 0.5
"""


# ========== Parsing Tests ==========


@pytest.mark.skipif(
    "not config.getoption('--run-pymatgen')",
    reason="Requires pymatgen installation"
)
class TestStructureParsing:
    """Test structure parsing from different formats."""

    def test_parse_cif_silicon(self):
        """Test parsing Silicon structure from CIF."""
        structure = parse_structure(SILICON_CIF, StructureFormat.CIF)

        assert structure.formula == "Si"
        assert structure.num_atoms == 2
        assert structure.dimensionality == 3
        assert structure.a == pytest.approx(5.46873, rel=1e-4)
        assert structure.b == pytest.approx(5.46873, rel=1e-4)
        assert structure.c == pytest.approx(5.46873, rel=1e-4)
        assert structure.alpha == pytest.approx(90.0, rel=1e-4)
        assert structure.beta == pytest.approx(90.0, rel=1e-4)
        assert structure.gamma == pytest.approx(90.0, rel=1e-4)
        assert len(structure.atomic_species) == 2
        assert all(species == "Si" for species in structure.atomic_species)
        assert len(structure.lattice_vectors) == 3
        assert len(structure.atomic_positions) == 2

    def test_parse_poscar_silicon(self):
        """Test parsing Silicon structure from POSCAR."""
        structure = parse_structure(SILICON_POSCAR, StructureFormat.POSCAR)

        assert structure.formula == "Si"
        assert structure.num_atoms == 2
        assert structure.a == pytest.approx(5.46873, rel=1e-4)
        assert len(structure.atomic_species) == 2
        assert all(species == "Si" for species in structure.atomic_species)

    def test_parse_xyz_water(self):
        """Test parsing water molecule from XYZ."""
        structure = parse_structure(WATER_XYZ, StructureFormat.XYZ)

        assert structure.formula in ["H2O", "OH2"]  # Formula order may vary
        assert structure.num_atoms == 3
        assert structure.dimensionality == 0  # Molecule
        assert len(structure.atomic_species) == 3
        # Check we have O and H atoms
        assert "O" in structure.atomic_species
        assert "H" in structure.atomic_species

    def test_parse_graphene_2d(self):
        """Test parsing 2D graphene structure."""
        structure = parse_structure(GRAPHENE_CIF, StructureFormat.CIF)

        assert structure.formula == "C"
        assert structure.num_atoms == 2
        # Should detect as 2D due to large c parameter with vacuum
        assert structure.dimensionality == 2
        assert structure.c > 15.0  # Has vacuum in c direction

    def test_parse_empty_input(self):
        """Test that empty input raises appropriate error."""
        with pytest.raises(ParsingError) as exc_info:
            parse_structure("", StructureFormat.CIF)

        assert "empty" in exc_info.value.message.lower()

    def test_parse_whitespace_only(self):
        """Test that whitespace-only input raises appropriate error."""
        with pytest.raises(ParsingError) as exc_info:
            parse_structure("   \n  \n  ", StructureFormat.CIF)

        assert "empty" in exc_info.value.message.lower()

    def test_parse_malformed_cif(self):
        """Test parsing malformed CIF file."""
        malformed_cif = """data_bad
_cell_length_a    5.0
This is not valid CIF syntax
"""
        with pytest.raises(ParsingError) as exc_info:
            parse_structure(malformed_cif, StructureFormat.CIF)

        assert exc_info.value.details["format"] == "CIF"

    def test_parse_malformed_poscar(self):
        """Test parsing malformed POSCAR file."""
        malformed_poscar = """Short POSCAR
1.0
Too few lines"""
        with pytest.raises(ParsingError) as exc_info:
            parse_structure(malformed_poscar, StructureFormat.POSCAR)

        assert exc_info.value.details["format"] == "POSCAR"

    def test_parse_malformed_xyz(self):
        """Test parsing malformed XYZ file."""
        malformed_xyz = """3
Comment
Only one atom here
"""
        with pytest.raises(ParsingError) as exc_info:
            parse_structure(malformed_xyz, StructureFormat.XYZ)

        assert exc_info.value.details["format"] == "XYZ"


# ========== Export Tests ==========


@pytest.mark.skipif(
    "not config.getoption('--run-pymatgen')",
    reason="Requires pymatgen installation"
)
class TestStructureExport:
    """Test exporting structures to different formats."""

    def test_export_to_cif(self):
        """Test exporting structure to CIF format."""
        structure = parse_structure(SILICON_POSCAR, StructureFormat.POSCAR)
        cif_output = to_cif(structure)

        # Verify it's valid CIF by parsing it back
        reparsed = parse_structure(cif_output, StructureFormat.CIF)
        assert reparsed.formula == structure.formula
        assert reparsed.num_atoms == structure.num_atoms

    def test_export_to_poscar(self):
        """Test exporting structure to POSCAR format."""
        structure = parse_structure(SILICON_CIF, StructureFormat.CIF)
        poscar_output = to_poscar(structure)

        # Verify it's valid POSCAR by parsing it back
        reparsed = parse_structure(poscar_output, StructureFormat.POSCAR)
        assert reparsed.formula == structure.formula
        assert reparsed.num_atoms == structure.num_atoms

    def test_export_to_xyz(self):
        """Test exporting structure to XYZ format."""
        structure = parse_structure(SILICON_CIF, StructureFormat.CIF)
        xyz_output = to_xyz(structure)

        # Verify it's valid XYZ by parsing it back
        reparsed = parse_structure(xyz_output, StructureFormat.XYZ)
        assert reparsed.formula == structure.formula
        assert reparsed.num_atoms == structure.num_atoms


# ========== Conversion Tests ==========


@pytest.mark.skipif(
    "not config.getoption('--run-pymatgen')",
    reason="Requires pymatgen installation"
)
class TestFormatConversion:
    """Test conversion between different formats."""

    def test_cif_to_poscar_to_cif(self):
        """Test round-trip conversion CIF -> POSCAR -> CIF."""
        original = parse_structure(SILICON_CIF, StructureFormat.CIF)
        poscar = to_poscar(original)
        reparsed = parse_structure(poscar, StructureFormat.POSCAR)

        assert reparsed.formula == original.formula
        assert reparsed.num_atoms == original.num_atoms
        assert reparsed.a == pytest.approx(original.a, rel=1e-3)

    def test_poscar_to_xyz_to_poscar(self):
        """Test conversion POSCAR -> XYZ -> back to structure."""
        original = parse_structure(SILICON_POSCAR, StructureFormat.POSCAR)
        xyz = to_xyz(original)
        reparsed = parse_structure(xyz, StructureFormat.XYZ)

        # Note: XYZ loses lattice info, so we can't compare lattice parameters
        assert reparsed.formula == original.formula
        assert reparsed.num_atoms == original.num_atoms


# ========== InternalStructureModel Tests ==========


class TestInternalStructureModel:
    """Test InternalStructureModel validation and constraints."""

    def test_valid_structure_creation(self):
        """Test creating a valid InternalStructureModel."""
        structure = InternalStructureModel(
            lattice_vectors=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            atomic_species=["Si", "Si"],
            atomic_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            dimensionality=3,
            formula="Si",
            a=5.0, b=5.0, c=5.0,
            alpha=90.0, beta=90.0, gamma=90.0,
            volume=125.0,
            num_atoms=2
        )

        assert structure.formula == "Si"
        assert structure.num_atoms == 2

    def test_invalid_lattice_vectors_not_3x3(self):
        """Test that non-3x3 lattice vectors are rejected."""
        with pytest.raises(ValueError) as exc_info:
            InternalStructureModel(
                lattice_vectors=[[5.0, 0], [0, 5.0]],  # Only 2x2
                atomic_species=["Si"],
                atomic_positions=[[0.0, 0.0, 0.0]],
                dimensionality=3,
                formula="Si",
                a=5.0, b=5.0, c=5.0,
                alpha=90.0, beta=90.0, gamma=90.0,
                volume=125.0,
                num_atoms=1
            )

        assert "3x3 matrix" in str(exc_info.value)

    def test_invalid_atomic_positions_not_nx3(self):
        """Test that atomic positions must be Nx3."""
        with pytest.raises(ValueError) as exc_info:
            InternalStructureModel(
                lattice_vectors=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                atomic_species=["Si"],
                atomic_positions=[[0.0, 0.0]],  # Only 2 coordinates
                dimensionality=3,
                formula="Si",
                a=5.0, b=5.0, c=5.0,
                alpha=90.0, beta=90.0, gamma=90.0,
                volume=125.0,
                num_atoms=1
            )

        assert "3 components" in str(exc_info.value)

    def test_mismatch_species_positions(self):
        """Test that species and positions must have same length."""
        with pytest.raises(ValueError) as exc_info:
            InternalStructureModel(
                lattice_vectors=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                atomic_species=["Si", "Si"],  # 2 species
                atomic_positions=[[0.0, 0.0, 0.0]],  # 1 position
                dimensionality=3,
                formula="Si",
                a=5.0, b=5.0, c=5.0,
                alpha=90.0, beta=90.0, gamma=90.0,
                volume=125.0,
                num_atoms=2
            )

        assert "mismatch" in str(exc_info.value).lower()

    def test_invalid_dimensionality(self):
        """Test that dimensionality must be 0-3."""
        with pytest.raises(ValueError):
            InternalStructureModel(
                lattice_vectors=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                atomic_species=["Si"],
                atomic_positions=[[0.0, 0.0, 0.0]],
                dimensionality=4,  # Invalid
                formula="Si",
                a=5.0, b=5.0, c=5.0,
                alpha=90.0, beta=90.0, gamma=90.0,
                volume=125.0,
                num_atoms=1
            )

    def test_negative_lattice_parameter(self):
        """Test that lattice parameters must be positive."""
        with pytest.raises(ValueError):
            InternalStructureModel(
                lattice_vectors=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
                atomic_species=["Si"],
                atomic_positions=[[0.0, 0.0, 0.0]],
                dimensionality=3,
                formula="Si",
                a=-5.0,  # Invalid
                b=5.0, c=5.0,
                alpha=90.0, beta=90.0, gamma=90.0,
                volume=125.0,
                num_atoms=1
            )


# ========== pytest configuration ==========


def pytest_addoption(parser):
    """Add custom pytest command line options."""
    parser.addoption(
        "--run-pymatgen",
        action="store_true",
        default=False,
        help="Run tests that require pymatgen installation"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "skipif: skip tests conditionally"
    )

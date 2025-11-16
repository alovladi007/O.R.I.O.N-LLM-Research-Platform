"""
Crystallographic structure file parsing and conversion module.

This module provides comprehensive parsing, validation, and export capabilities
for common crystallographic file formats using pymatgen as the underlying engine.

Supported formats:
- CIF (Crystallographic Information File)
- POSCAR (VASP structure format)
- XYZ (Simple Cartesian coordinates)

The module provides a clean abstraction layer over pymatgen with:
- Consistent error handling
- Type-safe Pydantic models
- Detailed validation
- Format conversion utilities

Example:
    >>> from backend.common.structures import parse_structure, StructureFormat, to_cif
    >>>
    >>> # Parse a CIF file
    >>> with open('structure.cif', 'r') as f:
    ...     structure = parse_structure(f.read(), StructureFormat.CIF)
    >>>
    >>> # Access structure properties
    >>> print(f"Formula: {structure.formula}")
    >>> print(f"Space group dimensionality: {structure.dimensionality}")
    >>> print(f"Lattice parameters: a={structure.a:.3f}, b={structure.b:.3f}, c={structure.c:.3f}")
    >>> print(f"Volume: {structure.volume:.2f} Å³")
    >>>
    >>> # Convert to different format
    >>> poscar_text = to_poscar(structure)
    >>> xyz_text = to_xyz(structure)
"""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import traceback

try:
    from pymatgen.core import Structure, Lattice
    from pymatgen.io.cif import CifParser
    from pymatgen.io.vasp import Poscar
    from pymatgen.io.xyz import XYZ
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False
    # Define placeholder classes for type hints when pymatgen is not installed
    Structure = None
    Lattice = None

from src.api.exceptions import ParsingError


class StructureFormat(str, Enum):
    """Supported crystallographic file formats."""
    CIF = "CIF"
    POSCAR = "POSCAR"
    XYZ = "XYZ"


class InternalStructureModel(BaseModel):
    """
    Normalized internal representation of a crystal structure.

    This model provides a unified interface for working with crystallographic
    structures regardless of their original file format. All atomic positions
    are stored in fractional coordinates (0-1 range).

    Attributes:
        lattice_vectors: 3x3 matrix of lattice vectors in Angstroms [[ax,ay,az], [bx,by,bz], [cx,cy,cz]]
        atomic_species: List of element symbols for each atom (e.g., ['Si', 'Si', 'O', 'O'])
        atomic_positions: Nx3 matrix of fractional coordinates (0-1) for each atom
        dimensionality: Dimensionality of the structure (0=molecule, 1=chain, 2=layer, 3=bulk)
        formula: Reduced chemical formula (e.g., 'SiO2')
        a: Lattice parameter a in Angstroms
        b: Lattice parameter b in Angstroms
        c: Lattice parameter c in Angstroms
        alpha: Lattice angle alpha in degrees
        beta: Lattice angle beta in degrees
        gamma: Lattice angle gamma in degrees
        volume: Unit cell volume in Angstrom³
        num_atoms: Total number of atoms in the unit cell
        space_group: International space group symbol (optional)
        space_group_number: International space group number (optional)

    Example:
        >>> structure = InternalStructureModel(
        ...     lattice_vectors=[[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
        ...     atomic_species=['Si', 'Si'],
        ...     atomic_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        ...     dimensionality=3,
        ...     formula='Si',
        ...     a=5.0, b=5.0, c=5.0,
        ...     alpha=90.0, beta=90.0, gamma=90.0,
        ...     volume=125.0,
        ...     num_atoms=2
        ... )
    """

    # Lattice information
    lattice_vectors: List[List[float]] = Field(
        ...,
        description="3x3 matrix of lattice vectors in Angstroms",
        min_length=3,
        max_length=3
    )

    # Atomic information
    atomic_species: List[str] = Field(
        ...,
        description="List of element symbols",
        min_length=1
    )
    atomic_positions: List[List[float]] = Field(
        ...,
        description="Nx3 matrix of fractional coordinates (0-1)",
        min_length=1
    )

    # Derived properties
    dimensionality: int = Field(
        ...,
        ge=0,
        le=3,
        description="Dimensionality: 0=molecule, 1=chain, 2=layer, 3=bulk"
    )
    formula: str = Field(
        ...,
        description="Reduced chemical formula"
    )

    # Lattice parameters
    a: float = Field(..., gt=0, description="Lattice parameter a in Angstroms")
    b: float = Field(..., gt=0, description="Lattice parameter b in Angstroms")
    c: float = Field(..., gt=0, description="Lattice parameter c in Angstroms")
    alpha: float = Field(..., gt=0, le=180, description="Lattice angle alpha in degrees")
    beta: float = Field(..., gt=0, le=180, description="Lattice angle beta in degrees")
    gamma: float = Field(..., gt=0, le=180, description="Lattice angle gamma in degrees")
    volume: float = Field(..., gt=0, description="Unit cell volume in Angstrom³")

    # Counts
    num_atoms: int = Field(..., ge=1, description="Total number of atoms")

    # Optional symmetry information
    space_group: Optional[str] = Field(None, description="International space group symbol")
    space_group_number: Optional[int] = Field(None, ge=1, le=230, description="Space group number")

    @field_validator('lattice_vectors')
    @classmethod
    def validate_lattice_vectors(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate that lattice_vectors is a proper 3x3 matrix."""
        if len(v) != 3:
            raise ValueError("lattice_vectors must be a 3x3 matrix")
        for i, vec in enumerate(v):
            if len(vec) != 3:
                raise ValueError(f"lattice_vectors[{i}] must have exactly 3 components")
            if not all(isinstance(x, (int, float)) for x in vec):
                raise ValueError(f"lattice_vectors[{i}] must contain only numbers")
        return v

    @field_validator('atomic_positions')
    @classmethod
    def validate_atomic_positions(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate that atomic_positions is an Nx3 matrix with valid fractional coordinates."""
        for i, pos in enumerate(v):
            if len(pos) != 3:
                raise ValueError(f"atomic_positions[{i}] must have exactly 3 components")
            if not all(isinstance(x, (int, float)) for x in pos):
                raise ValueError(f"atomic_positions[{i}] must contain only numbers")
        return v

    @model_validator(mode='after')
    def validate_consistency(self) -> 'InternalStructureModel':
        """Validate that atomic_species and atomic_positions have consistent lengths."""
        if len(self.atomic_species) != len(self.atomic_positions):
            raise ValueError(
                f"Mismatch: {len(self.atomic_species)} atomic_species but "
                f"{len(self.atomic_positions)} atomic_positions"
            )
        if len(self.atomic_species) != self.num_atoms:
            raise ValueError(
                f"Mismatch: num_atoms={self.num_atoms} but found "
                f"{len(self.atomic_species)} atoms"
            )
        return self

    class Config:
        """Pydantic model configuration."""
        json_schema_extra = {
            "example": {
                "lattice_vectors": [[5.46873, 0.0, 0.0], [0.0, 5.46873, 0.0], [0.0, 0.0, 5.46873]],
                "atomic_species": ["Si", "Si"],
                "atomic_positions": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
                "dimensionality": 3,
                "formula": "Si",
                "a": 5.46873,
                "b": 5.46873,
                "c": 5.46873,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0,
                "volume": 163.54,
                "num_atoms": 2,
                "space_group": "Fd-3m",
                "space_group_number": 227
            }
        }


def _check_pymatgen() -> None:
    """
    Check if pymatgen is available.

    Raises:
        ParsingError: If pymatgen is not installed
    """
    if not PYMATGEN_AVAILABLE:
        raise ParsingError(
            file_format="ANY",
            message="pymatgen library is required but not installed. "
                   "Install with: pip install pymatgen",
            details={"requirement": "pymatgen>=2023.5.0"}
        )


def _determine_dimensionality(structure: 'Structure') -> int:
    """
    Determine the dimensionality of a structure.

    Uses heuristics based on lattice parameters and vacuum spacing:
    - 3D: All three lattice parameters are reasonable (< 50 Å typically)
    - 2D: One direction has significant vacuum (> 15 Å)
    - 1D: Two directions have significant vacuum
    - 0D: All three directions have significant vacuum (molecule)

    Args:
        structure: Pymatgen Structure object

    Returns:
        Dimensionality (0, 1, 2, or 3)
    """
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c

    # Threshold for considering a direction as "vacuum"
    VACUUM_THRESHOLD = 15.0  # Angstroms

    # Check which directions might have vacuum
    has_vacuum = [
        a > VACUUM_THRESHOLD,
        b > VACUUM_THRESHOLD,
        c > VACUUM_THRESHOLD
    ]

    vacuum_count = sum(has_vacuum)

    if vacuum_count == 0:
        return 3  # Bulk 3D crystal
    elif vacuum_count == 1:
        return 2  # 2D layer/slab
    elif vacuum_count == 2:
        return 1  # 1D chain/wire
    else:
        return 0  # 0D molecule/cluster


def _pymatgen_to_internal(structure: 'Structure') -> InternalStructureModel:
    """
    Convert a pymatgen Structure to InternalStructureModel.

    Args:
        structure: Pymatgen Structure object

    Returns:
        InternalStructureModel with all fields populated

    Raises:
        ParsingError: If conversion fails
    """
    try:
        lattice = structure.lattice

        # Extract lattice vectors
        lattice_vectors = lattice.matrix.tolist()

        # Extract atomic information
        atomic_species = [str(site.specie) for site in structure]
        atomic_positions = [site.frac_coords.tolist() for site in structure]

        # Get lattice parameters
        a, b, c = lattice.a, lattice.b, lattice.c
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        volume = lattice.volume

        # Get formula
        formula = structure.composition.reduced_formula

        # Determine dimensionality
        dimensionality = _determine_dimensionality(structure)

        # Try to get space group information
        space_group = None
        space_group_number = None
        try:
            spa = SpacegroupAnalyzer(structure)
            space_group = spa.get_space_group_symbol()
            space_group_number = spa.get_space_group_number()
        except Exception:
            # Space group determination can fail for low-symmetry or molecular structures
            pass

        return InternalStructureModel(
            lattice_vectors=lattice_vectors,
            atomic_species=atomic_species,
            atomic_positions=atomic_positions,
            dimensionality=dimensionality,
            formula=formula,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            volume=volume,
            num_atoms=len(structure),
            space_group=space_group,
            space_group_number=space_group_number
        )
    except Exception as e:
        raise ParsingError(
            file_format="INTERNAL",
            message=f"Failed to convert pymatgen Structure to internal model: {str(e)}",
            details={"error": str(e), "traceback": traceback.format_exc()}
        )


def parse_structure(text: str, format: StructureFormat) -> InternalStructureModel:
    """
    Parse a structure file from text in the specified format.

    This function acts as the main entry point for parsing crystallographic
    structure files. It handles format-specific parsing logic and converts
    the result to a unified internal representation.

    Args:
        text: Raw text content of the structure file
        format: File format (CIF, POSCAR, or XYZ)

    Returns:
        InternalStructureModel containing the parsed structure

    Raises:
        ParsingError: If parsing fails with detailed error information

    Example:
        >>> cif_content = '''
        ... data_crystal
        ... _cell_length_a    5.46873
        ... _cell_length_b    5.46873
        ... _cell_length_c    5.46873
        ... _cell_angle_alpha 90.0
        ... _cell_angle_beta  90.0
        ... _cell_angle_gamma 90.0
        ... _symmetry_space_group_name_H-M 'F d -3 m'
        ... loop_
        ... _atom_site_label
        ... _atom_site_fract_x
        ... _atom_site_fract_y
        ... _atom_site_fract_z
        ... Si 0.0 0.0 0.0
        ... Si 0.25 0.25 0.25
        ... '''
        >>> structure = parse_structure(cif_content, StructureFormat.CIF)
        >>> print(structure.formula)
        Si
    """
    _check_pymatgen()

    if not text or not text.strip():
        raise ParsingError(
            file_format=format.value,
            message="Empty or whitespace-only input",
            details={"input_length": len(text)}
        )

    try:
        if format == StructureFormat.CIF:
            return _parse_cif(text)
        elif format == StructureFormat.POSCAR:
            return _parse_poscar(text)
        elif format == StructureFormat.XYZ:
            return _parse_xyz(text)
        else:
            raise ParsingError(
                file_format=format.value,
                message=f"Unsupported format: {format}",
                details={"supported_formats": [f.value for f in StructureFormat]}
            )
    except ParsingError:
        # Re-raise ParsingErrors as-is
        raise
    except Exception as e:
        raise ParsingError(
            file_format=format.value,
            message=f"Unexpected error during parsing: {str(e)}",
            details={"error": str(e), "traceback": traceback.format_exc()}
        )


def _parse_cif(text: str) -> InternalStructureModel:
    """
    Parse a CIF (Crystallographic Information File) format structure.

    Args:
        text: CIF file content as string

    Returns:
        InternalStructureModel

    Raises:
        ParsingError: If CIF parsing fails
    """
    try:
        # CifParser can handle string input directly
        parser = CifParser.from_string(text)

        # Get all structures from CIF (CIF can contain multiple structures)
        structures = parser.get_structures()

        if not structures:
            raise ParsingError(
                file_format="CIF",
                message="No structures found in CIF file",
                details={"num_structures": 0}
            )

        if len(structures) > 1:
            # Log warning but use the first structure
            import warnings
            warnings.warn(
                f"CIF file contains {len(structures)} structures. Using the first one."
            )

        structure = structures[0]
        return _pymatgen_to_internal(structure)

    except ParsingError:
        raise
    except Exception as e:
        error_msg = str(e)
        details = {"error": error_msg}

        # Try to provide more specific error information
        if "ParseError" in error_msg or "parse" in error_msg.lower():
            details["hint"] = "Check for malformed CIF syntax or missing required fields"
        elif "KeyError" in error_msg:
            details["hint"] = "Missing required CIF tags (e.g., _cell_length_a, _atom_site_*)"

        raise ParsingError(
            file_format="CIF",
            message=f"Failed to parse CIF file: {error_msg}",
            details=details
        )


def _parse_poscar(text: str) -> InternalStructureModel:
    """
    Parse a POSCAR (VASP structure) format file.

    POSCAR format specification:
    - Line 1: Comment line
    - Line 2: Universal scaling factor
    - Lines 3-5: Lattice vectors
    - Line 6: Element symbols (VASP 5.x) or optional
    - Line 7: Number of atoms per element
    - Line 8: Coordinate type (Direct/Cartesian)
    - Remaining: Atomic positions

    Args:
        text: POSCAR file content as string

    Returns:
        InternalStructureModel

    Raises:
        ParsingError: If POSCAR parsing fails
    """
    try:
        # Poscar.from_string expects the content directly
        poscar = Poscar.from_string(text)
        structure = poscar.structure

        return _pymatgen_to_internal(structure)

    except Exception as e:
        error_msg = str(e)
        details = {"error": error_msg}

        # Provide helpful error messages
        lines = text.strip().split('\n')
        if len(lines) < 8:
            details["hint"] = f"POSCAR file too short: expected at least 8 lines, got {len(lines)}"
        elif "scaling" in error_msg.lower():
            details["hint"] = "Invalid scaling factor on line 2 (must be a positive number)"
        elif "lattice" in error_msg.lower():
            details["hint"] = "Invalid lattice vectors on lines 3-5 (must be 3 lines of 3 numbers each)"

        raise ParsingError(
            file_format="POSCAR",
            message=f"Failed to parse POSCAR file: {error_msg}",
            details=details
        )


def _parse_xyz(text: str) -> InternalStructureModel:
    """
    Parse an XYZ format structure file.

    XYZ format specification:
    - Line 1: Number of atoms
    - Line 2: Comment line
    - Remaining lines: Element X Y Z (Cartesian coordinates in Angstroms)

    Note: XYZ format does not contain lattice information. A cubic box
    is constructed automatically with padding around the molecule.

    Args:
        text: XYZ file content as string

    Returns:
        InternalStructureModel

    Raises:
        ParsingError: If XYZ parsing fails
    """
    try:
        # XYZ files don't have lattice information, so we need to construct one
        # Pymatgen's XYZ parser returns a Molecule, not a Structure
        xyz = XYZ.from_string(text)
        molecule = xyz.molecule

        # Convert molecule to structure by placing it in a box
        # Add 10 Angstrom padding on each side
        coords = molecule.cart_coords
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        # Create a cubic lattice with padding
        padding = 10.0  # Angstroms
        box_size = (max_coords - min_coords) + 2 * padding

        # Make it cubic (use max dimension)
        box_size = max(box_size) * 1.0

        # Create lattice
        lattice = Lattice.cubic(box_size)

        # Shift coordinates to center of box and convert to fractional
        centered_coords = coords - min_coords + padding
        frac_coords = lattice.get_fractional_coords(centered_coords)

        # Create structure
        structure = Structure(
            lattice,
            molecule.species,
            frac_coords,
            coords_are_cartesian=False
        )

        return _pymatgen_to_internal(structure)

    except Exception as e:
        error_msg = str(e)
        details = {"error": error_msg}

        # Provide helpful error messages
        lines = text.strip().split('\n')
        if len(lines) < 3:
            details["hint"] = f"XYZ file too short: expected at least 3 lines, got {len(lines)}"
        else:
            try:
                num_atoms = int(lines[0].strip())
                if len(lines) < num_atoms + 2:
                    details["hint"] = (
                        f"Expected {num_atoms} atom lines (from line 1) but "
                        f"only found {len(lines) - 2}"
                    )
            except ValueError:
                details["hint"] = "Line 1 must contain the number of atoms (integer)"

        raise ParsingError(
            file_format="XYZ",
            message=f"Failed to parse XYZ file: {error_msg}",
            details=details
        )


def to_cif(structure: InternalStructureModel) -> str:
    """
    Export structure to CIF (Crystallographic Information File) format.

    Args:
        structure: InternalStructureModel to export

    Returns:
        CIF format string

    Raises:
        ParsingError: If export fails

    Example:
        >>> structure = parse_structure(poscar_text, StructureFormat.POSCAR)
        >>> cif_output = to_cif(structure)
        >>> print(cif_output)
    """
    _check_pymatgen()

    try:
        # Convert internal model back to pymatgen Structure
        lattice = Lattice(structure.lattice_vectors)
        pmg_structure = Structure(
            lattice,
            structure.atomic_species,
            structure.atomic_positions,
            coords_are_cartesian=False
        )

        # Use pymatgen's CIF writer
        from pymatgen.io.cif import CifWriter
        writer = CifWriter(pmg_structure)

        return str(writer)

    except Exception as e:
        raise ParsingError(
            file_format="CIF",
            message=f"Failed to export to CIF format: {str(e)}",
            details={"error": str(e), "traceback": traceback.format_exc()}
        )


def to_poscar(structure: InternalStructureModel) -> str:
    """
    Export structure to POSCAR (VASP structure) format.

    Args:
        structure: InternalStructureModel to export

    Returns:
        POSCAR format string

    Raises:
        ParsingError: If export fails

    Example:
        >>> structure = parse_structure(cif_text, StructureFormat.CIF)
        >>> poscar_output = to_poscar(structure)
        >>> print(poscar_output)
    """
    _check_pymatgen()

    try:
        # Convert internal model back to pymatgen Structure
        lattice = Lattice(structure.lattice_vectors)
        pmg_structure = Structure(
            lattice,
            structure.atomic_species,
            structure.atomic_positions,
            coords_are_cartesian=False
        )

        # Use pymatgen's Poscar writer
        poscar = Poscar(pmg_structure)

        return str(poscar)

    except Exception as e:
        raise ParsingError(
            file_format="POSCAR",
            message=f"Failed to export to POSCAR format: {str(e)}",
            details={"error": str(e), "traceback": traceback.format_exc()}
        )


def to_xyz(structure: InternalStructureModel) -> str:
    """
    Export structure to XYZ format.

    Note: XYZ format only contains Cartesian coordinates and element symbols.
    Lattice information is lost in this format.

    Args:
        structure: InternalStructureModel to export

    Returns:
        XYZ format string

    Raises:
        ParsingError: If export fails

    Example:
        >>> structure = parse_structure(cif_text, StructureFormat.CIF)
        >>> xyz_output = to_xyz(structure)
        >>> print(xyz_output)
    """
    _check_pymatgen()

    try:
        # Convert internal model back to pymatgen Structure
        lattice = Lattice(structure.lattice_vectors)
        pmg_structure = Structure(
            lattice,
            structure.atomic_species,
            structure.atomic_positions,
            coords_are_cartesian=False
        )

        # Use pymatgen's XYZ writer
        xyz = XYZ(pmg_structure)

        return str(xyz)

    except Exception as e:
        raise ParsingError(
            file_format="XYZ",
            message=f"Failed to export to XYZ format: {str(e)}",
            details={"error": str(e), "traceback": traceback.format_exc()}
        )


# ========== Convenience Functions ==========


def convert_structure(
    input_text: str,
    input_format: StructureFormat,
    output_format: StructureFormat
) -> str:
    """
    Convert a structure from one format to another.

    This is a convenience function that combines parsing and export.

    Args:
        input_text: Structure file content in input format
        input_format: Input file format
        output_format: Desired output format

    Returns:
        Structure in output format

    Raises:
        ParsingError: If parsing or export fails

    Example:
        >>> # Convert POSCAR to CIF
        >>> cif_output = convert_structure(poscar_text, StructureFormat.POSCAR, StructureFormat.CIF)
    """
    # Parse the input
    structure = parse_structure(input_text, input_format)

    # Export to output format
    if output_format == StructureFormat.CIF:
        return to_cif(structure)
    elif output_format == StructureFormat.POSCAR:
        return to_poscar(structure)
    elif output_format == StructureFormat.XYZ:
        return to_xyz(structure)
    else:
        raise ParsingError(
            file_format=output_format.value,
            message=f"Unsupported output format: {output_format}",
            details={"supported_formats": [f.value for f in StructureFormat]}
        )

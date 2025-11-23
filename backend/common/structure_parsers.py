"""
Structure File Parsers
======================

Parsers for common atomic structure file formats:
- CIF (Crystallographic Information File)
- POSCAR (VASP format)
- XYZ (XYZ coordinate file)

These parsers extract:
- Chemical formula
- Lattice parameters
- Atomic positions
- Number of atoms
- Dimensionality
"""

import re
import logging
from typing import Dict, List, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


def parse_structure(text: str, format: str) -> Dict[str, Any]:
    """
    Parse structure file based on format.

    Args:
        text: Raw structure file content
        format: File format (CIF, POSCAR, XYZ, INTERNAL)

    Returns:
        Dictionary with parsed structure data

    Raises:
        ValueError: If format is not supported or parsing fails
    """
    format = format.upper()

    parsers = {
        "CIF": parse_cif,
        "POSCAR": parse_poscar,
        "XYZ": parse_xyz,
        "VASP": parse_poscar,  # Alias
    }

    if format not in parsers:
        raise ValueError(f"Unsupported format: {format}. Supported: {list(parsers.keys())}")

    try:
        return parsers[format](text)
    except Exception as e:
        logger.error(f"Failed to parse {format} file: {e}")
        raise ValueError(f"Parsing {format} file failed: {e}")


def parse_cif(text: str) -> Dict[str, Any]:
    """
    Parse CIF (Crystallographic Information File) format.

    Basic CIF parser that extracts:
    - Chemical formula
    - Lattice parameters (a, b, c, alpha, beta, gamma)
    - Atomic positions
    - Space group (if available)

    Args:
        text: CIF file content

    Returns:
        Parsed structure dictionary
    """
    lines = [line.strip() for line in text.split('\n') if line.strip() and not line.strip().startswith('#')]

    result = {
        "formula": "Unknown",
        "num_atoms": 0,
        "dimensionality": 3,
        "lattice": {},
        "atoms": [],
        "lattice_parameters": {},
        "space_group": None
    }

    # Extract lattice parameters
    lattice_params = {}
    for line in lines:
        if '_cell_length_a' in line:
            lattice_params['a'] = float(re.search(r'[\d.]+', line.split()[-1]).group())
        elif '_cell_length_b' in line:
            lattice_params['b'] = float(re.search(r'[\d.]+', line.split()[-1]).group())
        elif '_cell_length_c' in line:
            lattice_params['c'] = float(re.search(r'[\d.]+', line.split()[-1]).group())
        elif '_cell_angle_alpha' in line:
            lattice_params['alpha'] = float(re.search(r'[\d.]+', line.split()[-1]).group())
        elif '_cell_angle_beta' in line:
            lattice_params['beta'] = float(re.search(r'[\d.]+', line.split()[-1]).group())
        elif '_cell_angle_gamma' in line:
            lattice_params['gamma'] = float(re.search(r'[\d.]+', line.split()[-1]).group())
        elif '_chemical_formula_sum' in line:
            # Extract formula (e.g., 'Mo S2' -> 'MoS2')
            formula_match = re.search(r"'([^']+)'", line)
            if formula_match:
                result['formula'] = formula_match.group(1).replace(' ', '')

    # Calculate volume (approximate for orthorhombic)
    if all(k in lattice_params for k in ['a', 'b', 'c']):
        a, b, c = lattice_params['a'], lattice_params['b'], lattice_params['c']
        alpha = lattice_params.get('alpha', 90.0)
        beta = lattice_params.get('beta', 90.0)
        gamma = lattice_params.get('gamma', 90.0)

        # Convert to radians
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

        # Calculate volume
        volume = a * b * c * np.sqrt(
            1 - np.cos(alpha_rad)**2 - np.cos(beta_rad)**2 - np.cos(gamma_rad)**2
            + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
        )
        lattice_params['volume'] = volume

    result['lattice_parameters'] = lattice_params

    # Parse atomic positions (simplified - full CIF parsing is complex)
    in_atom_site = False
    atom_data = []
    for line in lines:
        if 'loop_' in line:
            in_atom_site = False
        if '_atom_site' in line:
            in_atom_site = True
            continue
        if in_atom_site and line and not line.startswith('_'):
            parts = line.split()
            if len(parts) >= 4:
                atom_data.append({
                    "element": parts[0],
                    "position": [float(parts[1]), float(parts[2]), float(parts[3])]
                })

    result['atoms'] = atom_data
    result['num_atoms'] = len(atom_data)

    logger.info(f"Parsed CIF: {result['formula']}, {result['num_atoms']} atoms")
    return result


def parse_poscar(text: str) -> Dict[str, Any]:
    """
    Parse POSCAR/CONTCAR (VASP format).

    Format:
    Line 1: Comment
    Line 2: Scaling factor
    Lines 3-5: Lattice vectors
    Line 6: Element symbols
    Line 7: Number of atoms per element
    Line 8: Coordinate type (Direct/Cartesian)
    Lines 9+: Atomic positions

    Args:
        text: POSCAR file content

    Returns:
        Parsed structure dictionary
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if len(lines) < 8:
        raise ValueError("POSCAR file too short")

    result = {
        "formula": "Unknown",
        "num_atoms": 0,
        "dimensionality": 3,
        "lattice": {},
        "atoms": [],
        "lattice_parameters": {}
    }

    # Line 1: Comment (often contains formula)
    comment = lines[0]
    result['formula'] = comment.split()[0] if comment else "Unknown"

    # Line 2: Scaling factor
    scale = float(lines[1])

    # Lines 3-5: Lattice vectors
    lattice_vectors = []
    for i in range(2, 5):
        vector = [float(x) * scale for x in lines[i].split()]
        lattice_vectors.append(vector)

    result['lattice']['vectors'] = lattice_vectors

    # Calculate lattice parameters
    a_vec = np.array(lattice_vectors[0])
    b_vec = np.array(lattice_vectors[1])
    c_vec = np.array(lattice_vectors[2])

    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    alpha = np.degrees(np.arccos(np.dot(b_vec, c_vec) / (b * c)))
    beta = np.degrees(np.arccos(np.dot(a_vec, c_vec) / (a * c)))
    gamma = np.degrees(np.arccos(np.dot(a_vec, b_vec) / (a * b)))

    volume = np.abs(np.dot(a_vec, np.cross(b_vec, c_vec)))

    result['lattice_parameters'] = {
        'a': a, 'b': b, 'c': c,
        'alpha': alpha, 'beta': beta, 'gamma': gamma,
        'volume': volume
    }

    # Line 6: Element symbols
    elements = lines[5].split()

    # Line 7: Number of atoms per element
    atom_counts = [int(x) for x in lines[6].split()]

    total_atoms = sum(atom_counts)
    result['num_atoms'] = total_atoms

    # Build formula
    formula_parts = [f"{elem}{count}" if count > 1 else elem
                     for elem, count in zip(elements, atom_counts)]
    result['formula'] = ''.join(formula_parts)

    # Line 8: Coordinate type
    coord_type = lines[7][0].upper()  # 'D' for Direct, 'C' for Cartesian

    # Parse atomic positions
    atom_idx = 0
    for elem_idx, (element, count) in enumerate(zip(elements, atom_counts)):
        for i in range(count):
            line_idx = 8 + atom_idx
            if line_idx < len(lines):
                coords = [float(x) for x in lines[line_idx].split()[:3]]
                result['atoms'].append({
                    "element": element,
                    "position": coords,
                    "coordinate_type": "fractional" if coord_type == 'D' else "cartesian"
                })
                atom_idx += 1

    logger.info(f"Parsed POSCAR: {result['formula']}, {result['num_atoms']} atoms")
    return result


def parse_xyz(text: str) -> Dict[str, Any]:
    """
    Parse XYZ coordinate file.

    Format:
    Line 1: Number of atoms
    Line 2: Comment (optional)
    Lines 3+: Element X Y Z

    Args:
        text: XYZ file content

    Returns:
        Parsed structure dictionary
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if len(lines) < 2:
        raise ValueError("XYZ file too short")

    result = {
        "formula": "Unknown",
        "num_atoms": 0,
        "dimensionality": 0,  # XYZ doesn't specify periodicity
        "lattice": {},
        "atoms": [],
        "lattice_parameters": {}
    }

    # Line 1: Number of atoms
    num_atoms = int(lines[0])
    result['num_atoms'] = num_atoms

    # Line 2: Comment (might contain formula)
    comment = lines[1] if len(lines) > 1 else ""

    # Parse atoms
    atoms = []
    element_counts = {}
    for i in range(2, min(2 + num_atoms, len(lines))):
        parts = lines[i].split()
        if len(parts) >= 4:
            element = parts[0]
            coords = [float(parts[1]), float(parts[2]), float(parts[3])]
            atoms.append({
                "element": element,
                "position": coords,
                "coordinate_type": "cartesian"
            })
            element_counts[element] = element_counts.get(element, 0) + 1

    result['atoms'] = atoms

    # Build formula from element counts
    formula_parts = [f"{elem}{count}" if count > 1 else elem
                     for elem, count in sorted(element_counts.items())]
    result['formula'] = ''.join(formula_parts)

    # Estimate lattice (bounding box + padding)
    if atoms:
        positions = np.array([atom['position'] for atom in atoms])
        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)

        # Add 5 Angstrom padding
        padding = 5.0
        a = max_coords[0] - min_coords[0] + 2 * padding
        b = max_coords[1] - min_coords[1] + 2 * padding
        c = max_coords[2] - min_coords[2] + 2 * padding

        result['lattice_parameters'] = {
            'a': a, 'b': b, 'c': c,
            'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0,
            'volume': a * b * c
        }

    logger.info(f"Parsed XYZ: {result['formula']}, {result['num_atoms']} atoms")
    return result


# Helper functions
def calculate_distance(pos1: List[float], pos2: List[float]) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def detect_dimensionality(atoms: List[Dict], lattice_params: Dict) -> int:
    """
    Detect structure dimensionality (0D, 1D, 2D, 3D).

    Based on spatial distribution of atoms.

    Returns:
        0 = molecule, 1 = 1D chain, 2 = 2D sheet, 3 = 3D bulk
    """
    if not atoms or not lattice_params:
        return 3  # Default to 3D

    positions = np.array([atom['position'] for atom in atoms])

    # Calculate extent in each direction
    extents = positions.max(axis=0) - positions.min(axis=0)

    # Threshold for "thin" dimension (Angstroms)
    threshold = 3.0

    thin_dimensions = sum(1 for extent in extents if extent < threshold)

    if thin_dimensions >= 2:
        return 1  # 1D
    elif thin_dimensions == 1:
        return 2  # 2D
    else:
        return 3  # 3D

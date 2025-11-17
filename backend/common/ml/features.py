"""
Feature Extraction Pipeline for Crystal Structures
===================================================

This module provides feature extraction for crystal structures to support
GNN-based property prediction models. It extracts:

1. Graph Representation:
   - Neighbor lists
   - Bond distances
   - Atomic species
   - Coordination numbers

2. Scalar Features:
   - Composition (element fractions)
   - Average electronegativity
   - Average atomic mass
   - Volume per atom
   - Density

The extracted features are stored in the StructureFeatures table for caching
and can be used directly by CGCNN-style models.

Session 14: ML Infrastructure Layer
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


# ============================================================================
# Element Properties (Simplified - for full implementation, use pymatgen)
# ============================================================================

ELEMENT_PROPERTIES = {
    "H": {"number": 1, "mass": 1.008, "electronegativity": 2.20, "radius": 0.53},
    "C": {"number": 6, "mass": 12.011, "electronegativity": 2.55, "radius": 0.77},
    "N": {"number": 7, "mass": 14.007, "electronegativity": 3.04, "radius": 0.75},
    "O": {"number": 8, "mass": 15.999, "electronegativity": 3.44, "radius": 0.73},
    "B": {"number": 5, "mass": 10.81, "electronegativity": 2.04, "radius": 0.82},
    "Al": {"number": 13, "mass": 26.982, "electronegativity": 1.61, "radius": 1.43},
    "Si": {"number": 14, "mass": 28.085, "electronegativity": 1.90, "radius": 1.18},
    "Ga": {"number": 31, "mass": 69.723, "electronegativity": 1.81, "radius": 1.36},
    "Mo": {"number": 42, "mass": 95.95, "electronegativity": 2.16, "radius": 1.40},
    "W": {"number": 74, "mass": 183.84, "electronegativity": 2.36, "radius": 1.41},
    "S": {"number": 16, "mass": 32.06, "electronegativity": 2.58, "radius": 1.04},
    "Se": {"number": 34, "mass": 78.971, "electronegativity": 2.55, "radius": 1.16},
    "Te": {"number": 52, "mass": 127.60, "electronegativity": 2.10, "radius": 1.36},
}


@dataclass
class GraphRepresentation:
    """
    Graph representation of a crystal structure for GNN input.

    This format is compatible with CGCNN and similar models.
    """
    # Node features (atoms)
    atom_features: List[Dict[str, Any]]  # Per-atom features
    num_atoms: int

    # Edge features (bonds)
    neighbor_lists: Dict[int, List[int]]  # atom_idx -> [neighbor indices]
    bond_distances: Dict[Tuple[int, int], float]  # (i, j) -> distance
    num_edges: int

    # Lattice information
    lattice_vectors: List[List[float]]  # 3x3 matrix
    volume: float

    # Metadata
    formula: str
    dimensionality: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        # Convert tuple keys to strings for JSON serialization
        bond_distances_serializable = {
            f"{i}_{j}": dist
            for (i, j), dist in self.bond_distances.items()
        }

        return {
            "atom_features": self.atom_features,
            "num_atoms": self.num_atoms,
            "neighbor_lists": self.neighbor_lists,
            "bond_distances": bond_distances_serializable,
            "num_edges": self.num_edges,
            "lattice_vectors": self.lattice_vectors,
            "volume": self.volume,
            "formula": self.formula,
            "dimensionality": self.dimensionality
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphRepresentation":
        """Reconstruct from dictionary."""
        # Convert string keys back to tuples
        bond_distances = {
            tuple(map(int, k.split("_"))): v
            for k, v in data["bond_distances"].items()
        }

        return cls(
            atom_features=data["atom_features"],
            num_atoms=data["num_atoms"],
            neighbor_lists=data["neighbor_lists"],
            bond_distances=bond_distances,
            num_edges=data["num_edges"],
            lattice_vectors=data["lattice_vectors"],
            volume=data["volume"],
            formula=data["formula"],
            dimensionality=data["dimensionality"]
        )


@dataclass
class ScalarFeatures:
    """Scalar (non-graph) features for a structure."""
    # Composition
    elements: List[str]
    element_fractions: Dict[str, float]
    num_elements: int

    # Aggregate properties
    avg_atomic_mass: float
    avg_electronegativity: float
    avg_atomic_radius: float

    # Structural
    volume_per_atom: float
    density: float  # g/cm³ (approximate)
    space_filling: float  # Fraction of volume occupied by atoms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "elements": self.elements,
            "element_fractions": self.element_fractions,
            "num_elements": self.num_elements,
            "avg_atomic_mass": self.avg_atomic_mass,
            "avg_electronegativity": self.avg_electronegativity,
            "avg_atomic_radius": self.avg_atomic_radius,
            "volume_per_atom": self.volume_per_atom,
            "density": self.density,
            "space_filling": self.space_filling
        }


# ============================================================================
# Feature Extraction Functions
# ============================================================================

def compute_distance(pos1: List[float], pos2: List[float]) -> float:
    """Compute Euclidean distance between two positions."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def compute_lattice_volume(lattice_vectors: List[List[float]]) -> float:
    """
    Compute volume of lattice defined by three vectors.

    Volume = |a · (b × c)|
    """
    a, b, c = np.array(lattice_vectors)
    volume = abs(np.dot(a, np.cross(b, c)))
    return volume


def find_neighbors(
    atoms: List[Dict[str, Any]],
    lattice_vectors: List[List[float]],
    cutoff_radius: float = 5.0,
    max_neighbors: int = 12
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], float]]:
    """
    Find neighboring atoms within cutoff radius.

    This is a simplified implementation. For production, use:
    - pymatgen.core.Structure.get_neighbors()
    - ASE neighbor list
    - matscipy.neighbours

    Args:
        atoms: List of atom dictionaries with 'position' key
        lattice_vectors: 3x3 lattice matrix
        cutoff_radius: Maximum distance for neighbors (Angstroms)
        max_neighbors: Maximum number of neighbors per atom

    Returns:
        (neighbor_lists, bond_distances)
    """
    num_atoms = len(atoms)
    neighbor_lists = {i: [] for i in range(num_atoms)}
    bond_distances = {}

    # Extract positions
    positions = np.array([atom["position"] for atom in atoms])

    # Compute pairwise distances (simplified - no PBC handling)
    for i in range(num_atoms):
        distances_to_i = []

        for j in range(num_atoms):
            if i == j:
                continue

            dist = compute_distance(positions[i], positions[j])

            if dist < cutoff_radius:
                distances_to_i.append((j, dist))

        # Sort by distance and keep top max_neighbors
        distances_to_i.sort(key=lambda x: x[1])
        distances_to_i = distances_to_i[:max_neighbors]

        # Store neighbors and distances
        for j, dist in distances_to_i:
            neighbor_lists[i].append(j)
            bond_distances[(i, j)] = dist

    return neighbor_lists, bond_distances


def compute_atom_features(atom: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute features for a single atom.

    For GNN input, these will be converted to vectors.
    """
    species = atom.get("species", "C")  # Default to carbon if not specified

    # Get element properties (with fallback)
    if species not in ELEMENT_PROPERTIES:
        logger.warning(f"Unknown element: {species}, using default properties")
        elem_props = {"number": 6, "mass": 12.0, "electronegativity": 2.0, "radius": 1.0}
    else:
        elem_props = ELEMENT_PROPERTIES[species]

    return {
        "species": species,
        "atomic_number": elem_props["number"],
        "atomic_mass": elem_props["mass"],
        "electronegativity": elem_props["electronegativity"],
        "atomic_radius": elem_props["radius"],
        "position": atom.get("position", [0.0, 0.0, 0.0]),
        "fractional": atom.get("fractional", [0.0, 0.0, 0.0])
    }


def build_graph_representation(
    atoms: List[Dict[str, Any]],
    lattice_vectors: List[List[float]],
    formula: str,
    dimensionality: int,
    cutoff_radius: float = 5.0
) -> GraphRepresentation:
    """
    Build graph representation for GNN input.

    Args:
        atoms: List of atom dictionaries
        lattice_vectors: 3x3 lattice matrix (Angstroms)
        formula: Chemical formula
        dimensionality: 0=molecule, 1=1D, 2=2D, 3=bulk
        cutoff_radius: Neighbor cutoff distance

    Returns:
        GraphRepresentation object
    """
    logger.info(f"Building graph representation for {formula} ({len(atoms)} atoms)")

    # Compute atom features
    atom_features = [compute_atom_features(atom) for atom in atoms]

    # Find neighbors
    neighbor_lists, bond_distances = find_neighbors(
        atoms,
        lattice_vectors,
        cutoff_radius=cutoff_radius
    )

    num_edges = sum(len(neighbors) for neighbors in neighbor_lists.values())

    # Compute volume
    volume = compute_lattice_volume(lattice_vectors)

    graph = GraphRepresentation(
        atom_features=atom_features,
        num_atoms=len(atoms),
        neighbor_lists=neighbor_lists,
        bond_distances=bond_distances,
        num_edges=num_edges,
        lattice_vectors=lattice_vectors,
        volume=volume,
        formula=formula,
        dimensionality=dimensionality
    )

    logger.info(f"Graph built: {graph.num_atoms} nodes, {graph.num_edges} edges")
    return graph


def compute_scalar_features(
    atoms: List[Dict[str, Any]],
    lattice_vectors: List[List[float]],
    formula: str
) -> ScalarFeatures:
    """
    Compute scalar features for a structure.

    Args:
        atoms: List of atom dictionaries
        lattice_vectors: 3x3 lattice matrix
        formula: Chemical formula

    Returns:
        ScalarFeatures object
    """
    # Count element occurrences
    element_counts = {}
    for atom in atoms:
        species = atom.get("species", "C")
        element_counts[species] = element_counts.get(species, 0) + 1

    # Compute element fractions
    num_atoms = len(atoms)
    element_fractions = {
        elem: count / num_atoms
        for elem, count in element_counts.items()
    }

    # Compute average properties
    total_mass = 0.0
    total_electroneg = 0.0
    total_radius = 0.0

    for atom in atoms:
        species = atom.get("species", "C")
        if species not in ELEMENT_PROPERTIES:
            props = {"mass": 12.0, "electronegativity": 2.0, "radius": 1.0}
        else:
            props = ELEMENT_PROPERTIES[species]

        total_mass += props["mass"]
        total_electroneg += props["electronegativity"]
        total_radius += props["radius"]

    avg_mass = total_mass / num_atoms
    avg_electroneg = total_electroneg / num_atoms
    avg_radius = total_radius / num_atoms

    # Compute volume-based features
    volume = compute_lattice_volume(lattice_vectors)
    volume_per_atom = volume / num_atoms

    # Approximate density (g/cm³)
    # density = (total_mass in amu) / (volume in Å³) * conversion
    # 1 amu/Å³ = 1.66054 g/cm³
    density = (total_mass / volume) * 1.66054

    # Space filling (approximate as fraction of volume occupied by atomic spheres)
    atomic_volumes = sum(
        (4/3) * np.pi * (ELEMENT_PROPERTIES.get(atom["species"], {"radius": 1.0})["radius"] ** 3)
        for atom in atoms
    )
    space_filling = min(1.0, atomic_volumes / volume)

    features = ScalarFeatures(
        elements=sorted(element_counts.keys()),
        element_fractions=element_fractions,
        num_elements=len(element_counts),
        avg_atomic_mass=avg_mass,
        avg_electronegativity=avg_electroneg,
        avg_atomic_radius=avg_radius,
        volume_per_atom=volume_per_atom,
        density=density,
        space_filling=space_filling
    )

    logger.info(f"Scalar features computed for {formula}")
    return features


def extract_structure_features(
    structure: Any,
    cutoff_radius: float = 5.0
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main entry point: Extract both graph and scalar features from a structure.

    Args:
        structure: Structure model object from src.api.models.structure
        cutoff_radius: Neighbor cutoff for graph building

    Returns:
        Tuple of (graph_repr_dict, scalar_features_dict)
    """
    logger.info(f"Extracting features for structure {structure.id}")

    # Extract structure data
    atoms = structure.atoms if structure.atoms else []
    lattice_vectors = structure.lattice_vectors if structure.lattice_vectors else [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    formula = structure.formula or "Unknown"
    dimensionality = structure.dimensionality if structure.dimensionality is not None else 3

    # Validate atoms
    if not atoms or len(atoms) == 0:
        logger.warning(f"Structure {structure.id} has no atoms, using placeholder")
        atoms = [{"species": "C", "position": [0, 0, 0], "fractional": [0, 0, 0]}]

    # Build graph representation
    graph_repr = build_graph_representation(
        atoms=atoms,
        lattice_vectors=lattice_vectors,
        formula=formula,
        dimensionality=dimensionality,
        cutoff_radius=cutoff_radius
    )

    # Compute scalar features
    scalar_features = compute_scalar_features(
        atoms=atoms,
        lattice_vectors=lattice_vectors,
        formula=formula
    )

    logger.info(f"Feature extraction complete for {structure.id}")

    return graph_repr.to_dict(), scalar_features.to_dict()


# ============================================================================
# Utility Functions
# ============================================================================

def features_to_cgcnn_format(graph_repr_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert graph representation to CGCNN-compatible format.

    This is a placeholder for actual CGCNN data loader integration.
    In a real implementation, this would interface with torch_geometric
    or a custom CGCNN data loader.
    """
    graph = GraphRepresentation.from_dict(graph_repr_dict)

    # Convert to format expected by CGCNN
    # (Simplified - actual implementation would create torch tensors)
    cgcnn_data = {
        "atom_fea": [  # Atom feature matrix (N_atoms x feature_dim)
            [
                atom["atomic_number"],
                atom["atomic_mass"],
                atom["electronegativity"],
                atom["atomic_radius"]
            ]
            for atom in graph.atom_features
        ],
        "nbr_fea_idx": [],  # Neighbor indices
        "nbr_fea": [],      # Neighbor distances
        "num_atoms": graph.num_atoms,
        "num_edges": graph.num_edges,
        "crystal_atom_idx": list(range(graph.num_atoms))
    }

    # Build edge list
    for i, neighbors in graph.neighbor_lists.items():
        for j in neighbors:
            cgcnn_data["nbr_fea_idx"].append([i, j])
            cgcnn_data["nbr_fea"].append(graph.bond_distances.get((i, j), 0.0))

    return cgcnn_data

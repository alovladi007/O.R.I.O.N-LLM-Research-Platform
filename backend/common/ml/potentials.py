"""
ML Interatomic Potential Training Module.

Provides functions for:
- Extracting training data from DFT simulations (structures + forces)
- Training ML potentials (SNAP, NequIP, MACE, etc.)
- Exporting potentials for use in LAMMPS

Session 18: Hybrid DFT-ML Workflows
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrainingSnapshot:
    """
    Single training snapshot with structure and energies/forces.

    Represents one configuration (structure) along with its DFT-computed
    energy and atomic forces.
    """
    structure_id: str
    atoms: List[Dict[str, Any]]  # [{"element": "Si", "position": [x,y,z]}, ...]
    lattice_vectors: List[List[float]]  # 3x3 matrix
    energy: float  # Total energy in eV
    forces: List[List[float]]  # Atomic forces in eV/Angstrom
    stress: Optional[List[List[float]]] = None  # Stress tensor (optional)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrainingDataset:
    """
    Complete training dataset for ML potential.

    Contains multiple snapshots extracted from DFT simulations,
    along with metadata about the dataset.
    """
    snapshots: List[TrainingSnapshot]
    elements: List[str]  # Unique elements in dataset
    num_snapshots: int
    energy_range: Tuple[float, float]  # (min, max) energy
    force_max: float  # Maximum force component
    metadata: Dict[str, Any]


@dataclass
class PotentialConfig:
    """
    Configuration for ML potential training.

    Specifies the descriptor type, hyperparameters, and training settings.
    """
    name: str
    descriptor_type: str  # "SNAP", "SOAP", "NequIP", "MACE"
    elements: List[str]

    # Descriptor-specific parameters
    descriptor_params: Dict[str, Any]

    # Training parameters
    train_fraction: float = 0.8
    learning_rate: float = 0.001
    num_epochs: int = 100
    batch_size: int = 32

    # Validation
    energy_weight: float = 1.0
    force_weight: float = 10.0
    stress_weight: float = 0.0


# ============================================================================
# Data Extraction Functions
# ============================================================================

def extract_snapshots_from_simulation(
    simulation_result: Dict[str, Any],
    include_forces: bool = True,
    include_stress: bool = False
) -> List[TrainingSnapshot]:
    """
    Extract training snapshots from a DFT simulation result.

    Parses the simulation output and extracts structures along with their
    computed energies and forces.

    Args:
        simulation_result: SimulationResult dict with parsed output
        include_forces: Whether to extract atomic forces
        include_stress: Whether to extract stress tensor

    Returns:
        List of TrainingSnapshot objects

    Example:
        >>> result = simulation_result.to_dict()
        >>> snapshots = extract_snapshots_from_simulation(result)
        >>> print(f"Extracted {len(snapshots)} snapshots")
    """
    snapshots = []

    # Check if result contains trajectory data (MD) or single point
    output = simulation_result.get("output", {})

    if "trajectory" in output:
        # MD trajectory: multiple snapshots
        trajectory = output["trajectory"]

        for i, frame in enumerate(trajectory.get("frames", [])):
            snapshot = TrainingSnapshot(
                structure_id=f"{simulation_result['id']}_frame_{i}",
                atoms=frame.get("atoms", []),
                lattice_vectors=frame.get("lattice_vectors", []),
                energy=frame.get("energy", 0.0),
                forces=frame.get("forces", []) if include_forces else [],
                stress=frame.get("stress") if include_stress else None,
                metadata={
                    "source": "trajectory",
                    "frame_index": i,
                    "temperature": frame.get("temperature"),
                    "pressure": frame.get("pressure"),
                }
            )
            snapshots.append(snapshot)

    else:
        # Single-point calculation
        snapshot = TrainingSnapshot(
            structure_id=str(simulation_result["id"]),
            atoms=output.get("atoms", []),
            lattice_vectors=output.get("lattice_vectors", []),
            energy=output.get("energy", 0.0),
            forces=output.get("forces", []) if include_forces else [],
            stress=output.get("stress") if include_stress else None,
            metadata={
                "source": "single_point",
            }
        )
        snapshots.append(snapshot)

    logger.info(f"Extracted {len(snapshots)} snapshots from simulation {simulation_result['id']}")

    return snapshots


def build_training_dataset(
    simulation_results: List[Dict[str, Any]],
    min_force_threshold: Optional[float] = None,
    max_force_threshold: Optional[float] = None,
) -> TrainingDataset:
    """
    Build a complete training dataset from multiple simulation results.

    Aggregates snapshots from multiple DFT simulations and computes
    dataset statistics.

    Args:
        simulation_results: List of SimulationResult dicts
        min_force_threshold: Filter out snapshots with max force below this (eV/Å)
        max_force_threshold: Filter out snapshots with max force above this (eV/Å)

    Returns:
        TrainingDataset object

    Example:
        >>> results = [job1.result.to_dict(), job2.result.to_dict()]
        >>> dataset = build_training_dataset(results)
        >>> print(f"Dataset: {dataset.num_snapshots} snapshots, {len(dataset.elements)} elements")
    """
    all_snapshots = []
    all_elements = set()

    # Extract snapshots from all simulations
    for result in simulation_results:
        snapshots = extract_snapshots_from_simulation(result)

        # Apply force filtering if requested
        for snapshot in snapshots:
            if snapshot.forces:
                max_force = max(
                    np.linalg.norm(f) for f in snapshot.forces
                ) if snapshot.forces else 0.0

                if min_force_threshold and max_force < min_force_threshold:
                    continue
                if max_force_threshold and max_force > max_force_threshold:
                    continue

            all_snapshots.append(snapshot)

            # Collect elements
            for atom in snapshot.atoms:
                all_elements.add(atom.get("element", "Unknown"))

    # Compute dataset statistics
    energies = [s.energy for s in all_snapshots]
    all_forces = []
    for s in all_snapshots:
        if s.forces:
            all_forces.extend([np.linalg.norm(f) for f in s.forces])

    dataset = TrainingDataset(
        snapshots=all_snapshots,
        elements=sorted(list(all_elements)),
        num_snapshots=len(all_snapshots),
        energy_range=(min(energies) if energies else 0.0, max(energies) if energies else 0.0),
        force_max=max(all_forces) if all_forces else 0.0,
        metadata={
            "num_source_simulations": len(simulation_results),
            "avg_energy": np.mean(energies) if energies else 0.0,
            "avg_force_magnitude": np.mean(all_forces) if all_forces else 0.0,
        }
    )

    logger.info(
        f"Built training dataset: {dataset.num_snapshots} snapshots, "
        f"{len(dataset.elements)} elements, "
        f"energy range: {dataset.energy_range}, "
        f"max force: {dataset.force_max:.3f} eV/Å"
    )

    return dataset


def export_dataset_to_file(
    dataset: TrainingDataset,
    output_path: Path,
    format: str = "extxyz"
) -> Path:
    """
    Export training dataset to file format.

    Supports formats:
    - "extxyz": Extended XYZ format (compatible with most ML potential trainers)
    - "json": JSON format (for inspection and debugging)
    - "lammps_data": LAMMPS data format

    Args:
        dataset: TrainingDataset to export
        output_path: Path to output file
        format: Output format

    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "extxyz":
        _export_to_extxyz(dataset, output_path)
    elif format == "json":
        _export_to_json(dataset, output_path)
    elif format == "lammps_data":
        _export_to_lammps_data(dataset, output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Exported dataset to {output_path} ({format} format)")

    return output_path


def _export_to_extxyz(dataset: TrainingDataset, output_path: Path):
    """Export to Extended XYZ format."""
    with open(output_path, 'w') as f:
        for snapshot in dataset.snapshots:
            # Header line with number of atoms
            num_atoms = len(snapshot.atoms)
            f.write(f"{num_atoms}\n")

            # Properties line
            lattice_str = " ".join(
                f"{v:.6f}" for row in snapshot.lattice_vectors for v in row
            )
            properties_line = (
                f"Lattice=\"{lattice_str}\" "
                f"Properties=species:S:1:pos:R:3:forces:R:3 "
                f"energy={snapshot.energy:.8f} "
                f"config_type=dft "
                f"pbc=\"T T T\"\n"
            )
            f.write(properties_line)

            # Atom lines
            for atom, force in zip(snapshot.atoms, snapshot.forces or []):
                element = atom.get("element", "X")
                pos = atom.get("position", [0, 0, 0])
                force_vec = force if force else [0, 0, 0]

                f.write(
                    f"{element} "
                    f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{force_vec[0]:.6f} {force_vec[1]:.6f} {force_vec[2]:.6f}\n"
                )


def _export_to_json(dataset: TrainingDataset, output_path: Path):
    """Export to JSON format."""
    data = {
        "metadata": dataset.metadata,
        "elements": dataset.elements,
        "num_snapshots": dataset.num_snapshots,
        "energy_range": dataset.energy_range,
        "force_max": dataset.force_max,
        "snapshots": [
            {
                "structure_id": s.structure_id,
                "atoms": s.atoms,
                "lattice_vectors": s.lattice_vectors,
                "energy": s.energy,
                "forces": s.forces,
                "stress": s.stress,
                "metadata": s.metadata,
            }
            for s in dataset.snapshots
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def _export_to_lammps_data(dataset: TrainingDataset, output_path: Path):
    """Export to LAMMPS data format (simplified, first snapshot only)."""
    if not dataset.snapshots:
        raise ValueError("Dataset contains no snapshots")

    # Use first snapshot as template
    snapshot = dataset.snapshots[0]

    with open(output_path, 'w') as f:
        f.write("# LAMMPS data file from NANO-OS training dataset\n\n")
        f.write(f"{len(snapshot.atoms)} atoms\n")
        f.write(f"{len(dataset.elements)} atom types\n\n")

        # Box bounds (from lattice vectors)
        # Simplified: assume orthorhombic
        lattice = snapshot.lattice_vectors
        f.write(f"0.0 {lattice[0][0]:.6f} xlo xhi\n")
        f.write(f"0.0 {lattice[1][1]:.6f} ylo yhi\n")
        f.write(f"0.0 {lattice[2][2]:.6f} zlo zhi\n\n")

        # Atom type mapping
        element_to_type = {elem: i+1 for i, elem in enumerate(dataset.elements)}

        # Atoms
        f.write("Atoms\n\n")
        for i, atom in enumerate(snapshot.atoms):
            element = atom.get("element", "X")
            atom_type = element_to_type.get(element, 1)
            pos = atom.get("position", [0, 0, 0])
            f.write(f"{i+1} {atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")


# ============================================================================
# Training Functions (Stub Implementations)
# ============================================================================

def train_ml_potential(
    config: PotentialConfig,
    dataset: TrainingDataset,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Train an ML interatomic potential.

    **NOTE**: This is currently a stub implementation that simulates training
    and creates dummy potential files. Full implementation would integrate
    with actual ML potential trainers (e.g., LAMMPS SNAP, NequIP, MACE).

    Args:
        config: Training configuration
        dataset: Training dataset
        output_dir: Directory to save potential files

    Returns:
        Training result dict with metrics and file paths

    Example:
        >>> config = PotentialConfig(
        ...     name="snap_silicon_v1",
        ...     descriptor_type="SNAP",
        ...     elements=["Si"],
        ...     descriptor_params={"twojmax": 6, "rcutfac": 4.67}
        ... )
        >>> dataset = build_training_dataset(simulation_results)
        >>> result = train_ml_potential(config, dataset, Path("/models/potentials"))
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Training ML potential: {config.name} ({config.descriptor_type}) "
        f"on {dataset.num_snapshots} snapshots"
    )

    # Split dataset into train/val
    num_train = int(dataset.num_snapshots * config.train_fraction)
    train_snapshots = dataset.snapshots[:num_train]
    val_snapshots = dataset.snapshots[num_train:]

    logger.info(f"Train: {len(train_snapshots)}, Val: {len(val_snapshots)}")

    # Stub training implementation
    # In production, this would call the actual trainer
    if config.descriptor_type == "SNAP":
        result = _train_snap_potential(config, train_snapshots, val_snapshots, output_dir)
    elif config.descriptor_type in ["NequIP", "MACE"]:
        result = _train_neural_network_potential(config, train_snapshots, val_snapshots, output_dir)
    else:
        # Generic stub
        result = _train_generic_potential(config, train_snapshots, val_snapshots, output_dir)

    logger.info(
        f"Training complete: Train RMSE = {result['metrics']['train_energy_rmse']:.4f} eV, "
        f"Val RMSE = {result['metrics']['val_energy_rmse']:.4f} eV"
    )

    return result


def _train_snap_potential(
    config: PotentialConfig,
    train_data: List[TrainingSnapshot],
    val_data: List[TrainingSnapshot],
    output_dir: Path
) -> Dict[str, Any]:
    """Stub implementation for SNAP potential training."""
    # In production, would call LAMMPS SNAP trainer or FitSNAP

    # Create dummy coefficient file
    coeff_file = output_dir / f"{config.name}.snapcoeff"
    param_file = output_dir / f"{config.name}.snapparam"

    with open(coeff_file, 'w') as f:
        f.write(f"# SNAP coefficients for {config.name}\n")
        f.write("# Trained on DFT data from NANO-OS\n")
        f.write("# NOTE: This is a stub file for development\n")

    with open(param_file, 'w') as f:
        f.write(f"# SNAP parameters for {config.name}\n")
        f.write(f"# Descriptor: SNAP\n")
        f.write(f"# Elements: {' '.join(config.elements)}\n")
        for key, value in config.descriptor_params.items():
            f.write(f"{key} {value}\n")

    # Compute mock metrics
    metrics = {
        "train_energy_rmse": 0.005,  # eV/atom
        "train_force_mae": 0.05,  # eV/Å
        "val_energy_rmse": 0.008,
        "val_force_mae": 0.07,
        "num_coefficients": 100,
    }

    return {
        "status": "success",
        "potential_type": "SNAP",
        "files": {
            "coefficients": str(coeff_file),
            "parameters": str(param_file),
        },
        "metrics": metrics,
        "num_train": len(train_data),
        "num_val": len(val_data),
    }


def _train_neural_network_potential(
    config: PotentialConfig,
    train_data: List[TrainingSnapshot],
    val_data: List[TrainingSnapshot],
    output_dir: Path
) -> Dict[str, Any]:
    """Stub implementation for neural network potential (NequIP, MACE) training."""
    # In production, would call NequIP/MACE trainer

    # Create dummy model file
    model_file = output_dir / f"{config.name}.pth"
    config_file = output_dir / f"{config.name}_config.json"

    with open(model_file, 'w') as f:
        f.write(f"# Neural network potential weights for {config.name}\n")
        f.write("# NOTE: This is a stub file for development\n")

    with open(config_file, 'w') as f:
        json.dump({
            "model_type": config.descriptor_type,
            "elements": config.elements,
            "hyperparameters": config.descriptor_params,
        }, f, indent=2)

    # Compute mock metrics
    metrics = {
        "train_energy_rmse": 0.003,  # eV/atom
        "train_force_mae": 0.03,  # eV/Å
        "val_energy_rmse": 0.006,
        "val_force_mae": 0.05,
        "num_parameters": 50000,
    }

    return {
        "status": "success",
        "potential_type": config.descriptor_type,
        "files": {
            "model": str(model_file),
            "config": str(config_file),
        },
        "metrics": metrics,
        "num_train": len(train_data),
        "num_val": len(val_data),
    }


def _train_generic_potential(
    config: PotentialConfig,
    train_data: List[TrainingSnapshot],
    val_data: List[TrainingSnapshot],
    output_dir: Path
) -> Dict[str, Any]:
    """Generic stub implementation for other potential types."""
    potential_file = output_dir / f"{config.name}.potential"

    with open(potential_file, 'w') as f:
        f.write(f"# ML potential for {config.name}\n")
        f.write(f"# Type: {config.descriptor_type}\n")
        f.write("# NOTE: This is a stub file for development\n")

    metrics = {
        "train_energy_rmse": 0.010,
        "train_force_mae": 0.08,
        "val_energy_rmse": 0.015,
        "val_force_mae": 0.12,
    }

    return {
        "status": "success",
        "potential_type": config.descriptor_type,
        "files": {
            "potential": str(potential_file),
        },
        "metrics": metrics,
        "num_train": len(train_data),
        "num_val": len(val_data),
    }

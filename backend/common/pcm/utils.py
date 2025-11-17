"""
Phase-change memory utilities.

Session 26: PCM Vertical
"""

import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def compute_phase_energy_difference(
    crystalline_job: Dict[str, Any],
    amorphous_job: Dict[str, Any]
) -> float:
    """
    Compute energy difference between crystalline and amorphous phases.

    Args:
        crystalline_job: DFT job results for crystalline phase
        amorphous_job: DFT job results for amorphous phase

    Returns:
        Energy difference in eV/atom
    """
    E_cryst = crystalline_job.get("energy", 0.0)
    E_amorph = amorphous_job.get("energy", 0.0)
    N_cryst = crystalline_job.get("num_atoms", 1)
    N_amorph = amorphous_job.get("num_atoms", 1)

    # Per atom energies
    e_cryst = E_cryst / N_cryst
    e_amorph = E_amorph / N_amorph

    delta_E = e_amorph - e_cryst

    logger.info(f"Phase energy difference: {delta_E:.4f} eV/atom")
    return float(delta_E)


def estimate_switching_energy(
    delta_E: float,
    volume: float,
    device_geometry: Dict[str, Any]
) -> float:
    """
    Estimate switching energy for PCM device.

    E_switch ≈ ΔE * N_atoms_active

    Args:
        delta_E: Phase energy difference (eV/atom)
        volume: Unit cell volume (Å³)
        device_geometry: Device dimensions

    Returns:
        Switching energy in pJ
    """
    # Stub: simplified calculation
    device_volume_nm3 = device_geometry.get("volume_nm3", 100)  # nm³
    atoms_per_nm3 = 50  # Typical for GST

    N_atoms = device_volume_nm3 * atoms_per_nm3
    E_switch_eV = abs(delta_E) * N_atoms

    # Convert eV to pJ (1 eV = 0.16 pJ)
    E_switch_pJ = E_switch_eV * 0.16

    logger.info(f"Estimated switching energy: {E_switch_pJ:.2f} pJ")
    return float(E_switch_pJ)


def estimate_resistivity_contrast(
    crystalline_structure: Dict[str, Any],
    amorphous_structure: Dict[str, Any]
) -> float:
    """
    Estimate resistivity contrast between phases.

    Based on bandgap difference (stub).

    Args:
        crystalline_structure: Crystalline structure info
        amorphous_structure: Amorphous structure info

    Returns:
        Resistivity contrast ratio (ρ_amorph / ρ_cryst)
    """
    logger.warning("Using stubbed resistivity estimation")

    # Stub: use bandgap as proxy
    # Real implementation would compute conductivity from electronic structure

    gap_cryst = crystalline_structure.get("bandgap", 0.5)
    gap_amorph = amorphous_structure.get("bandgap", 1.5)

    # Simple exponential model
    contrast = np.exp(gap_amorph - gap_cryst)

    logger.info(f"Estimated resistivity contrast: {contrast:.1f}x")
    return float(contrast)

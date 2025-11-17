"""
Battery materials utilities.

Session 23: Battery Materials Vertical
"""

import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def estimate_average_voltage(
    structure_charged: Dict[str, Any],
    structure_discharged: Dict[str, Any],
    redox_species: str = "Li"
) -> float:
    """
    Estimate average voltage from energy difference.

    V_avg = -(E_discharged - E_charged) / (n * e)

    Args:
        structure_charged: Charged state structure with energy
        structure_discharged: Discharged state structure with energy
        redox_species: Redox species (Li, Na, K)

    Returns:
        Average voltage in V
    """
    E_charged = structure_charged.get("energy", 0.0)
    E_discharged = structure_discharged.get("energy", 0.0)
    n_charged = structure_charged.get("num_atoms", 1)
    n_discharged = structure_discharged.get("num_atoms", 1)

    # Number of transferred ions
    delta_n = abs(n_discharged - n_charged)

    if delta_n == 0:
        logger.warning("No change in number of atoms, cannot estimate voltage")
        return 0.0

    # Energy difference in eV
    delta_E = E_discharged - E_charged

    # Average voltage (eV per ion = V)
    V_avg = -delta_E / delta_n

    logger.info(f"Estimated average voltage: {V_avg:.3f} V")
    return float(V_avg)


def estimate_capacity(formula: str, redox_species: str = "Li") -> float:
    """
    Estimate theoretical capacity.

    Capacity (mAh/g) = (n * F) / (3.6 * M)
    where n = number of electrons transferred
          F = Faraday constant (96485 C/mol)
          M = molecular weight (g/mol)

    Args:
        formula: Chemical formula
        redox_species: Redox species

    Returns:
        Theoretical capacity in mAh/g
    """
    # Stub: simplified calculation
    # Real implementation would parse formula and compute MW
    logger.warning("Using stubbed capacity estimation")

    # Placeholder values
    n_electrons = 1  # Assume 1e- transfer per redox species
    molar_mass = 100  # g/mol (placeholder)

    F = 96485  # C/mol
    capacity = (n_electrons * F) / (3.6 * molar_mass)

    logger.info(f"Estimated capacity: {capacity:.2f} mAh/g")
    return float(capacity)


def estimate_volume_change(
    structure_charged: Dict[str, Any],
    structure_discharged: Dict[str, Any]
) -> float:
    """
    Estimate volume change on (de)lithiation.

    ΔV/V = (V_discharged - V_charged) / V_charged * 100%

    Args:
        structure_charged: Charged state structure
        structure_discharged: Discharged state structure

    Returns:
        Volume change percentage
    """
    V_charged = structure_charged.get("volume", 100.0)
    V_discharged = structure_discharged.get("volume", 100.0)

    volume_change_percent = (V_discharged - V_charged) / V_charged * 100

    logger.info(f"Estimated volume change: {volume_change_percent:.2f}%")
    return float(volume_change_percent)

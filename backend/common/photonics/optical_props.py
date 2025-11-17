"""
Optical properties extraction for photonics.

Session 22: Photonics Vertical
"""

import logging
from typing import Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def estimate_refractive_index(structure: Dict[str, Any]) -> float:
    """
    Estimate refractive index from structure properties.

    Stub implementation using simple correlations.

    Args:
        structure: Structure dictionary with composition and properties

    Returns:
        Estimated refractive index
    """
    # Stub: use bandgap and density correlations
    bandgap = structure.get("bandgap", 2.0)
    density = structure.get("density", 2.5)  # g/cm^3

    # Simple empirical formula (not physical, just placeholder)
    # Real implementation would use Kramers-Kronig, DFT-calculated ε(ω)
    n = 1.5 + 0.5 * np.sqrt(density) / (1 + bandgap)

    logger.info(f"Estimated refractive index: {n:.3f}")
    return float(n)


def build_waveguide_mode_problem(
    geometry_params: Dict[str, Any],
    wavelength_nm: float,
    refractive_index: float
) -> Dict[str, Any]:
    """
    Build waveguide mode problem description for EM solver.

    Args:
        geometry_params: Waveguide geometry (thickness, width, etc.)
        wavelength_nm: Operating wavelength in nm
        refractive_index: Material refractive index

    Returns:
        Problem description for EM solver
    """
    return {
        "type": "waveguide_mode_solver",
        "geometry": geometry_params,
        "wavelength_nm": wavelength_nm,
        "material_index": refractive_index,
        "substrate_index": geometry_params.get("substrate_index", 1.45),
        "cladding_index": geometry_params.get("cladding_index", 1.0)
    }


def build_photonic_crystal_problem(
    geometry_params: Dict[str, Any],
    wavelength_range: Dict[str, float]
) -> Dict[str, Any]:
    """
    Build photonic crystal band diagram problem.

    Args:
        geometry_params: PC geometry (period, hole radius, slab thickness)
        wavelength_range: Wavelength range for band diagram

    Returns:
        Problem description for EM solver
    """
    return {
        "type": "photonic_crystal_bands",
        "geometry": geometry_params,
        "wavelength_min_nm": wavelength_range.get("min", 400),
        "wavelength_max_nm": wavelength_range.get("max", 1600),
        "lattice_type": geometry_params.get("lattice_type", "square")
    }

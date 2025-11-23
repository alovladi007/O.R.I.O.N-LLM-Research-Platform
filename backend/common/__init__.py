"""
Common backend utilities for ORION platform.

This module provides shared functionality including:
- Structure file parsers (CIF, POSCAR, XYZ)
- Simulation engines
- Provenance tracking
"""

from .structure_parsers import parse_structure, parse_cif, parse_poscar, parse_xyz

__all__ = [
    "parse_structure",
    "parse_cif",
    "parse_poscar",
    "parse_xyz",
]

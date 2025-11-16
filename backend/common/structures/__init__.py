"""
Crystal structure parsing and manipulation utilities for ORION platform.

This module provides tools for parsing, converting, and analyzing crystallographic
structure files in multiple formats (CIF, POSCAR, XYZ).

Key Components:
- parsers: Core parsing and export functionality for structure formats
- InternalStructureModel: Normalized internal representation of crystal structures

Example:
    from backend.common.structures import parse_structure, StructureFormat

    with open('structure.cif', 'r') as f:
        structure = parse_structure(f.read(), StructureFormat.CIF)

    print(f"Formula: {structure.formula}")
    print(f"Volume: {structure.volume} Å³")
"""

from .parsers import (
    InternalStructureModel,
    StructureFormat,
    parse_structure,
    to_cif,
    to_poscar,
    to_xyz,
)

__all__ = [
    "InternalStructureModel",
    "StructureFormat",
    "parse_structure",
    "to_cif",
    "to_poscar",
    "to_xyz",
]

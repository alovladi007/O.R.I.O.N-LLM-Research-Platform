"""
Simulation Engine Abstraction Layer
====================================

This module provides an abstraction layer for different simulation engines.

Available Engines:
- MockSimulationEngine: For testing and development
- QuantumEspressoEngine: DFT calculations using Quantum ESPRESSO
- VASPEngine: DFT calculations using VASP (planned)
- LAMMPSEngine: Molecular dynamics using LAMMPS (planned)

Usage:
    from backend.common.engines import get_engine

    engine_class = get_engine("QE")
    engine = engine_class()
    engine.setup(structure, parameters)
    results = engine.run()
    engine.cleanup()
"""

from backend.common.engines.base import SimulationEngine
from backend.common.engines.registry import (
    get_engine,
    ENGINE_REGISTRY,
    list_engines,
    is_engine_available,
    register_engine,
)

__all__ = [
    "SimulationEngine",
    "get_engine",
    "ENGINE_REGISTRY",
    "list_engines",
    "is_engine_available",
    "register_engine",
]

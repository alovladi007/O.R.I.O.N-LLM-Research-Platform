"""
Simulation Engine Registry
===========================

Central registry for all simulation engines.

This module maintains a mapping of engine names to engine classes and provides
a factory function to instantiate engines by name.

Usage:
    from backend.common.engines.registry import get_engine

    # Get engine class
    engine_class = get_engine("QE")

    # Instantiate and use
    engine = engine_class()
    engine.setup(structure, parameters)
    results = engine.run()
    engine.cleanup()
"""

from typing import Type, Dict, Optional
import logging

from backend.common.engines.base import SimulationEngine
from backend.common.engines.mock import MockSimulationEngine
from backend.common.engines.qe import QuantumEspressoEngine
from backend.common.engines.mesoscale import PhaseFieldEngine, MonteCarloEngine
from backend.common.engines.continuum import FEMEngine, FVMEngine

logger = logging.getLogger(__name__)


# Engine registry mapping
ENGINE_REGISTRY: Dict[str, Optional[Type[SimulationEngine]]] = {
    # Mock engine for testing
    "MOCK": MockSimulationEngine,

    # Quantum ESPRESSO
    "QE": QuantumEspressoEngine,
    "QUANTUM_ESPRESSO": QuantumEspressoEngine,
    "PWX": QuantumEspressoEngine,

    # VASP (placeholder for future implementation)
    "VASP": None,

    # LAMMPS (placeholder for future implementation)
    "LAMMPS": None,
    "LAMMPS_MD": None,

    # Gaussian (placeholder for future implementation)
    "GAUSSIAN": None,
    "G16": None,

    # CP2K (placeholder for future implementation)
    "CP2K": None,

    # ORCA (placeholder for future implementation)
    "ORCA": None,

    # SIESTA (placeholder for future implementation)
    "SIESTA": None,

    # ABINIT (placeholder for future implementation)
    "ABINIT": None,

    # CASTEP (placeholder for future implementation)
    "CASTEP": None,
}


def get_engine(engine_name: str) -> Type[SimulationEngine]:
    """
    Get engine class by name.

    Args:
        engine_name: Name of the engine (case-insensitive).
            Supported: "MOCK", "QE", "QUANTUM_ESPRESSO", "VASP", "LAMMPS", etc.

    Returns:
        Engine class (subclass of SimulationEngine)

    Raises:
        ValueError: If engine is not supported or not yet implemented

    Examples:
        >>> engine_class = get_engine("QE")
        >>> engine = engine_class()
        >>> engine.setup(structure, parameters)

        >>> # Using mock engine
        >>> mock_engine = get_engine("MOCK")()
    """
    # Normalize engine name
    engine_name_upper = engine_name.upper().strip()

    # Check if engine exists in registry
    if engine_name_upper not in ENGINE_REGISTRY:
        available_engines = ", ".join(sorted(ENGINE_REGISTRY.keys()))
        raise ValueError(
            f"Engine '{engine_name}' not found in registry. "
            f"Available engines: {available_engines}"
        )

    # Get engine class
    engine_class = ENGINE_REGISTRY[engine_name_upper]

    # Check if implemented
    if engine_class is None:
        implemented_engines = ", ".join(
            sorted(k for k, v in ENGINE_REGISTRY.items() if v is not None)
        )
        raise ValueError(
            f"Engine '{engine_name}' is not yet implemented. "
            f"Implemented engines: {implemented_engines}. "
            f"Use 'MOCK' for testing or contribute an implementation!"
        )

    logger.debug(f"Retrieved engine class: {engine_class.__name__} for '{engine_name}'")

    return engine_class


def list_engines(include_unimplemented: bool = False) -> Dict[str, bool]:
    """
    List all registered engines.

    Args:
        include_unimplemented: If True, include engines that are registered but
            not yet implemented (placeholder entries)

    Returns:
        Dictionary mapping engine names to implementation status (True if implemented)

    Examples:
        >>> engines = list_engines()
        >>> print(engines)
        {'MOCK': True, 'QE': True, 'QUANTUM_ESPRESSO': True}

        >>> all_engines = list_engines(include_unimplemented=True)
        >>> print(all_engines)
        {'MOCK': True, 'QE': True, 'VASP': False, 'LAMMPS': False, ...}
    """
    if include_unimplemented:
        return {name: (cls is not None) for name, cls in ENGINE_REGISTRY.items()}
    else:
        return {name: True for name, cls in ENGINE_REGISTRY.items() if cls is not None}


def is_engine_available(engine_name: str) -> bool:
    """
    Check if an engine is available (registered and implemented).

    Args:
        engine_name: Name of the engine (case-insensitive)

    Returns:
        True if engine is available and implemented, False otherwise

    Examples:
        >>> is_engine_available("QE")
        True
        >>> is_engine_available("VASP")
        False
        >>> is_engine_available("NONEXISTENT")
        False
    """
    engine_name_upper = engine_name.upper().strip()
    return (
        engine_name_upper in ENGINE_REGISTRY and
        ENGINE_REGISTRY[engine_name_upper] is not None
    )


def register_engine(engine_name: str, engine_class: Type[SimulationEngine]) -> None:
    """
    Register a new engine or override existing registration.

    This allows for dynamic engine registration, useful for:
    - Plugin systems
    - Custom engine implementations
    - Testing with custom engines

    Args:
        engine_name: Name to register the engine under (will be converted to uppercase)
        engine_class: Engine class (must be subclass of SimulationEngine)

    Raises:
        TypeError: If engine_class is not a subclass of SimulationEngine

    Examples:
        >>> from backend.common.engines.base import SimulationEngine
        >>>
        >>> class MyCustomEngine(SimulationEngine):
        ...     def setup(self, structure, parameters):
        ...         pass
        ...     def run(self, progress_callback=None):
        ...         pass
        ...     def cleanup(self):
        ...         pass
        >>>
        >>> register_engine("CUSTOM", MyCustomEngine)
        >>> engine = get_engine("CUSTOM")()
    """
    if not issubclass(engine_class, SimulationEngine):
        raise TypeError(
            f"Engine class must be a subclass of SimulationEngine. "
            f"Got: {engine_class}"
        )

    engine_name_upper = engine_name.upper().strip()

    if engine_name_upper in ENGINE_REGISTRY:
        logger.warning(
            f"Overriding existing engine registration: {engine_name_upper}"
        )

    ENGINE_REGISTRY[engine_name_upper] = engine_class
    logger.info(f"Registered engine: {engine_name_upper} -> {engine_class.__name__}")


# ============================================================================
# Multi-scale Simulation Engine Registries (Session 11)
# ============================================================================

# Mesoscale engine registry
MESOSCALE_ENGINE_REGISTRY: Dict[str, Optional[Type[SimulationEngine]]] = {
    # Phase Field Modeling
    "PHASE_FIELD": PhaseFieldEngine,
    "PF": PhaseFieldEngine,
    "PHASE_FIELD_MODEL": PhaseFieldEngine,

    # Monte Carlo
    "MONTE_CARLO": MonteCarloEngine,
    "MC": MonteCarloEngine,
    "POTTS": MonteCarloEngine,

    # Kinetic Monte Carlo (placeholder for future implementation)
    "KMC": None,
    "KINETIC_MONTE_CARLO": None,
}


# Continuum engine registry
CONTINUUM_ENGINE_REGISTRY: Dict[str, Optional[Type[SimulationEngine]]] = {
    # Finite Element Method
    "FEM": FEMEngine,
    "FINITE_ELEMENT": FEMEngine,
    "FINITE_ELEMENT_METHOD": FEMEngine,

    # Finite Volume Method
    "FVM": FVMEngine,
    "FINITE_VOLUME": FVMEngine,
    "FINITE_VOLUME_METHOD": FVMEngine,

    # Boundary Element Method (placeholder for future implementation)
    "BEM": None,
    "BOUNDARY_ELEMENT": None,
    "BOUNDARY_ELEMENT_METHOD": None,
}


def get_mesoscale_engine(engine_name: str) -> Type[SimulationEngine]:
    """
    Get mesoscale engine class by name.

    Args:
        engine_name: Name of the mesoscale engine (case-insensitive).
            Supported: "PHASE_FIELD", "MONTE_CARLO", "KMC", etc.

    Returns:
        Engine class (subclass of SimulationEngine)

    Raises:
        ValueError: If engine is not supported or not yet implemented

    Examples:
        >>> engine_class = get_mesoscale_engine("PHASE_FIELD")
        >>> engine = engine_class()
        >>> engine.setup(structure, parameters)
    """
    engine_name_upper = engine_name.upper().strip()

    if engine_name_upper not in MESOSCALE_ENGINE_REGISTRY:
        available_engines = ", ".join(sorted(MESOSCALE_ENGINE_REGISTRY.keys()))
        raise ValueError(
            f"Mesoscale engine '{engine_name}' not found in registry. "
            f"Available engines: {available_engines}"
        )

    engine_class = MESOSCALE_ENGINE_REGISTRY[engine_name_upper]

    if engine_class is None:
        implemented_engines = ", ".join(
            sorted(k for k, v in MESOSCALE_ENGINE_REGISTRY.items() if v is not None)
        )
        raise ValueError(
            f"Mesoscale engine '{engine_name}' is not yet implemented. "
            f"Implemented engines: {implemented_engines}"
        )

    logger.debug(f"Retrieved mesoscale engine class: {engine_class.__name__} for '{engine_name}'")

    return engine_class


def get_continuum_engine(engine_name: str) -> Type[SimulationEngine]:
    """
    Get continuum engine class by name.

    Args:
        engine_name: Name of the continuum engine (case-insensitive).
            Supported: "FEM", "FVM", "BEM", etc.

    Returns:
        Engine class (subclass of SimulationEngine)

    Raises:
        ValueError: If engine is not supported or not yet implemented

    Examples:
        >>> engine_class = get_continuum_engine("FEM")
        >>> engine = engine_class()
        >>> engine.setup(structure, parameters)
    """
    engine_name_upper = engine_name.upper().strip()

    if engine_name_upper not in CONTINUUM_ENGINE_REGISTRY:
        available_engines = ", ".join(sorted(CONTINUUM_ENGINE_REGISTRY.keys()))
        raise ValueError(
            f"Continuum engine '{engine_name}' not found in registry. "
            f"Available engines: {available_engines}"
        )

    engine_class = CONTINUUM_ENGINE_REGISTRY[engine_name_upper]

    if engine_class is None:
        implemented_engines = ", ".join(
            sorted(k for k, v in CONTINUUM_ENGINE_REGISTRY.items() if v is not None)
        )
        raise ValueError(
            f"Continuum engine '{engine_name}' is not yet implemented. "
            f"Implemented engines: {implemented_engines}"
        )

    logger.debug(f"Retrieved continuum engine class: {engine_class.__name__} for '{engine_name}'")

    return engine_class


def list_mesoscale_engines(include_unimplemented: bool = False) -> Dict[str, bool]:
    """
    List all registered mesoscale engines.

    Args:
        include_unimplemented: If True, include engines that are registered but
            not yet implemented

    Returns:
        Dictionary mapping engine names to implementation status
    """
    if include_unimplemented:
        return {name: (cls is not None) for name, cls in MESOSCALE_ENGINE_REGISTRY.items()}
    else:
        return {name: True for name, cls in MESOSCALE_ENGINE_REGISTRY.items() if cls is not None}


def list_continuum_engines(include_unimplemented: bool = False) -> Dict[str, bool]:
    """
    List all registered continuum engines.

    Args:
        include_unimplemented: If True, include engines that are registered but
            not yet implemented

    Returns:
        Dictionary mapping engine names to implementation status
    """
    if include_unimplemented:
        return {name: (cls is not None) for name, cls in CONTINUUM_ENGINE_REGISTRY.items()}
    else:
        return {name: True for name, cls in CONTINUUM_ENGINE_REGISTRY.items() if cls is not None}

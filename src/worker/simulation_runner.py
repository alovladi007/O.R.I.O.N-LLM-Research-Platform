"""
Simulation Execution Engine (Legacy Module)
============================================

DEPRECATED: This module is maintained for backwards compatibility only.

New code should use the engine abstraction layer:
    from backend.common.engines import get_engine

This module now re-exports engines from the new location:
- backend.common.engines.base: SimulationEngine base class
- backend.common.engines.mock: MockSimulationEngine
- backend.common.engines.qe: QuantumEspressoEngine
- backend.common.engines.registry: get_engine, ENGINE_REGISTRY

Migration guide:
    OLD: from src.worker.simulation_runner import MockSimulationEngine
    NEW: from backend.common.engines.mock import MockSimulationEngine

    OLD: from src.worker.simulation_runner import get_engine
    NEW: from backend.common.engines import get_engine
"""

import logging
import warnings

# Re-export from new location for backwards compatibility
from backend.common.engines.mock import MockSimulationEngine, run_mock_simulation
from backend.common.engines.registry import get_engine, ENGINE_REGISTRY

logger = logging.getLogger(__name__)

# Emit deprecation warning
warnings.warn(
    "src.worker.simulation_runner is deprecated. "
    "Use backend.common.engines instead. "
    "See module docstring for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

# Export legacy names for compatibility
__all__ = [
    "MockSimulationEngine",
    "run_mock_simulation",
    "get_engine",
    "ENGINE_REGISTRY",
]

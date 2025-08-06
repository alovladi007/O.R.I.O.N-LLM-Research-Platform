"""
ORION Core Module
================

Main orchestration and coordination module for the ORION platform.
"""

from .orion_system import ORIONSystem
from .config_manager import ConfigManager
from .monitoring import PerformanceMonitor, BottleneckAnalyzer
from .exceptions import ORIONException, ValidationError, SimulationError

__all__ = [
    "ORIONSystem",
    "ConfigManager",
    "PerformanceMonitor",
    "BottleneckAnalyzer",
    "ORIONException",
    "ValidationError",
    "SimulationError",
]

__version__ = "1.0.0"
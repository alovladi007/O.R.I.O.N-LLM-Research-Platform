"""
ORION Core Module
================

Main orchestration and coordination module for the ORION platform.
"""

from .orion_system import ORIONSystem
from .config_manager import ConfigManager
from .monitoring import PerformanceMonitor, BottleneckAnalyzer
from .advanced_monitoring import AdvancedBottleneckAnalyzer, PerformanceMetrics
from .physics_validator import PhysicsSanityChecker
from .exceptions import (
    ORIONException, 
    ValidationError, 
    SimulationError,
    ConfigurationError,
    KnowledgeGraphError,
    RAGError,
    LLMError,
    DatabaseError,
    AuthenticationError,
    APIError,
    ProcessingError,
    CacheError,
    StorageError,
    NetworkError
)

__all__ = [
    # Core system
    "ORIONSystem",
    "ConfigManager",
    
    # Monitoring
    "PerformanceMonitor",
    "BottleneckAnalyzer",
    "AdvancedBottleneckAnalyzer",
    "PerformanceMetrics",
    
    # Physics validation
    "PhysicsSanityChecker",
    
    # Exceptions
    "ORIONException",
    "ValidationError",
    "SimulationError",
    "ConfigurationError",
    "KnowledgeGraphError",
    "RAGError",
    "LLMError",
    "DatabaseError",
    "AuthenticationError",
    "APIError",
    "ProcessingError",
    "CacheError",
    "StorageError",
    "NetworkError",
]

__version__ = "1.0.0"
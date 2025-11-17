"""
Base instrument adapter for lab equipment integration.

Defines the abstract interface that all instrument adapters must implement.

Session 21: Lab Integration & Experiment Management
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExperimentExecutionResult:
    """
    Result of experiment execution on an instrument.

    Attributes:
        success: Whether experiment completed successfully
        results: Experiment results (measured properties, files, etc.)
        external_job_id: External job ID from instrument control system
        error_message: Error message if failed
        metadata: Additional metadata
    """
    success: bool
    results: Dict[str, Any]
    external_job_id: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class InstrumentAdapter(ABC):
    """
    Abstract base class for instrument adapters.

    All instrument adapters must implement this interface to enable
    plug-and-play integration with different lab equipment.

    Adapters can support various protocols:
    - REST API
    - OPC-UA (industrial automation)
    - SSH/command-line
    - Direct Python libraries
    - Mock/simulation

    Example implementation:
        >>> class MyCustomAdapter(InstrumentAdapter):
        ...     def connect(self):
        ...         # Connect to instrument
        ...         pass
        ...
        ...     def execute_experiment(self, experiment):
        ...         # Execute experiment on instrument
        ...         # Return results
        ...         pass
        ...
        ...     def disconnect(self):
        ...         # Clean up
        ...         pass
    """

    def __init__(self, connection_info: Dict[str, Any]):
        """
        Initialize instrument adapter.

        Args:
            connection_info: Connection configuration
                (endpoint, credentials, etc.)
        """
        self.connection_info = connection_info
        self.is_connected = False
        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to instrument.

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close connection to instrument.

        Should handle cleanup gracefully even if not connected.
        """
        pass

    @abstractmethod
    def execute_experiment(
        self,
        experiment_type: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExperimentExecutionResult:
        """
        Execute an experiment on the instrument.

        Args:
            experiment_type: Type of experiment (synthesis, measurement, etc.)
            parameters: Experiment parameters
            metadata: Optional experiment metadata

        Returns:
            ExperimentExecutionResult with results and status

        Raises:
            RuntimeError: If execution fails
        """
        pass

    @abstractmethod
    def check_status(self) -> Dict[str, Any]:
        """
        Check instrument status.

        Returns:
            Status information (online, busy, error, etc.)
        """
        pass

    def validate_parameters(
        self,
        experiment_type: str,
        parameters: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate experiment parameters.

        Can be overridden by subclasses for type-specific validation.

        Args:
            experiment_type: Type of experiment
            parameters: Parameters to validate

        Returns:
            (is_valid, error_message)
        """
        # Default: accept all parameters
        return True, None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

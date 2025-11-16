"""
Base Simulation Engine
======================

Abstract base class for all simulation engines.

All engines must implement:
- setup(): Initialize engine with structure and parameters
- run(): Execute simulation and return results
- cleanup(): Clean up temporary files

This ensures a consistent interface across all simulation backends.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SimulationEngine(ABC):
    """
    Abstract base class for simulation engines.

    All simulation engines (VASP, Quantum ESPRESSO, LAMMPS, etc.) must inherit
    from this class and implement the required methods.

    Attributes:
        structure: Atomic structure data
        parameters: Simulation parameters
        work_dir: Working directory for simulation files
        logger: Logger instance
    """

    def __init__(self):
        """Initialize base engine."""
        self.structure = None
        self.parameters = None
        self.work_dir = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def setup(self, structure: Dict[str, Any], parameters: Dict[str, Any]) -> None:
        """
        Initialize engine with structure and parameters.

        This method should:
        1. Validate input structure and parameters
        2. Create working directory
        3. Generate input files for the simulation
        4. Prepare any additional resources

        Args:
            structure: Atomic structure data containing:
                - atoms: List of atomic species
                - positions: Atomic coordinates
                - cell: Unit cell parameters (if applicable)
                - Additional metadata (formula, n_atoms, etc.)
            parameters: Simulation parameters specific to the engine

        Raises:
            ValueError: If structure or parameters are invalid
            IOError: If file creation fails
        """
        pass

    @abstractmethod
    def run(self, progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        """
        Run simulation and return results.

        This method should:
        1. Execute the simulation
        2. Monitor progress (if possible)
        3. Parse output files
        4. Extract relevant results
        5. Handle errors and convergence issues

        Args:
            progress_callback: Optional callback function for progress updates.
                Called with (progress: float, step: str) where progress is 0-1.

        Returns:
            Dictionary containing simulation results with keys:
                - summary: Main results (energy, forces, etc.)
                - convergence_reached: Boolean indicating convergence
                - quality_score: Optional quality metric (0-1)
                - metadata: Additional metadata (engine version, etc.)

        Raises:
            RuntimeError: If simulation fails
            ValueError: If output parsing fails
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up temporary files.

        This method should:
        1. Remove temporary working directory
        2. Delete intermediate files (unless debugging)
        3. Release any held resources

        Note: Should NOT delete important output files that may be needed later.
        """
        pass

    def validate_structure(self, structure: Dict[str, Any]) -> bool:
        """
        Validate structure data (common validation logic).

        Args:
            structure: Structure data to validate

        Returns:
            True if valid

        Raises:
            ValueError: If structure is invalid
        """
        if not structure:
            raise ValueError("Structure data is required")

        if not isinstance(structure, dict):
            raise ValueError("Structure must be a dictionary")

        # Basic validation - subclasses can override for engine-specific checks
        return True

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters (common validation logic).

        Args:
            parameters: Parameters to validate

        Returns:
            True if valid

        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary")

        # Basic validation - subclasses can override for engine-specific checks
        return True

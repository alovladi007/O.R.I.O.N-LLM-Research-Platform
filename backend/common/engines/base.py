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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """
    Return value from a raw engine subprocess invocation.

    Engines (LAMMPS / QE / custom) produce these from their
    ``run_subprocess``-style helpers. Higher layers (the Celery task
    base class in Phase 2) inspect ``success`` to decide state-machine
    transitions and ``stdout`` / ``stderr`` for log capture.

    ``returncode == 0`` ≙ ``success``, but we keep both for tools that
    differentiate "ran, exited clean, output parsed as a failure" from
    "exited with non-zero code". ``timed_out`` is True only when the
    engine wrapper hit its wall-clock limit.
    """

    success: bool
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }


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

    def execute_command(
        self,
        cmd: List[str],
        run_dir: Path,
        *,
        execution_kind: str = "local",
        cpus: int = 1,
        gpus: int = 0,
        memory_gb: Optional[int] = None,
        walltime_minutes: Optional[int] = None,
        queue: Optional[str] = None,
        account: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> "ExecutionResult":
        """Run *cmd* via the execution backend and return an :class:`ExecutionResult`.

        Thin wrapper that:
        1. Resolves the backend (``local`` or ``slurm``) via
           :func:`backend.common.execution.get_execution_backend`.
        2. Uses :func:`backend.common.execution.local.sync_execute` to run
           submit → poll-loop → terminal state under ``asyncio.run``.
        3. Reads stdout/stderr from the backend's run-dir files and
           packages them into :class:`ExecutionResult` so legacy parsers
           keep working.

        Session 2.3: engines that called ``subprocess.run`` / ``Popen``
        directly use this helper instead. Behavior on `local` is
        equivalent; `slurm` gets it for free.
        """
        from backend.common.execution import (
            JobState,
            Resources,
            get_execution_backend,
            sync_execute,
        )

        backend = get_execution_backend(execution_kind)
        resources = Resources(
            cpus=cpus,
            gpus=gpus,
            memory_gb=memory_gb,
            walltime_minutes=walltime_minutes,
            queue=queue,
            account=account,
            env=env or {},
        )
        state = sync_execute(backend, cmd, Path(run_dir), resources)

        # Read stdout/stderr from the files the backend wrote.
        stdout_path = Path(run_dir) / "stdout.txt"
        stderr_path = Path(run_dir) / "stderr.txt"
        stdout = stdout_path.read_text(errors="replace") if stdout_path.exists() else ""
        stderr = stderr_path.read_text(errors="replace") if stderr_path.exists() else ""

        if state == JobState.COMPLETED:
            return ExecutionResult(success=True, returncode=0, stdout=stdout, stderr=stderr)
        if state == JobState.CANCELLED:
            return ExecutionResult(
                success=False, returncode=-15, stdout=stdout, stderr=stderr or "cancelled",
            )
        # FAILED or any unknown terminal → non-zero returncode marker.
        return ExecutionResult(
            success=False, returncode=1, stdout=stdout, stderr=stderr or f"state={state.value}",
        )

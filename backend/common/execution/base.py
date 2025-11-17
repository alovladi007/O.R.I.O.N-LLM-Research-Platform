"""
Execution backend abstraction for HPC and cloud.

Session 27: HPC and Cloud Scaling
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class JobResources:
    """Resource requirements for a job."""
    nodes: int = 1
    cores_per_node: int = 1
    memory_gb: Optional[int] = None
    walltime_hours: Optional[float] = None
    gpus: Optional[int] = None


@dataclass
class JobStatus:
    """Job status information."""
    external_job_id: str
    status: str  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ExecutionBackend(ABC):
    """
    Abstract base class for execution backends.

    Supports:
    - Local execution
    - HPC schedulers (SLURM, PBS, etc.)
    - Cloud APIs
    - SSH-based remote execution
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize execution backend.

        Args:
            config: Backend-specific configuration
        """
        self.config = config
        logger.info(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def submit(
        self,
        job_script: str,
        resources: JobResources,
        job_name: str,
        working_dir: str
    ) -> str:
        """
        Submit a job for execution.

        Args:
            job_script: Job script content (shell script)
            resources: Resource requirements
            job_name: Job name
            working_dir: Working directory path

        Returns:
            External job ID

        Raises:
            RuntimeError: If submission fails
        """
        pass

    @abstractmethod
    def check_status(self, external_job_id: str) -> JobStatus:
        """
        Check job status.

        Args:
            external_job_id: External job ID

        Returns:
            Job status information

        Raises:
            RuntimeError: If status check fails
        """
        pass

    @abstractmethod
    def cancel(self, external_job_id: str) -> bool:
        """
        Cancel a running job.

        Args:
            external_job_id: External job ID

        Returns:
            True if cancelled successfully

        Raises:
            RuntimeError: If cancellation fails
        """
        pass

    @abstractmethod
    def fetch_results(
        self,
        external_job_id: str,
        dest_path: str
    ) -> bool:
        """
        Fetch job results to local destination.

        Args:
            external_job_id: External job ID
            dest_path: Destination path for results

        Returns:
            True if fetch successful

        Raises:
            RuntimeError: If fetch fails
        """
        pass

    def validate_resources(self, resources: JobResources) -> tuple[bool, Optional[str]]:
        """
        Validate resource requirements.

        Can be overridden by subclasses.

        Args:
            resources: Resource requirements

        Returns:
            (is_valid, error_message)
        """
        if resources.nodes < 1:
            return False, "Number of nodes must be >= 1"
        if resources.cores_per_node < 1:
            return False, "Cores per node must be >= 1"
        return True, None

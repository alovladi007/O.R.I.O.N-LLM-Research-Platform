"""
Local execution backend.

Session 27: HPC and Cloud Scaling
"""

import logging
import subprocess
import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any

from .base import ExecutionBackend, JobResources, JobStatus

logger = logging.getLogger(__name__)


class LocalExecutionBackend(ExecutionBackend):
    """
    Local execution backend using subprocess.

    Runs jobs directly on the local machine.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize local backend.

        Config options:
        - max_concurrent_jobs: Maximum number of concurrent jobs (default: 4)
        """
        super().__init__(config)
        self.max_concurrent = config.get("max_concurrent_jobs", 4)
        self.running_jobs: Dict[str, subprocess.Popen] = {}

    def submit(
        self,
        job_script: str,
        resources: JobResources,
        job_name: str,
        working_dir: str
    ) -> str:
        """
        Submit job locally via subprocess.

        Args:
            job_script: Shell script to execute
            resources: Resource requirements (mostly ignored for local)
            job_name: Job name
            working_dir: Working directory

        Returns:
            External job ID (UUID)
        """
        # Check concurrent job limit
        active_jobs = [jid for jid, proc in self.running_jobs.items() if proc.poll() is None]
        if len(active_jobs) >= self.max_concurrent:
            raise RuntimeError(
                f"Maximum concurrent jobs ({self.max_concurrent}) reached. "
                f"Currently running: {len(active_jobs)}"
            )

        # Generate job ID
        external_job_id = f"local_{uuid.uuid4().hex[:12]}"

        # Create working directory
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        # Write script to file
        script_path = work_path / "job_script.sh"
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(job_script)

        os.chmod(script_path, 0o755)

        # Submit via subprocess
        logger.info(f"Submitting local job {external_job_id}: {job_name}")

        stdout_file = work_path / "stdout.txt"
        stderr_file = work_path / "stderr.txt"

        with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
            proc = subprocess.Popen(
                ["/bin/bash", str(script_path)],
                cwd=str(work_path),
                stdout=stdout,
                stderr=stderr,
                env=os.environ.copy()
            )

        self.running_jobs[external_job_id] = proc

        logger.info(f"Job {external_job_id} submitted with PID {proc.pid}")
        return external_job_id

    def check_status(self, external_job_id: str) -> JobStatus:
        """
        Check local job status.

        Args:
            external_job_id: Job ID

        Returns:
            Job status
        """
        if external_job_id not in self.running_jobs:
            logger.warning(f"Job {external_job_id} not found in running jobs")
            return JobStatus(
                external_job_id=external_job_id,
                status="UNKNOWN"
            )

        proc = self.running_jobs[external_job_id]
        exit_code = proc.poll()

        if exit_code is None:
            # Still running
            status = "RUNNING"
        elif exit_code == 0:
            status = "COMPLETED"
        else:
            status = "FAILED"

        return JobStatus(
            external_job_id=external_job_id,
            status=status,
            exit_code=exit_code
        )

    def cancel(self, external_job_id: str) -> bool:
        """
        Cancel local job.

        Args:
            external_job_id: Job ID

        Returns:
            True if cancelled
        """
        if external_job_id not in self.running_jobs:
            logger.warning(f"Job {external_job_id} not found")
            return False

        proc = self.running_jobs[external_job_id]
        if proc.poll() is None:  # Still running
            logger.info(f"Terminating job {external_job_id} (PID {proc.pid})")
            proc.terminate()
            time.sleep(1)
            if proc.poll() is None:  # Still alive, kill it
                proc.kill()
            return True
        else:
            logger.info(f"Job {external_job_id} already terminated")
            return True

    def fetch_results(
        self,
        external_job_id: str,
        dest_path: str
    ) -> bool:
        """
        Fetch results (no-op for local, files are already local).

        Args:
            external_job_id: Job ID
            dest_path: Destination path

        Returns:
            True (always succeeds for local)
        """
        logger.info(f"Local job {external_job_id}: results already in working directory")
        return True

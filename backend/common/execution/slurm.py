"""
SLURM execution backend.

Session 27: HPC and Cloud Scaling
"""

import logging
import subprocess
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional

from .base import ExecutionBackend, JobResources, JobStatus

logger = logging.getLogger(__name__)


class SlurmExecutionBackend(ExecutionBackend):
    """
    SLURM execution backend using sbatch, squeue, scancel.

    Submits jobs to a SLURM cluster via command-line tools.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SLURM backend.

        Config options:
        - partition: SLURM partition (default: None)
        - account: SLURM account (default: None)
        - qos: Quality of Service (default: None)
        - modules: List of modules to load (default: [])
        - working_dir_base: Base directory for job files (default: ~/slurm_jobs)
        """
        super().__init__(config)
        self.partition = config.get("partition")
        self.account = config.get("account")
        self.qos = config.get("qos")
        self.modules = config.get("modules", [])
        self.working_dir_base = Path(config.get("working_dir_base", "~/slurm_jobs")).expanduser()
        self.working_dir_base.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        job_script: str,
        resources: JobResources,
        job_name: str,
        working_dir: str
    ) -> str:
        """
        Submit job to SLURM via sbatch.

        Args:
            job_script: Shell script to execute
            resources: Resource requirements
            job_name: Job name
            working_dir: Working directory

        Returns:
            External job ID (SLURM job ID)
        """
        # Create working directory
        work_path = Path(working_dir)
        work_path.mkdir(parents=True, exist_ok=True)

        # Generate SLURM batch script
        batch_script = self._generate_slurm_script(
            job_script=job_script,
            resources=resources,
            job_name=job_name,
            working_dir=str(work_path)
        )

        # Write batch script to file
        script_path = work_path / "slurm_job.sh"
        with open(script_path, "w") as f:
            f.write(batch_script)

        # Submit via sbatch
        logger.info(f"Submitting SLURM job: {job_name}")

        try:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                cwd=str(work_path),
                capture_output=True,
                text=True,
                check=True
            )

            # Parse job ID from sbatch output
            # Expected format: "Submitted batch job 12345"
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not match:
                raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")

            slurm_job_id = match.group(1)
            logger.info(f"Job submitted successfully. SLURM job ID: {slurm_job_id}")

            return slurm_job_id

        except subprocess.CalledProcessError as e:
            logger.error(f"sbatch failed: {e.stderr}")
            raise RuntimeError(f"Failed to submit SLURM job: {e.stderr}")

    def check_status(self, external_job_id: str) -> JobStatus:
        """
        Check SLURM job status via squeue.

        Args:
            external_job_id: SLURM job ID

        Returns:
            Job status
        """
        try:
            # Query job status with squeue
            result = subprocess.run(
                ["squeue", "-j", external_job_id, "--format=%T,%r"],
                capture_output=True,
                text=True,
                check=False  # Don't raise on non-zero exit (job might be completed)
            )

            # If job not found in queue, check sacct for completed jobs
            if result.returncode != 0 or not result.stdout.strip():
                return self._check_completed_job(external_job_id)

            # Parse squeue output
            # Format: STATE,REASON
            # Skip header line
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                # Job not in queue, check sacct
                return self._check_completed_job(external_job_id)

            state_line = lines[1].strip()
            state, reason = state_line.split(",", 1) if "," in state_line else (state_line, "")

            # Map SLURM states to our JobStatus
            status_map = {
                "PENDING": "PENDING",
                "RUNNING": "RUNNING",
                "SUSPENDED": "RUNNING",  # Treat suspended as running
                "COMPLETING": "RUNNING",
                "COMPLETED": "COMPLETED",
                "CANCELLED": "CANCELLED",
                "FAILED": "FAILED",
                "TIMEOUT": "FAILED",
                "NODE_FAIL": "FAILED",
                "PREEMPTED": "FAILED",
                "OUT_OF_MEMORY": "FAILED"
            }

            job_status = status_map.get(state, "UNKNOWN")

            return JobStatus(
                external_job_id=external_job_id,
                status=job_status,
                reason=reason if reason else None
            )

        except Exception as e:
            logger.error(f"Failed to check SLURM job status: {e}")
            return JobStatus(
                external_job_id=external_job_id,
                status="UNKNOWN",
                error_message=str(e)
            )

    def _check_completed_job(self, external_job_id: str) -> JobStatus:
        """
        Check completed job status using sacct.

        Args:
            external_job_id: SLURM job ID

        Returns:
            Job status
        """
        try:
            result = subprocess.run(
                ["sacct", "-j", external_job_id, "--format=State,ExitCode", "--noheader", "--parsable2"],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0 or not result.stdout.strip():
                logger.warning(f"Job {external_job_id} not found in squeue or sacct")
                return JobStatus(
                    external_job_id=external_job_id,
                    status="UNKNOWN"
                )

            # Parse sacct output
            # Format: State|ExitCode
            lines = result.stdout.strip().split("\n")
            if not lines:
                return JobStatus(external_job_id=external_job_id, status="UNKNOWN")

            # Take first line (main job, not job steps)
            state_line = lines[0].strip()
            parts = state_line.split("|")
            state = parts[0].strip()
            exit_code_str = parts[1].strip() if len(parts) > 1 else "0:0"

            # Parse exit code (format: "exit_code:signal")
            exit_code = 0
            if ":" in exit_code_str:
                exit_code = int(exit_code_str.split(":")[0])

            # Map state to our status
            if state == "COMPLETED":
                job_status = "COMPLETED" if exit_code == 0 else "FAILED"
            elif state in ["FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"]:
                job_status = "FAILED"
            elif state == "CANCELLED":
                job_status = "CANCELLED"
            else:
                job_status = "UNKNOWN"

            return JobStatus(
                external_job_id=external_job_id,
                status=job_status,
                exit_code=exit_code if exit_code != 0 else None
            )

        except Exception as e:
            logger.error(f"Failed to check completed job with sacct: {e}")
            return JobStatus(
                external_job_id=external_job_id,
                status="UNKNOWN",
                error_message=str(e)
            )

    def cancel(self, external_job_id: str) -> bool:
        """
        Cancel SLURM job via scancel.

        Args:
            external_job_id: SLURM job ID

        Returns:
            True if cancelled successfully
        """
        try:
            logger.info(f"Cancelling SLURM job {external_job_id}")

            result = subprocess.run(
                ["scancel", external_job_id],
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                logger.info(f"Job {external_job_id} cancelled successfully")
                return True
            else:
                logger.warning(f"scancel returned non-zero exit code: {result.stderr}")
                # Job might already be finished, check status
                status = self.check_status(external_job_id)
                if status.status in ["COMPLETED", "FAILED", "CANCELLED"]:
                    logger.info(f"Job {external_job_id} already finished")
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to cancel SLURM job: {e}")
            return False

    def fetch_results(
        self,
        external_job_id: str,
        dest_path: str
    ) -> bool:
        """
        Fetch results from SLURM job working directory.

        For local SLURM clusters, files are already accessible.
        For remote clusters, this would use scp/rsync.

        Args:
            external_job_id: SLURM job ID
            dest_path: Destination path

        Returns:
            True if successful
        """
        # For local SLURM, files are already in the working directory
        # For remote SLURM, would need to implement scp/rsync here
        logger.info(
            f"SLURM job {external_job_id}: assuming results already in working directory "
            f"(local cluster). For remote clusters, implement scp/rsync."
        )
        return True

    def _generate_slurm_script(
        self,
        job_script: str,
        resources: JobResources,
        job_name: str,
        working_dir: str
    ) -> str:
        """
        Generate SLURM batch script with directives.

        Args:
            job_script: User's job script
            resources: Resource requirements
            job_name: Job name
            working_dir: Working directory

        Returns:
            Complete SLURM batch script
        """
        lines = ["#!/bin/bash"]

        # SLURM directives
        lines.append(f"#SBATCH --job-name={job_name}")
        lines.append(f"#SBATCH --output={working_dir}/slurm-%j.out")
        lines.append(f"#SBATCH --error={working_dir}/slurm-%j.err")

        # Resources
        lines.append(f"#SBATCH --nodes={resources.nodes}")
        lines.append(f"#SBATCH --ntasks-per-node={resources.cores_per_node}")

        if resources.memory_gb:
            lines.append(f"#SBATCH --mem={resources.memory_gb}G")

        if resources.walltime_hours:
            # Convert hours to HH:MM:SS
            hours = int(resources.walltime_hours)
            minutes = int((resources.walltime_hours - hours) * 60)
            lines.append(f"#SBATCH --time={hours:02d}:{minutes:02d}:00")

        if resources.gpus:
            lines.append(f"#SBATCH --gres=gpu:{resources.gpus}")

        # Optional directives
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")

        if self.account:
            lines.append(f"#SBATCH --account={self.account}")

        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")

        lines.append("")

        # Load modules
        if self.modules:
            lines.append("# Load modules")
            for module in self.modules:
                lines.append(f"module load {module}")
            lines.append("")

        # Change to working directory
        lines.append(f"cd {working_dir}")
        lines.append("")

        # User's job script
        lines.append("# User job script")
        lines.append(job_script)

        return "\n".join(lines)

"""
Celery Task Definitions
=======================

This module defines all Celery tasks for NANO-OS simulations.

Tasks:
- run_dummy_job: Dummy simulation for testing (Session 1)
- run_simulation_job: Main simulation task handler
- update_job_status: Update job status in database

All tasks include:
- Automatic retries on failure
- Progress tracking
- Error handling and logging
- Database integration
"""

import logging
import time
import uuid
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

from celery import Task
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .celery_app import celery_app
from .simulation_runner import run_mock_simulation, MockSimulationEngine
from src.api.models.simulation import SimulationJob, SimulationResult, JobStatus
from src.api.database import async_session_factory

logger = logging.getLogger(__name__)


class DatabaseTask(Task):
    """
    Base task class with database session management.

    Provides helper methods for async database operations from sync Celery tasks.
    """

    def get_db_session(self) -> AsyncSession:
        """Get a new database session."""
        if not async_session_factory:
            raise RuntimeError("Database not initialized")
        return async_session_factory()

    async def _update_job_status_async(
        self,
        job_id: str,
        status: str,
        **kwargs
    ) -> None:
        """Update job status in database (async)."""
        async with self.get_db_session() as db:
            try:
                job_uuid = uuid.UUID(job_id)

                # Build update values
                update_values = {
                    "status": JobStatus[status],
                    "updated_at": datetime.utcnow(),
                }

                # Add optional fields
                if "progress" in kwargs:
                    update_values["progress"] = kwargs["progress"]
                if "current_step" in kwargs:
                    update_values["current_step"] = kwargs["current_step"]
                if "error_message" in kwargs:
                    update_values["error_message"] = kwargs["error_message"]
                if "worker_id" in kwargs:
                    update_values["worker_id"] = kwargs["worker_id"]
                if "started_at" in kwargs:
                    update_values["started_at"] = kwargs["started_at"]
                if "finished_at" in kwargs:
                    update_values["finished_at"] = kwargs["finished_at"]
                if "retry_count" in kwargs:
                    update_values["retry_count"] = kwargs["retry_count"]

                # Update job
                stmt = (
                    update(SimulationJob)
                    .where(SimulationJob.id == job_uuid)
                    .values(**update_values)
                )
                await db.execute(stmt)
                await db.commit()

                logger.info(f"Updated job {job_id} status to {status}")

            except Exception as e:
                logger.error(f"Failed to update job status: {e}")
                await db.rollback()
                raise

    async def _get_job_async(self, job_id: str) -> Optional[SimulationJob]:
        """Get job from database (async)."""
        async with self.get_db_session() as db:
            try:
                job_uuid = uuid.UUID(job_id)
                stmt = select(SimulationJob).where(SimulationJob.id == job_uuid)
                result = await db.execute(stmt)
                job = result.scalar_one_or_none()
                return job
            except Exception as e:
                logger.error(f"Failed to get job: {e}")
                return None

    async def _create_simulation_result_async(
        self,
        job_id: str,
        summary: Dict[str, Any],
        convergence_reached: bool = True,
        quality_score: Optional[float] = None,
        artifacts_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create simulation result in database (async)."""
        async with self.get_db_session() as db:
            try:
                job_uuid = uuid.UUID(job_id)

                result = SimulationResult(
                    simulation_job_id=job_uuid,
                    summary=summary,
                    convergence_reached=convergence_reached,
                    quality_score=quality_score,
                    artifacts_path=artifacts_path,
                    metadata=metadata or {},
                )

                db.add(result)
                await db.commit()

                logger.info(f"Created simulation result for job {job_id}")

            except Exception as e:
                logger.error(f"Failed to create simulation result: {e}")
                await db.rollback()
                raise


@celery_app.task(
    name="run_dummy_job",
    base=DatabaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def run_dummy_job(self, job_id: str) -> Dict[str, Any]:
    """
    Dummy simulation job for Session 1 testing.

    This task simulates a simple job that:
    1. Updates status to RUNNING
    2. Sleeps for a few seconds (simulating work)
    3. Updates progress periodically
    4. Completes successfully

    Args:
        job_id: UUID of the simulation job

    Returns:
        Dictionary with job results
    """
    logger.info(f"Starting dummy job {job_id}")

    try:
        # Update status to RUNNING
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "RUNNING",
                started_at=datetime.utcnow(),
                worker_id=self.request.id,
                current_step="Initializing",
            )
        )

        # Simulate work with progress updates
        steps = ["Initializing", "Processing", "Analyzing", "Finalizing"]
        total_duration = 10  # seconds
        step_duration = total_duration / len(steps)

        for i, step in enumerate(steps):
            logger.info(f"Dummy job {job_id}: {step}")

            # Update progress
            progress = (i + 1) / len(steps)
            asyncio.run(
                self._update_job_status_async(
                    job_id,
                    "RUNNING",
                    progress=progress,
                    current_step=step,
                )
            )

            # Simulate work
            time.sleep(step_duration)

        # Create dummy result
        summary = {
            "status": "success",
            "energy": -42.123,
            "band_gap": 1.5,
            "message": "Dummy simulation completed successfully",
        }

        asyncio.run(
            self._create_simulation_result_async(
                job_id,
                summary=summary,
                convergence_reached=True,
                quality_score=0.95,
            )
        )

        # Update status to COMPLETED
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "COMPLETED",
                finished_at=datetime.utcnow(),
                progress=1.0,
                current_step="Completed",
            )
        )

        logger.info(f"Dummy job {job_id} completed successfully")

        return {
            "job_id": job_id,
            "status": "success",
            "summary": summary,
        }

    except Exception as e:
        logger.error(f"Dummy job {job_id} failed: {e}", exc_info=True)

        # Update status to FAILED
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "FAILED",
                error_message=str(e),
                finished_at=datetime.utcnow(),
            )
        )

        # Retry on failure
        raise self.retry(exc=e)


@celery_app.task(
    name="run_simulation_job",
    base=DatabaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def run_simulation_job(self, job_id: str) -> Dict[str, Any]:
    """
    Main simulation job task handler.

    This task:
    1. Fetches job from database
    2. Validates job parameters
    3. Runs the simulation (using mock or real engine)
    4. Stores results in database
    5. Updates job status

    Args:
        job_id: UUID of the simulation job

    Returns:
        Dictionary with job results

    Raises:
        Retry: On recoverable errors
    """
    logger.info(f"Starting simulation job {job_id}")

    try:
        # Get job from database
        job = asyncio.run(self._get_job_async(job_id))

        if not job:
            logger.error(f"Job {job_id} not found in database")
            raise ValueError(f"Job {job_id} not found")

        # Check if job is already terminal
        if job.is_terminal:
            logger.warning(f"Job {job_id} is already in terminal state: {job.status}")
            return {"job_id": job_id, "status": "already_terminal"}

        # Update status to RUNNING
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "RUNNING",
                started_at=datetime.utcnow(),
                worker_id=self.request.id,
                current_step="Starting simulation",
                retry_count=self.request.retries,
            )
        )

        # Extract job parameters
        structure_data = job.structure.structure_data if hasattr(job, "structure") else {}
        parameters = job.parameters or {}
        engine = job.engine

        logger.info(f"Running {engine} simulation for job {job_id}")

        # Update progress: Initializing
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "RUNNING",
                progress=0.1,
                current_step="Initializing simulation engine",
            )
        )

        # Run simulation (mock for now)
        # In future sessions, this will route to real engines (VASP, QE, LAMMPS)
        simulation_result = asyncio.run(
            run_mock_simulation(
                structure=structure_data,
                parameters=parameters,
                engine=engine,
                job_id=job_id,
                progress_callback=lambda p, s: asyncio.run(
                    self._update_job_status_async(
                        job_id,
                        "RUNNING",
                        progress=0.1 + 0.8 * p,  # Reserve 0-10% for init, 90-100% for finalization
                        current_step=s,
                    )
                ),
            )
        )

        # Update progress: Finalizing
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "RUNNING",
                progress=0.9,
                current_step="Finalizing results",
            )
        )

        # Store results in database
        asyncio.run(
            self._create_simulation_result_async(
                job_id,
                summary=simulation_result["summary"],
                convergence_reached=simulation_result.get("convergence_reached", True),
                quality_score=simulation_result.get("quality_score"),
                artifacts_path=simulation_result.get("artifacts_path"),
                metadata=simulation_result.get("metadata", {}),
            )
        )

        # Update status to COMPLETED
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "COMPLETED",
                finished_at=datetime.utcnow(),
                progress=1.0,
                current_step="Completed",
            )
        )

        logger.info(f"Simulation job {job_id} completed successfully")

        return {
            "job_id": job_id,
            "status": "success",
            "summary": simulation_result["summary"],
        }

    except Exception as e:
        logger.error(f"Simulation job {job_id} failed: {e}", exc_info=True)

        # Update status to FAILED
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "FAILED",
                error_message=str(e),
                finished_at=datetime.utcnow(),
                retry_count=self.request.retries,
            )
        )

        # Retry on failure (up to max_retries)
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying job {job_id} (attempt {self.request.retries + 1}/{self.max_retries})")
            raise self.retry(exc=e)
        else:
            logger.error(f"Job {job_id} failed after {self.max_retries} retries")
            raise


@celery_app.task(
    name="update_job_status",
    base=DatabaseTask,
    bind=True,
    max_retries=5,
    default_retry_delay=10,
)
def update_job_status(
    self,
    job_id: str,
    status: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Update simulation job status in database.

    This is a utility task for updating job status from other parts of the system.

    Args:
        job_id: UUID of the simulation job
        status: New status (PENDING, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED)
        **kwargs: Additional fields to update (progress, current_step, error_message, etc.)

    Returns:
        Dictionary with update status
    """
    try:
        asyncio.run(self._update_job_status_async(job_id, status, **kwargs))

        return {
            "job_id": job_id,
            "status": status,
            "updated": True,
        }

    except Exception as e:
        logger.error(f"Failed to update job status: {e}", exc_info=True)

        # Retry on failure
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        else:
            raise


@celery_app.task(name="cancel_job", base=DatabaseTask, bind=True)
def cancel_job(self, job_id: str) -> Dict[str, Any]:
    """
    Cancel a running simulation job.

    This task:
    1. Checks if job is cancellable
    2. Terminates the running simulation
    3. Updates job status to CANCELLED

    Args:
        job_id: UUID of the simulation job

    Returns:
        Dictionary with cancellation status
    """
    try:
        # Get job
        job = asyncio.run(self._get_job_async(job_id))

        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Check if job is cancellable
        if job.is_terminal:
            logger.warning(f"Cannot cancel job {job_id}: already in terminal state {job.status}")
            return {
                "job_id": job_id,
                "cancelled": False,
                "reason": f"Job already in terminal state: {job.status}",
            }

        # Update status to CANCELLED
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "CANCELLED",
                finished_at=datetime.utcnow(),
                current_step="Cancelled by user",
            )
        )

        # TODO: In future, terminate the actual simulation process if running

        logger.info(f"Job {job_id} cancelled successfully")

        return {
            "job_id": job_id,
            "cancelled": True,
        }

    except Exception as e:
        logger.error(f"Failed to cancel job {job_id}: {e}", exc_info=True)
        raise

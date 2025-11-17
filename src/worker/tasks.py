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
import os
import tempfile
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from celery import Task
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .celery_app import celery_app
# Import new engine system
from backend.common.engines import get_engine
from backend.common.engines.mock import run_mock_simulation  # For backwards compatibility
from src.api.models.simulation import SimulationJob, SimulationResult, JobStatus
from src.api.models.provenance import EntityType, EventType
from src.api.database import async_session_factory
from backend.common.provenance import record_provenance, get_system_info, get_code_version

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

    def _create_job_artifacts(
        self,
        job_id: str,
        results: Dict[str, Any],
        job_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create mock artifacts for a simulation job.

        Creates a temporary directory with mock output files including:
        - simulation.log: Mock simulation log file
        - results.json: JSON file with results
        - structure.cif: Mock structure file (if applicable)

        Args:
            job_id: UUID of the simulation job
            results: Simulation results dictionary
            job_data: Optional job metadata

        Returns:
            Path to the artifacts directory
        """
        try:
            # Create temporary directory for artifacts
            # In production, this would be in a shared storage (MinIO, S3, etc.)
            artifacts_dir = os.path.join(tempfile.gettempdir(), "orion_simulations", str(job_id))
            Path(artifacts_dir).mkdir(parents=True, exist_ok=True)

            # Create simulation.log
            log_path = os.path.join(artifacts_dir, "simulation.log")
            with open(log_path, 'w') as f:
                f.write(f"ORION Simulation Log\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(f"Job ID: {job_id}\n")
                f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
                f.write(f"Engine: {results.get('summary', {}).get('engine', 'MOCK')}\n\n")
                f.write(f"Structure Information:\n")
                f.write(f"  Formula: {results.get('summary', {}).get('formula', 'Unknown')}\n")
                f.write(f"  N atoms: {results.get('summary', {}).get('n_atoms', 0)}\n")
                f.write(f"  Dimensionality: {results.get('summary', {}).get('dimensionality', 3)}\n\n")
                f.write(f"Calculation Results:\n")
                f.write(f"  Total Energy: {results.get('summary', {}).get('total_energy', 0):.6f} eV\n")
                f.write(f"  Energy/Atom: {results.get('summary', {}).get('energy_per_atom', 0):.6f} eV\n")
                f.write(f"  Band Gap: {results.get('summary', {}).get('bandgap', 0):.4f} eV\n")
                f.write(f"  Is Stable: {results.get('summary', {}).get('is_stable', False)}\n\n")

                convergence = results.get('summary', {}).get('convergence', {})
                f.write(f"Convergence:\n")
                f.write(f"  Reached: {convergence.get('reached', False)}\n")
                f.write(f"  Iterations: {convergence.get('iterations', 0)}\n")
                f.write(f"  Final Error: {convergence.get('final_error', 0):.2e}\n\n")

                f.write(f"Forces:\n")
                forces = results.get('summary', {}).get('forces', {})
                f.write(f"  Max Force: {forces.get('max_force', 0):.6f} eV/Å\n")
                f.write(f"  RMS Force: {forces.get('rms_force', 0):.6f} eV/Å\n\n")

                f.write(f"Simulation completed successfully!\n")

            # Create results.json
            json_path = os.path.join(artifacts_dir, "results.json")
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            # Create mock OUTCAR or output file
            output_path = os.path.join(artifacts_dir, "output.txt")
            with open(output_path, 'w') as f:
                f.write(f"ORION Mock Simulation Output\n")
                f.write(f"Job ID: {job_id}\n")
                f.write(f"Status: COMPLETED\n")
                f.write(f"Results written to {json_path}\n")

            logger.info(f"Created artifacts for job {job_id} in {artifacts_dir}")
            return artifacts_dir

        except Exception as e:
            logger.error(f"Failed to create artifacts for job {job_id}: {e}")
            # Don't fail the job if artifact creation fails
            return None


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
    Main simulation job task handler (Session 4 enhanced version).

    This task implements the complete job lifecycle:
    1. Validates job is in QUEUED status
    2. Transitions to RUNNING with started_at timestamp
    3. Executes simulation with progress tracking
    4. Creates artifacts (logs, results files)
    5. Stores results in database with artifacts_path
    6. Transitions to COMPLETED with finished_at timestamp
    7. Handles failures with proper error messages

    Status transitions:
    - QUEUED → RUNNING (on start)
    - RUNNING → COMPLETED (on success)
    - RUNNING → FAILED (on error)

    Args:
        job_id: UUID of the simulation job

    Returns:
        Dictionary with job results and status

    Raises:
        Retry: On recoverable errors (up to max_retries)
        ValueError: If job not found or in invalid state
    """
    logger.info(f"[Job {job_id}] Starting simulation task (worker: {self.request.id})")
    started_at = datetime.utcnow()

    try:
        # ===== STEP 1: Fetch and validate job =====
        job = asyncio.run(self._get_job_async(job_id))

        if not job:
            error_msg = f"Job {job_id} not found in database"
            logger.error(f"[Job {job_id}] {error_msg}")
            raise ValueError(error_msg)

        # Idempotency check: if already in terminal state, just return
        if job.is_terminal:
            logger.warning(
                f"[Job {job_id}] Already in terminal state: {job.status.value}. "
                f"Skipping execution."
            )
            return {
                "job_id": job_id,
                "status": job.status.value,
                "message": "Job already in terminal state"
            }

        # Validate job is in QUEUED status (should be set by submit endpoint)
        if job.status != JobStatus.QUEUED:
            logger.warning(
                f"[Job {job_id}] Expected QUEUED status, got {job.status.value}. "
                f"Proceeding anyway..."
            )

        # ===== STEP 2: Transition to RUNNING =====
        logger.info(f"[Job {job_id}] Transitioning QUEUED → RUNNING")
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "RUNNING",
                started_at=started_at,
                worker_id=self.request.id,
                current_step="Initializing simulation",
                retry_count=self.request.retries,
                progress=0.0
            )
        )

        # Record STARTED provenance event
        async def record_started_event():
            async with self.get_db_session() as db:
                await record_provenance(
                    db,
                    EntityType.JOB,
                    uuid.UUID(job_id),
                    EventType.STARTED,
                    details={
                        "engine_name": job.engine,
                        "worker_id": self.request.id,
                        "host_info": get_system_info(),
                        "code_version": get_code_version(),
                        "started_at": started_at.isoformat(),
                        "retry_count": self.request.retries,
                    }
                )

        asyncio.run(record_started_event())

        # ===== STEP 3: Extract job data =====
        # Get structure data (with safe fallback if structure not loaded)
        structure_data = {}
        if hasattr(job, "structure") and job.structure:
            structure_data = job.structure.structure_data or {}
            # Add metadata from structure
            structure_data["id"] = str(job.structure.id) if hasattr(job.structure, "id") else "unknown"
            structure_data["formula"] = getattr(job.structure, "formula", "UnknownMaterial")
            structure_data["n_atoms"] = getattr(job.structure, "n_atoms", 10)
            structure_data["dimensionality"] = getattr(job.structure, "dimensionality", 3)
        else:
            logger.warning(f"[Job {job_id}] Structure not loaded, using empty structure data")
            structure_data = {
                "id": "unknown",
                "formula": "UnknownMaterial",
                "n_atoms": 10,
                "dimensionality": 3
            }

        parameters = job.parameters or {}
        engine = job.engine

        logger.info(
            f"[Job {job_id}] Configuration: engine={engine}, "
            f"n_atoms={structure_data.get('n_atoms', 0)}, "
            f"formula={structure_data.get('formula', 'Unknown')}"
        )

        # ===== STEP 4: Run simulation =====
        logger.info(f"[Job {job_id}] Executing {engine} simulation...")

        # Update progress: Initializing
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "RUNNING",
                progress=0.1,
                current_step="Initializing simulation engine",
            )
        )

        # Define progress callback
        def update_progress(progress: float, step: str):
            """Update job progress in database."""
            try:
                # Map simulation progress (0-1) to job progress (0.1 to 0.8)
                # Reserve 0-10% for init, 80-90% for artifact creation, 90-100% for DB writes
                job_progress = 0.1 + 0.7 * progress
                asyncio.run(
                    self._update_job_status_async(
                        job_id,
                        "RUNNING",
                        progress=job_progress,
                        current_step=step,
                    )
                )
            except Exception as e:
                logger.warning(f"[Job {job_id}] Failed to update progress: {e}")

        # ===== STEP 4.1: Get appropriate engine =====
        try:
            engine_class = get_engine(engine)
            logger.info(f"[Job {job_id}] Using engine: {engine_class.__name__}")
        except ValueError as e:
            logger.warning(f"[Job {job_id}] {e}. Falling back to MOCK engine")
            engine_class = get_engine("MOCK")

        # ===== STEP 4.2: Initialize engine =====
        engine_instance = engine_class()

        # ===== STEP 4.3: Setup engine =====
        try:
            engine_instance.setup(structure=structure_data, parameters=parameters)
        except Exception as e:
            logger.error(f"[Job {job_id}] Engine setup failed: {e}", exc_info=True)
            raise RuntimeError(f"Engine setup failed: {e}") from e

        # ===== STEP 4.4: Run simulation =====
        try:
            # Check if engine has async run method or sync
            if hasattr(engine_instance, '_run_async'):
                # Async engine (like MockSimulationEngine)
                simulation_result = asyncio.run(
                    engine_instance._run_async(progress_callback=update_progress)
                )
            else:
                # Sync engine (like QuantumEspressoEngine)
                simulation_result = engine_instance.run(progress_callback=update_progress)
        except Exception as e:
            logger.error(f"[Job {job_id}] Simulation execution failed: {e}", exc_info=True)
            # Attempt cleanup even on failure
            try:
                engine_instance.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"[Job {job_id}] Cleanup after failure failed: {cleanup_error}")
            raise RuntimeError(f"Simulation execution failed: {e}") from e

        # ===== STEP 4.5: Cleanup engine =====
        try:
            engine_instance.cleanup()
        except Exception as e:
            logger.warning(f"[Job {job_id}] Engine cleanup failed (non-fatal): {e}")

        logger.info(f"[Job {job_id}] Simulation completed successfully")

        # ===== STEP 5: Create artifacts =====
        logger.info(f"[Job {job_id}] Creating artifacts...")
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "RUNNING",
                progress=0.85,
                current_step="Creating artifacts",
            )
        )

        artifacts_path = self._create_job_artifacts(
            job_id=job_id,
            results=simulation_result,
            job_data={"engine": engine, "structure": structure_data}
        )

        if artifacts_path:
            logger.info(f"[Job {job_id}] Artifacts created at: {artifacts_path}")
        else:
            logger.warning(f"[Job {job_id}] Failed to create artifacts (non-fatal)")

        # ===== STEP 6: Store results in database =====
        logger.info(f"[Job {job_id}] Storing results in database...")
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "RUNNING",
                progress=0.95,
                current_step="Storing results",
            )
        )

        asyncio.run(
            self._create_simulation_result_async(
                job_id,
                summary=simulation_result.get("summary", {}),
                convergence_reached=simulation_result.get("convergence_reached", True),
                quality_score=simulation_result.get("quality_score"),
                artifacts_path=artifacts_path,
                metadata=simulation_result.get("metadata", {}),
            )
        )

        # ===== STEP 7: Transition to COMPLETED =====
        finished_at = datetime.utcnow()
        duration = (finished_at - started_at).total_seconds()

        logger.info(f"[Job {job_id}] Transitioning RUNNING → COMPLETED")
        asyncio.run(
            self._update_job_status_async(
                job_id,
                "COMPLETED",
                finished_at=finished_at,
                progress=1.0,
                current_step="Completed",
            )
        )

        # Record COMPLETED provenance event
        async def record_completed_event():
            async with self.get_db_session() as db:
                await record_provenance(
                    db,
                    EntityType.JOB,
                    uuid.UUID(job_id),
                    EventType.COMPLETED,
                    details={
                        "duration_ms": duration * 1000,
                        "result_summary": simulation_result.get("summary", {}),
                        "finished_at": finished_at.isoformat(),
                        "artifacts_path": artifacts_path,
                        "convergence_reached": simulation_result.get("convergence_reached", True),
                        "quality_score": simulation_result.get("quality_score"),
                    }
                )

        asyncio.run(record_completed_event())

        logger.info(
            f"[Job {job_id}] Simulation completed successfully in {duration:.2f}s"
        )

        return {
            "job_id": job_id,
            "status": "success",
            "duration_seconds": duration,
            "summary": simulation_result.get("summary", {}),
            "artifacts_path": artifacts_path,
        }

    except Exception as e:
        # ===== ERROR HANDLING =====
        finished_at = datetime.utcnow()
        duration = (finished_at - started_at).total_seconds()
        error_message = f"{type(e).__name__}: {str(e)}"

        logger.error(
            f"[Job {job_id}] Simulation failed after {duration:.2f}s: {error_message}",
            exc_info=True
        )

        # Update status to FAILED with error message
        try:
            asyncio.run(
                self._update_job_status_async(
                    job_id,
                    "FAILED",
                    error_message=error_message,
                    finished_at=finished_at,
                    retry_count=self.request.retries,
                )
            )

            # Record FAILED provenance event
            async def record_failed_event():
                async with self.get_db_session() as db:
                    import traceback
                    await record_provenance(
                        db,
                        EntityType.JOB,
                        uuid.UUID(job_id),
                        EventType.FAILED,
                        details={
                            "error_message": error_message,
                            "traceback": traceback.format_exc(),
                            "failed_at": finished_at.isoformat(),
                            "duration_ms": duration * 1000,
                            "retry_count": self.request.retries,
                        }
                    )

            asyncio.run(record_failed_event())

        except Exception as update_error:
            logger.error(
                f"[Job {job_id}] Failed to update job status to FAILED: {update_error}"
            )

        # Retry logic: retry on recoverable errors
        if self.request.retries < self.max_retries:
            retry_delay = self.default_retry_delay * (self.request.retries + 1)  # Exponential backoff
            logger.info(
                f"[Job {job_id}] Retrying in {retry_delay}s "
                f"(attempt {self.request.retries + 1}/{self.max_retries})"
            )
            raise self.retry(exc=e, countdown=retry_delay)
        else:
            logger.error(
                f"[Job {job_id}] Failed permanently after {self.max_retries} retries"
            )
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

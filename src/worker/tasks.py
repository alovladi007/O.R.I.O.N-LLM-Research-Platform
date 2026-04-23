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
from typing import Any, Callable, Dict, Optional
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


@celery_app.task(
    name="run_orchestrator_step_task",
    base=DatabaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=60,
)
def run_orchestrator_step_task(
    self,
    orchestrator_id: str,
    triggered_by: str = "scheduler",
    trigger_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute one orchestrator step.

    This task runs the orchestrator logic that:
    - Advances design campaigns
    - Schedules simulations and experiments
    - Triggers model retraining when needed

    Args:
        orchestrator_id: UUID of the orchestrator state
        triggered_by: Who/what triggered this run (scheduler, manual, api, agent)
        trigger_context: Additional context about the trigger

    Returns:
        Dictionary with orchestrator run results
    """
    from backend.orchestrator import run_orchestrator_step

    logger.info(f"Running orchestrator step (triggered by: {triggered_by})")

    try:
        # Run orchestrator step synchronously with async DB operations
        async def run_step_async():
            async with self.get_db_session() as db:
                run = run_orchestrator_step(
                    db=db,
                    orchestrator_id=uuid.UUID(orchestrator_id),
                    triggered_by=triggered_by,
                    trigger_context=trigger_context
                )
                return run

        run = asyncio.run(run_step_async())

        logger.info(
            f"Orchestrator step completed: "
            f"campaigns_advanced={len(run.actions.get('campaigns_advanced', []))}, "
            f"simulations_launched={run.actions.get('simulations_launched', 0)}, "
            f"experiments_launched={run.actions.get('experiments_launched', 0)}"
        )

        return {
            "orchestrator_id": orchestrator_id,
            "run_id": str(run.id),
            "success": run.success,
            "actions": run.actions,
            "duration_seconds": run.duration_seconds,
        }

    except Exception as e:
        logger.error(f"Orchestrator step failed: {e}", exc_info=True)

        # Retry on failure
        if self.request.retries < self.max_retries:
            retry_delay = self.default_retry_delay * (self.request.retries + 1)
            logger.info(f"Retrying orchestrator step in {retry_delay}s")
            raise self.retry(exc=e, countdown=retry_delay)
        else:
            raise


# ---------------------------------------------------------------------------
# Reaper — Session 2.1
# ---------------------------------------------------------------------------


def _reap_stalled_sync(stall_seconds: int) -> Dict[str, Any]:
    """
    Sync implementation of the stalled-job reaper. Split out for tests.

    Finds jobs whose ``status == RUNNING`` and ``updated_at`` is older
    than ``stall_seconds`` ago, and transitions them to FAILED with
    reason ``worker_lost``. Uses a sync engine so the Celery task (sync
    entrypoint) can run it without spinning up an event loop.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from src.api.config import settings
    from src.api.models.simulation import (
        IllegalJobTransitionError,
        JobStatus,
        SimulationJob,
    )

    # async → sync URL (strip +asyncpg driver).
    sync_url = settings.database_url.replace("+asyncpg", "")
    engine = create_engine(sync_url, pool_pre_ping=True, future=True)

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=stall_seconds)
    reaped: list[str] = []
    errors: list[str] = []

    with Session(engine, expire_on_commit=False) as session:
        stmt = select(SimulationJob).where(
            SimulationJob.status == JobStatus.RUNNING,
            SimulationJob.updated_at < cutoff,
        )
        for job in session.execute(stmt).scalars().all():
            try:
                job.transition_to(
                    JobStatus.FAILED,
                    set_finished=True,
                    error_message=f"worker_lost: no heartbeat in ≥{stall_seconds}s",
                )
                reaped.append(str(job.id))
            except IllegalJobTransitionError as exc:
                errors.append(f"{job.id}: {exc}")
        session.commit()

    engine.dispose()
    return {"reaped": reaped, "errors": errors}


# ---------------------------------------------------------------------------
# Mock static — Session 2.2
# ---------------------------------------------------------------------------


def _sync_session_for_worker():
    """Return ``(engine, Session)`` bound to the configured Postgres URL.

    Celery tasks run in threads where ``asyncio.run`` is expensive and
    awkward. The Session 2.2 mock task uses a sync engine so the hot
    path is pure blocking code that pairs cleanly with ``JobLifecycle``.
    Caller is responsible for disposing the engine.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from src.api.config import settings

    sync_url = settings.database_url.replace("+asyncpg", "")
    engine = create_engine(sync_url, pool_pre_ping=True, future=True)
    Session = sessionmaker(engine, expire_on_commit=False, future=True)
    return engine, Session


def _load_structure_atoms(session, structure_id: str):
    """Return ``(atoms, formula)`` for *structure_id* as a tuple.

    Accepts either the ``atoms`` JSON column or a fallback synthetic
    single-atom structure when the row has neither a parsed atoms list
    nor a ``num_atoms`` hint. Raises if the structure row is missing —
    the Celery task catches and marks FAILED.
    """
    from src.api.models import Structure

    structure = session.get(Structure, structure_id)
    if structure is None:
        raise ValueError(f"Structure {structure_id!r} not found")

    atoms = structure.atoms or []
    if not atoms:
        # Synthesize a plausible atom list so the mock still runs on
        # structures uploaded pre-parse. Uses the formula's first char.
        formula = structure.formula or "H"
        first = "".join([c for c in formula if c.isalpha()])[:1] or "H"
        atoms = [
            {"species": first, "position": [0.0, 0.0, 0.0]}
        ]
        for i in range(1, max(1, structure.num_atoms or 1)):
            atoms.append({"species": first, "position": [i * 1.5, 0.0, 0.0]})
    return atoms, (structure.formula or "Unknown")


def _load_structure_for_qe(session, structure_id: str) -> dict:
    """Return the QE-shaped structure dict for *structure_id*.

    Shape: ``{lattice: 3x3 Å, species: [str], frac_coords: [[x,y,z]]}``
    matching the contract of
    :func:`backend.common.engines.qe_input.generate_pw_input`.
    """
    from src.api.models import Structure

    structure = session.get(Structure, structure_id)
    if structure is None:
        raise ValueError(f"Structure {structure_id!r} not found")

    atoms_json = structure.atoms or []
    if not atoms_json:
        raise ValueError(
            f"Structure {structure_id!r} has no parsed atoms — re-upload via /structures."
        )
    # atoms_json is a list of dicts with "species" + "position".
    species = [a.get("species") or a.get("symbol") for a in atoms_json]
    frac_coords = [a.get("position") or a.get("coords") or [0, 0, 0] for a in atoms_json]

    # Lattice: Structure model uses `lattice_vectors` (list[list[float]]) or
    # the `lattice` JSON dict. Prefer the former (Session 1.1 parsers).
    lat = getattr(structure, "lattice_vectors", None)
    if not lat:
        lat_field = structure.lattice or {}
        # Some parsers stash the matrix under "matrix".
        lat = lat_field.get("matrix") if isinstance(lat_field, dict) else None
    if not lat:
        raise ValueError(
            f"Structure {structure_id!r} has no lattice vectors — "
            "QE needs a 3x3 matrix."
        )
    return {
        "lattice": lat,
        "species": species,
        "frac_coords": frac_coords,
    }


@celery_app.task(
    name="orion.mock.static",
    bind=True,
    autoretry_for=(),   # no retries for the mock — errors are logic bugs
    acks_late=True,
)
def run_mock_static_job(self, job_id: str) -> Dict[str, Any]:
    """Run the mock_static engine end-to-end for *job_id*.

    Wiring in one place:

    1. Open a sync session against Postgres.
    2. Use :class:`JobLifecycle` to advance state and emit events.
    3. Load the target :class:`Structure`, run
       :func:`backend.common.jobs.run_mock_static`, and populate
       ``lc.outputs`` + ``lc.bundle``.
    4. On exit, :class:`JobLifecycle` persists outputs and bundles the
       run dir into MinIO (best-effort).
    """
    from backend.common.jobs import (
        DEFAULT_ARTIFACTS_BUCKET,
        build_minio_client,
        ensure_bucket,
        run_mock_static,
        write_trajectory_xyz,
    )
    from backend.common.workers import (
        ArtifactBundle,
        JobLifecycle,
        build_run_dir,
        tar_and_upload_run_dir,
    )
    from backend.common.workers.events import NullEventEmitter, RedisPubSubEmitter

    engine, Session = _sync_session_for_worker()
    minio_client = None
    try:
        # EventEmitter: prefer Redis if reachable; falls back to Null on
        # any connection failure (the worker host is the one that
        # decides, not the caller).
        try:
            minio_client = build_minio_client()
            ensure_bucket(minio_client, DEFAULT_ARTIFACTS_BUCKET)
        except Exception as exc:  # noqa: BLE001
            logger.warning("mock_static: MinIO setup failed: %s", exc)
            minio_client = None

        emitter = RedisPubSubEmitter()

        with Session() as session:
            with JobLifecycle(
                job_id, session=session, emitter=emitter, worker_id=self.request.id,
            ) as lc:
                # Get structure id off the loaded job row.
                assert lc.job is not None
                structure_id = lc.job.structure_id  # type: ignore[attr-defined]

                atoms, formula = _load_structure_atoms(session, structure_id)
                result = run_mock_static(
                    structure_id=str(structure_id),
                    atoms=atoms,
                    formula=formula,
                )

                # Build a run dir, drop the three expected files in.
                run_dir = build_run_dir(str(job_id))
                energies_json = {
                    "schema_version": result.schema_version,
                    "energy_ev": result.energy_ev,
                    "energy_per_atom_ev": result.energy_per_atom_ev,
                    "converged": result.converged,
                }
                (run_dir / "energies.json").write_text(
                    json.dumps(energies_json, indent=2, sort_keys=True)
                )
                forces_json = [f.model_dump() for f in result.forces]
                (run_dir / "forces.json").write_text(
                    json.dumps(forces_json, indent=2)
                )
                species_list = [f.species for f in result.forces]
                (run_dir / "trajectory.xyz").write_text(
                    write_trajectory_xyz(result.trajectory, species_list)
                )

                # Tarball + MinIO upload. Non-fatal on failure.
                bundle = ArtifactBundle(
                    run_dir=run_dir,
                    job_id=str(job_id),
                    manifest={
                        "engine": "mock",
                        "kind": "mock_static",
                        "schema_version": result.schema_version,
                        "n_atoms": result.n_atoms,
                        "formula": result.formula,
                    },
                )
                upload = tar_and_upload_run_dir(
                    bundle, minio_client=minio_client, bucket=DEFAULT_ARTIFACTS_BUCKET,
                )

                # Stash artifact coords on the job row for the GET
                # /jobs/{id}/artifacts endpoint to read.
                extra = dict(getattr(lc.job, "extra_metadata", None) or {})
                extra["artifact"] = {
                    "bucket": upload["bucket"],
                    "key": upload["key"],
                    "tarball": upload["tarball"],
                    "size_bytes": upload["size_bytes"],
                    "uploaded": upload["uploaded"],
                }
                lc.job.extra_metadata = extra  # type: ignore[attr-defined]

                # Populate outputs — JobLifecycle persists these on exit.
                lc.outputs = result.model_dump()

        return {
            "job_id": job_id,
            "status": "success",
            "energy_ev": result.energy_ev,
            "n_atoms": result.n_atoms,
            "artifact_key": upload["key"],
        }
    finally:
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# DFT static via Quantum Espresso — Session 3.2
# ---------------------------------------------------------------------------


@celery_app.task(
    name="orion.dft.static",
    bind=True,
    acks_late=True,
)
def run_dft_static_job(self, job_id: str) -> Dict[str, Any]:
    """Run a single pw.x SCF for *job_id* and persist outputs + artifact.

    Wiring:
    1. Open sync session.
    2. Enter :class:`JobLifecycle`.
    3. Load Structure → build ``QEInputParams`` from
       ``job.parameters`` → generate_pw_input via
       :mod:`backend.common.engines.qe_input` (Session 3.1).
    4. Invoke :func:`run_pw` via :mod:`backend.common.engines.qe_run`
       (Session 3.2). UPFs staged into ``run_dir/pseudos``.
    5. Persist parsed output dict onto ``lc.outputs``; tar run_dir to
       MinIO for download via ``GET /jobs/{id}/artifacts``.

    Binary path: reads ``QE_EXECUTABLE`` env (default ``pw.x``).
    Pseudo library path: ``QE_PSEUDO_DIR`` env (the runner's default).
    """
    import os
    from pathlib import Path

    from backend.common.engines.qe_input import (
        PseudopotentialRegistry,
        QEInputParams,
        generate_pw_input,
    )
    from backend.common.engines.qe_run import run_pw
    from backend.common.jobs import (
        DEFAULT_ARTIFACTS_BUCKET,
        build_minio_client,
        ensure_bucket,
    )
    from backend.common.workers import (
        ArtifactBundle,
        JobLifecycle,
        build_run_dir,
        tar_and_upload_run_dir,
    )
    from backend.common.workers.events import NullEventEmitter, RedisPubSubEmitter

    engine, Session = _sync_session_for_worker()
    minio_client = None
    try:
        try:
            minio_client = build_minio_client()
            ensure_bucket(minio_client, DEFAULT_ARTIFACTS_BUCKET)
        except Exception as exc:  # noqa: BLE001
            logger.warning("dft_static: MinIO setup failed: %s", exc)
            minio_client = None

        emitter = RedisPubSubEmitter()

        with Session() as session:
            with JobLifecycle(
                job_id, session=session, emitter=emitter, worker_id=self.request.id,
            ) as lc:
                assert lc.job is not None
                structure_id = lc.job.structure_id  # type: ignore[attr-defined]
                parameters = lc.job.parameters or {}  # type: ignore[attr-defined]

                # Load + build input
                qe_struct = _load_structure_for_qe(session, structure_id)
                pseudo_dir = os.getenv(
                    "QE_PSEUDO_DIR",
                    str(Path.home() / "orion" / "pseudos" / "SSSP_1.3.0_PBE_efficiency"),
                )
                registry = PseudopotentialRegistry(pseudo_dir)

                qe_params_dict = {
                    k: v for k, v in parameters.items()
                    if k in QEInputParams.model_fields
                }
                params = QEInputParams(**qe_params_dict)
                rendered = generate_pw_input(qe_struct, params, registry)

                # Run pw.x
                run_dir = build_run_dir(str(job_id))
                pw_executable = os.getenv("QE_EXECUTABLE", "pw.x")
                walltime = int(parameters.get("walltime_minutes", 60))
                cpus = int(parameters.get("cpus", 1))
                run_result = run_pw(
                    rendered,
                    run_dir,
                    qe_executable=pw_executable,
                    pseudo_src_dir=registry.pseudo_dir,
                    cpus=cpus,
                    walltime_minutes=walltime,
                    species_hint=qe_struct["species"],
                )

                if not run_result.success:
                    raise RuntimeError(
                        f"pw.x stage={run_result.stage}: {run_result.error_message}"
                    )

                # Package artifact
                bundle = ArtifactBundle(
                    run_dir=run_dir,
                    job_id=str(job_id),
                    manifest={
                        "engine": "qe",
                        "kind": "dft_static",
                        "ecutwfc_ry": rendered.ecutwfc_ry,
                        "ecutrho_ry": rendered.ecutrho_ry,
                        "kpoints": list(
                            (rendered.kpoints.nk1, rendered.kpoints.nk2, rendered.kpoints.nk3)
                        ),
                    },
                )
                upload = tar_and_upload_run_dir(
                    bundle, minio_client=minio_client, bucket=DEFAULT_ARTIFACTS_BUCKET,
                )
                extra = dict(getattr(lc.job, "extra_metadata", None) or {})
                extra["artifact"] = {
                    "bucket": upload["bucket"],
                    "key": upload["key"],
                    "tarball": upload["tarball"],
                    "size_bytes": upload["size_bytes"],
                    "uploaded": upload["uploaded"],
                }
                lc.job.extra_metadata = extra  # type: ignore[attr-defined]

                # Outputs persisted to parameters._outputs on exit.
                lc.outputs = run_result.output.as_dict()

        return {
            "job_id": job_id,
            "status": "success",
            "energy_ev": run_result.output.energy.total_ev if run_result.output.energy else None,
            "artifact_key": upload["key"],
        }
    finally:
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Shared DFT helper — Session 3.3
# ---------------------------------------------------------------------------


def _run_pw_step(
    self,
    job_id: str,
    *,
    kind: str,
    calculation: str,
    prior_pw_run: Optional[str] = None,
    pre_pw: Optional[Callable] = None,
    post_pw: Optional[Callable] = None,
    extra_param_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Shared body for all four QE workflow kinds.

    ``calculation`` drives the pw.x run (scf / vc-relax / bands / nscf).
    ``prior_pw_run`` — when a previous step needs to be chained (e.g.
    bands relies on the upstream scf's outdir), the caller resolves a
    step-id-to-outdir mapping here by reading
    ``job.parameters["_upstream"]`` written by the workflow tick.
    Unused in Session 3.3's initial cut since Celery tasks each own
    their own run dir; the workflow executor will feed outputs via the
    ``{"uses": ...}`` resolver.

    ``post_pw(run_dir, registry, pw_result, session, job) -> dict`` is
    called after pw.x completes successfully; its return value is
    merged into ``lc.outputs``. DOS and phonon tasks use it to invoke
    the follow-up binary.
    """
    import os
    from pathlib import Path

    from backend.common.engines.qe_input import (
        PseudopotentialRegistry,
        QEInputParams,
        generate_pw_input,
    )
    from backend.common.engines.qe_run import run_pw
    from backend.common.jobs import (
        DEFAULT_ARTIFACTS_BUCKET,
        build_minio_client,
        ensure_bucket,
    )
    from backend.common.workers import (
        ArtifactBundle,
        JobLifecycle,
        build_run_dir,
        tar_and_upload_run_dir,
    )
    from backend.common.workers.events import RedisPubSubEmitter

    engine, Session = _sync_session_for_worker()
    minio_client = None
    try:
        try:
            minio_client = build_minio_client()
            ensure_bucket(minio_client, DEFAULT_ARTIFACTS_BUCKET)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: MinIO setup failed: %s", kind, exc)
            minio_client = None

        emitter = RedisPubSubEmitter()

        with Session() as session:
            with JobLifecycle(
                job_id, session=session, emitter=emitter, worker_id=self.request.id,
            ) as lc:
                assert lc.job is not None
                structure_id = lc.job.structure_id  # type: ignore[attr-defined]
                parameters = lc.job.parameters or {}  # type: ignore[attr-defined]

                qe_struct = _load_structure_for_qe(session, structure_id)

                # Pre-pw guard — fires after structure load but before
                # any compute. Raises propagate through JobLifecycle
                # so the job row records FAILED with the exception
                # message, matching every other failure mode.
                if pre_pw is not None:
                    pre_pw(qe_struct=qe_struct, params=parameters)

                pseudo_dir = os.getenv(
                    "QE_PSEUDO_DIR",
                    str(Path.home() / "orion" / "pseudos" / "SSSP_1.3.0_PBE_efficiency"),
                )
                registry = PseudopotentialRegistry(pseudo_dir)

                # Filter down to valid QEInputParams fields; extras go
                # through extra_control etc when the caller knows what
                # they're doing.
                qe_params_dict = {
                    k: v for k, v in parameters.items()
                    if k in QEInputParams.model_fields
                }
                qe_params_dict["calculation"] = calculation
                if extra_param_overrides:
                    qe_params_dict.update(extra_param_overrides)
                qe_input_params = QEInputParams(**qe_params_dict)

                rendered = generate_pw_input(qe_struct, qe_input_params, registry)
                run_dir = build_run_dir(str(job_id))
                pw_executable = os.getenv("QE_EXECUTABLE", "pw.x")
                walltime = int(parameters.get("walltime_minutes", 60))
                cpus = int(parameters.get("cpus", 1))

                run_result = run_pw(
                    rendered,
                    run_dir,
                    qe_executable=pw_executable,
                    pseudo_src_dir=registry.pseudo_dir,
                    cpus=cpus,
                    walltime_minutes=walltime,
                    species_hint=qe_struct["species"],
                )
                if not run_result.success:
                    raise RuntimeError(
                        f"{kind} pw.x stage={run_result.stage}: {run_result.error_message}"
                    )

                outputs: Dict[str, Any] = run_result.output.as_dict()
                if post_pw is not None:
                    extra_outputs = post_pw(
                        run_dir=run_dir,
                        registry=registry,
                        pw_result=run_result,
                        params=parameters,
                        lc=lc,
                        session=session,
                    )
                    if extra_outputs:
                        outputs.update(extra_outputs)

                bundle = ArtifactBundle(
                    run_dir=run_dir,
                    job_id=str(job_id),
                    manifest={
                        "engine": "qe",
                        "kind": kind,
                        "calculation": calculation,
                        "ecutwfc_ry": rendered.ecutwfc_ry,
                        "ecutrho_ry": rendered.ecutrho_ry,
                        "kpoints": [
                            rendered.kpoints.nk1, rendered.kpoints.nk2, rendered.kpoints.nk3,
                        ],
                    },
                )
                upload = tar_and_upload_run_dir(
                    bundle, minio_client=minio_client, bucket=DEFAULT_ARTIFACTS_BUCKET,
                )
                extra = dict(getattr(lc.job, "extra_metadata", None) or {})
                extra["artifact"] = {
                    "bucket": upload["bucket"],
                    "key": upload["key"],
                    "tarball": upload["tarball"],
                    "size_bytes": upload["size_bytes"],
                    "uploaded": upload["uploaded"],
                }
                lc.job.extra_metadata = extra  # type: ignore[attr-defined]

                lc.outputs = outputs

        return {
            "job_id": job_id,
            "status": "success",
            "kind": kind,
            "artifact_key": upload["key"],
            "energy_ev": (
                run_result.output.energy.total_ev if run_result.output.energy else None
            ),
        }
    finally:
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            pass


def _run_secondary_binary(
    run_dir: Path, executable: str, input_name: str, input_text: str,
    *,
    walltime_minutes: int = 30, cpus: int = 1,
) -> "PWRunResult":
    """Invoke a secondary QE binary (dos.x / ph.x) in an existing run_dir.

    Uses Session 2.3's execution backend so the same cancel / walltime
    machinery applies. The input file is written relative to ``run_dir``
    and the binary is expected to read ``prefix.save/`` that pw.x left.
    """
    from backend.common.engines.qe_run.runner import PWRunResult
    from backend.common.engines.qe_run.output import parse_pw_output
    from backend.common.execution import (
        JobState,
        Resources,
        get_execution_backend,
        sync_execute,
    )

    input_path = run_dir / input_name
    input_path.write_text(input_text)
    stdout_path = run_dir / f"{input_name}.stdout"
    stderr_path = run_dir / f"{input_name}.stderr"

    backend = get_execution_backend("local")
    # We pipe stdin to pw.x-style binaries via the subprocess stdin; the
    # canonical way for dos.x / ph.x is to pass the input on the
    # command line with "-i <input>" if they support it, else via stdin.
    # Both dos.x and ph.x accept -inp; use it.
    state = sync_execute(
        backend,
        [executable, "-inp", str(input_path)],
        run_dir,
        Resources(cpus=cpus, walltime_minutes=walltime_minutes),
        poll_interval_seconds=0.5,
    )
    # sync_execute wrote stdout.txt/stderr.txt in run_dir; rename so
    # subsequent binary calls don't overwrite.
    default_stdout = run_dir / "stdout.txt"
    default_stderr = run_dir / "stderr.txt"
    if default_stdout.exists():
        default_stdout.replace(stdout_path)
    if default_stderr.exists():
        default_stderr.replace(stderr_path)

    text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
    # We don't use parse_pw_output here — pw output parser wouldn't
    # find most fields. But we re-use its error detection for ERRORED.
    # For now return a shallow result; tasks below reparse stdout
    # for the kind-specific info they care about.
    success = state == JobState.COMPLETED
    return PWRunResult(
        run_dir=run_dir,
        input_path=input_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        output=None,  # secondary binaries have their own parsers
        returncode=0 if success else 1,
        success=success,
        stage="ok" if success else "nonzero_exit",
        error_message=None if success else f"{executable} state={state.value}",
        extra={"stdout_text": text},
    )


# ---------------------------------------------------------------------------
# orion.dft.relax — vc-relax and persist relaxed geometry
# ---------------------------------------------------------------------------


@celery_app.task(name="orion.dft.relax", bind=True, acks_late=True)
def run_dft_relax_job(self, job_id: str) -> Dict[str, Any]:
    """Full vc-relax: lattice + positions relaxed to forces < forc_conv_thr.

    Outputs include the relaxed structure block, which downstream
    workflow steps consume via ``{"uses": "step.outputs.relaxed.lattice_ang"}``.
    """
    return _run_pw_step(
        self, job_id,
        kind="dft_relax",
        calculation="vc-relax",
    )


# ---------------------------------------------------------------------------
# orion.dft.bands — non-SCF along a k-path
# ---------------------------------------------------------------------------


def _bands_post_pw(*, run_dir, registry, pw_result, params, lc, session):
    """Write band_structure.json in the run_dir + populate a minimal
    pymatgen-compatible dict on the job outputs.

    Doesn't invoke a separate binary; pw.x in ``calculation=bands``
    already did the work.
    """
    import json

    if pw_result.output is None or pw_result.output.bands is None:
        return {}
    bands_dict = pw_result.output.as_dict().get("bands", {})
    (run_dir / "band_structure.json").write_text(
        json.dumps(bands_dict, indent=2, default=str)
    )
    return {"band_structure_json": "band_structure.json"}


@celery_app.task(name="orion.dft.bands", bind=True, acks_late=True)
def run_dft_bands_job(self, job_id: str) -> Dict[str, Any]:
    """Run pw.x in ``calculation='bands'`` along the caller's k-path.

    The workflow template is expected to have set ``parameters.kpoints``
    to a Monkhorst-Pack dense grid OR left it None (SSSP default) —
    for real bands calculations, the workflow template pushes a list
    of k-points on a high-symmetry path, but that requires pw.x's
    ``K_POINTS crystal_b`` block which QEInputParams doesn't yet
    expose. Session 3.3 ships the task wiring + uses uniform kpoints
    as a placeholder; proper k-path comes in a follow-up that adds a
    kpath field to QEInputParams.
    """
    return _run_pw_step(
        self, job_id,
        kind="dft_bands",
        calculation="bands",
        post_pw=_bands_post_pw,
    )


# ---------------------------------------------------------------------------
# orion.dft.dos — scf + dos.x with delta_e=0.01 eV
# ---------------------------------------------------------------------------


def _dos_post_pw(*, run_dir, registry, pw_result, params, lc, session):
    """Run dos.x after a successful scf and persist DOS data.

    Writes dos.csv to the run_dir and records VBM/CBM/bandgap onto
    outputs so workflow resolvers can pull them.
    """
    from backend.common.engines.qe_run import parse_dos_output
    import csv
    import os

    prefix = params.get("prefix") or "orion"
    delta_e_ev = float(params.get("dos_delta_e_ev", 0.01))
    emin = params.get("dos_emin_ev", "-20.0")
    emax = params.get("dos_emax_ev", "20.0")
    dos_executable = os.getenv("QE_DOS_EXECUTABLE", "dos.x")

    # dos.x input: reads pwscf outdir + prefix, writes <prefix>.dos.
    dos_in = (
        f"&DOS\n"
        f"  prefix = '{prefix}'\n"
        f"  outdir = './tmp'\n"
        f"  fildos = '{prefix}.dos'\n"
        f"  Emin = {emin}\n"
        f"  Emax = {emax}\n"
        f"  DeltaE = {delta_e_ev}\n"
        f"/\n"
    )
    result = _run_secondary_binary(
        run_dir, dos_executable, "dos.in", dos_in,
        walltime_minutes=int(params.get("walltime_minutes", 30)),
        cpus=int(params.get("cpus", 1)),
    )
    if not result.success:
        raise RuntimeError(f"dos.x failed: {result.error_message}")

    dos_path = run_dir / f"{prefix}.dos"
    parsed = parse_dos_output(dos_path)

    csv_path = run_dir / "dos.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["E_eV", "DOS", "IntegratedDOS"])
        for e, d, i in zip(parsed.energies_ev, parsed.dos, parsed.idos):
            w.writerow([e, d, i])

    # VBM / CBM / bandgap
    vbm, cbm = _estimate_vbm_cbm(parsed)
    bandgap_ev = None if vbm is None or cbm is None else max(0.0, cbm - vbm)

    return {
        "dos_csv": "dos.csv",
        "fermi_energy_ev": parsed.fermi_energy_ev,
        "vbm_ev": vbm,
        "cbm_ev": cbm,
        "bandgap_ev": bandgap_ev,
        "dos_n_points": len(parsed.energies_ev),
    }


def _estimate_vbm_cbm(dos) -> tuple[Optional[float], Optional[float]]:
    """Rough VBM/CBM from a DOS + integrated DOS around E_F.

    VBM = highest energy with DOS > threshold below E_F;
    CBM = lowest energy with DOS > threshold above E_F.
    Threshold is 1% of the max DOS value — coarse but good enough for
    semiconductor screening. Metals return ``(None, None)`` by design.
    """
    if not dos.energies_ev or dos.fermi_energy_ev is None:
        return None, None
    ef = dos.fermi_energy_ev
    max_dos = max(dos.dos) if dos.dos else 0.0
    if max_dos <= 0:
        return None, None
    thr = 0.01 * max_dos

    vbm = None
    for e, d in zip(dos.energies_ev, dos.dos):
        if e <= ef and d > thr:
            vbm = e
    cbm = None
    for e, d in zip(dos.energies_ev, dos.dos):
        if e > ef and d > thr:
            cbm = e
            break
    # If VBM and CBM are both at Fermi (metal), call it no-gap.
    if vbm is not None and cbm is not None and cbm - vbm < 0.05:
        return None, None
    return vbm, cbm


@celery_app.task(name="orion.dft.dos", bind=True, acks_late=True)
def run_dft_dos_job(self, job_id: str) -> Dict[str, Any]:
    """SCF + dos.x post-processing."""
    return _run_pw_step(
        self, job_id,
        kind="dft_dos",
        calculation="scf",
        post_pw=_dos_post_pw,
    )


# ---------------------------------------------------------------------------
# orion.dft.phonons_gamma — scf + ph.x at Γ
# ---------------------------------------------------------------------------


def _phonons_gamma_post_pw(*, run_dir, registry, pw_result, params, lc, session):
    """Run ph.x at Γ after a successful scf and extract mode frequencies."""
    from backend.common.engines.qe_run import parse_ph_output
    import os

    prefix = params.get("prefix") or "orion"
    ph_executable = os.getenv("QE_PH_EXECUTABLE", "ph.x")
    tr2_ph = params.get("tr2_ph", "1.0d-14")

    ph_in = (
        f"Phonons at Gamma\n"
        f" &inputph\n"
        f"   tr2_ph = {tr2_ph}\n"
        f"   prefix = '{prefix}'\n"
        f"   outdir = './tmp'\n"
        f"   fildyn = '{prefix}.dynG'\n"
        f"   ldisp = .false.\n"
        f" /\n"
        f"0.0 0.0 0.0\n"
    )
    result = _run_secondary_binary(
        run_dir, ph_executable, "ph.in", ph_in,
        walltime_minutes=int(params.get("walltime_minutes", 60)),
        cpus=int(params.get("cpus", 1)),
    )
    if not result.success:
        raise RuntimeError(f"ph.x failed: {result.error_message}")

    parsed = parse_ph_output(result.stdout_path)
    return {
        "phonon_frequencies_cm1": parsed.frequencies_cm1,
        "phonon_frequencies_thz": parsed.frequencies_thz,
        "phonon_n_modes": len(parsed.frequencies_cm1),
        "phonon_n_imaginary": parsed.n_imaginary,
    }


def _is_cubic_lattice(lattice: list, *, length_tol: float = 0.01, angle_tol_deg: float = 0.5) -> bool:
    """True when *lattice* is numerically cubic.

    Cubic = three lattice vectors of equal length AND all three
    pairwise angles = 90°. ``length_tol`` is relative (|Δa|/a);
    ``angle_tol_deg`` is absolute degrees. Defaults are tight enough
    to reject tetragonal / orthorhombic / trigonal / anything with a
    dipole, which is what Session 3.3's phonon guard needs.
    """
    import math

    if not lattice or len(lattice) != 3:
        return False
    try:
        a = [float(x) for x in lattice[0]]
        b = [float(x) for x in lattice[1]]
        c = [float(x) for x in lattice[2]]
    except (TypeError, ValueError):
        return False

    def _norm(v):
        return math.sqrt(sum(x * x for x in v))

    def _angle_deg(u, v):
        nu, nv = _norm(u), _norm(v)
        if nu == 0 or nv == 0:
            return math.nan
        cos = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv)))
        return math.degrees(math.acos(cos))

    la, lb, lc = _norm(a), _norm(b), _norm(c)
    if la == 0 or lb == 0 or lc == 0:
        return False
    if abs(la - lb) / la > length_tol:
        return False
    if abs(la - lc) / la > length_tol:
        return False
    for u, v in ((a, b), (b, c), (a, c)):
        if abs(_angle_deg(u, v) - 90.0) > angle_tol_deg:
            return False
    return True


def _assert_cubic_for_phonons(qe_struct: dict) -> None:
    """Guard: refuse non-cubic inputs for Γ-only phonon tasks.

    Γ-only ph.x calculations silently produce wrong frequencies for
    anisotropic materials (missing directional averaging) and polar
    semiconductors (missing LO–TO splitting). Session 3.3b will lift
    this restriction. Until then, bail with a clear error rather
    than hand back a plausible-looking lie.
    """
    if not _is_cubic_lattice(qe_struct.get("lattice") or []):
        raise RuntimeError(
            "dft_phonons_gamma: input structure is not cubic. Γ-only "
            "phonon calculations silently produce wrong frequencies for "
            "non-cubic / polar materials (LO–TO splitting is missing). "
            "This guard lands with Session 3.3 addendum; Session 3.3b "
            "will lift the restriction by handling dielectric_constant + "
            "non-analytic corrections properly."
        )


def _phonons_gamma_pre_pw(*, qe_struct: dict, params: dict) -> None:
    """Pre-pw hook for the phonon task — refuses non-cubic inputs."""
    _assert_cubic_for_phonons(qe_struct)


@celery_app.task(name="orion.dft.phonons_gamma", bind=True, acks_late=True)
def run_dft_phonons_gamma_job(self, job_id: str) -> Dict[str, Any]:
    """SCF + ph.x Γ-only phonon calculation.

    Guard: refuses non-cubic inputs until Session 3.3b — Γ-only ph.x
    is only physically meaningful for cubic materials. The guard runs
    inside JobLifecycle, after the structure loads but before pw.x
    starts, so rejected jobs fail fast **and** the failure is recorded
    on the job row.
    """
    return _run_pw_step(
        self, job_id,
        kind="dft_phonons_gamma",
        calculation="scf",
        pre_pw=_phonons_gamma_pre_pw,
        post_pw=_phonons_gamma_post_pw,
    )


# ---------------------------------------------------------------------------
# LAMMPS MD (NVT / NVE / NPT) — Session 4.2
# ---------------------------------------------------------------------------


def _run_lammps_step(
    self,
    job_id: str,
    *,
    kind: str,
    ensemble_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared body for LAMMPS MD Celery tasks.

    Given *job_id*:
      1. Load the target Structure.
      2. Build :class:`LAMMPSInputParams` from ``job.parameters`` (with
         an optional ensemble override for task-specific ensembles).
      3. Render via :func:`generate_lammps_input`.
      4. Run ``lmp`` through :func:`run_lammps`.
      5. Persist log thermo + RDF + MSD derived values; tarball the
         run dir to MinIO.

    Binary: reads ``ORION_LMP_PATH`` env (default ``lmp_serial``).
    """
    import os

    from backend.common.engines.lammps_input import (
        LAMMPSInputParams,
        default_registry,
        generate_lammps_input,
    )
    from backend.common.engines.lammps_run import (
        compute_msd,
        compute_rdf,
        parse_lammps_dump,
        run_lammps,
    )
    from backend.common.jobs import (
        DEFAULT_ARTIFACTS_BUCKET,
        build_minio_client,
        ensure_bucket,
    )
    from backend.common.workers import (
        ArtifactBundle,
        JobLifecycle,
        build_run_dir,
        tar_and_upload_run_dir,
    )
    from backend.common.workers.events import RedisPubSubEmitter

    engine, Session = _sync_session_for_worker()
    minio_client = None
    try:
        try:
            minio_client = build_minio_client()
            ensure_bucket(minio_client, DEFAULT_ARTIFACTS_BUCKET)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: MinIO setup failed: %s", kind, exc)
            minio_client = None

        emitter = RedisPubSubEmitter()

        with Session() as session:
            with JobLifecycle(
                job_id, session=session, emitter=emitter, worker_id=self.request.id,
            ) as lc:
                assert lc.job is not None
                structure_id = lc.job.structure_id  # type: ignore[attr-defined]
                parameters = lc.job.parameters or {}  # type: ignore[attr-defined]

                qe_struct = _load_structure_for_qe(session, structure_id)

                lammps_params_dict = {
                    k: v for k, v in parameters.items()
                    if k in LAMMPSInputParams.model_fields
                }
                if ensemble_override is not None:
                    lammps_params_dict["ensemble"] = ensemble_override
                params = LAMMPSInputParams(**lammps_params_dict)

                # Session 4.3b: pick up step-marker fields that the MD
                # workflow templates attach to drive aggregate analyzers.
                # These live outside LAMMPSInputParams so they don't
                # silently influence the rendered deck; they ride in the
                # job parameters dict and get echoed into outputs below.
                strain_voigt = parameters.get("strain_voigt")
                strain_value = parameters.get("strain_value")

                rendered = generate_lammps_input(
                    qe_struct, params, registry=default_registry,
                )

                run_dir = build_run_dir(f"md-{kind}-{job_id}")
                lmp_executable = os.getenv("ORION_LMP_PATH", "lmp_serial")
                walltime = int(parameters.get("walltime_minutes", 60))
                cpus = int(parameters.get("cpus", 1))

                run_result = run_lammps(
                    rendered,
                    run_dir,
                    lmp_executable=lmp_executable,
                    cpus=cpus,
                    walltime_minutes=walltime,
                )
                if not run_result.success:
                    raise RuntimeError(
                        f"{kind} lmp stage={run_result.stage}: {run_result.error_message}"
                    )

                # Derived analyzers over the trajectory.
                outputs: Dict[str, Any] = {
                    "engine": "lammps",
                    "kind": kind,
                    "forcefield": (
                        rendered.forcefield.name if rendered.forcefield else None
                    ),
                    "ensemble": params.ensemble,
                    "n_steps": rendered.n_steps,
                    "wall_time_seconds": (
                        run_result.log.wall_time_seconds if run_result.log else None
                    ),
                    "final_thermo": (
                        run_result.log.final_values() if run_result.log else {}
                    ),
                    # Session 4.3b: stamp the *target* T (from params)
                    # so temperature-sweep analyzers (melting,
                    # Arrhenius) can read it. The achieved T lives in
                    # final_thermo.Temp.
                    "temperature_k": params.temperature_k,
                    "pressure_bar": (
                        params.pressure_bar if params.ensemble == "npt" else None
                    ),
                }
                if strain_voigt is not None:
                    outputs["strain_voigt"] = strain_voigt
                if strain_value is not None:
                    outputs["strain_value"] = strain_value
                if run_result.dump_paths:
                    frames = list(parse_lammps_dump(run_result.dump_paths[0]))
                    if frames:
                        lx = min(frames[0].box_lengths())
                        rdf = compute_rdf(
                            frames, r_max_ang=lx / 2 - 0.1, n_bins=80,
                        )
                        first_peak = rdf.first_peak()
                        outputs["rdf_first_peak"] = (
                            {"r": first_peak[0], "g": first_peak[1]}
                            if first_peak else None
                        )
                        msd = compute_msd(
                            frames,
                            timestep_ps=params.timestep_fs / 1000.0,
                        )
                        outputs["msd_final_ang2"] = (
                            msd.msd_ang2[-1] if msd.msd_ang2 else None
                        )
                        outputs["diffusion_coefficient_ang2_per_ps"] = (
                            msd.diffusion_coefficient_ang2_per_ps()
                        )

                bundle = ArtifactBundle(
                    run_dir=run_dir,
                    job_id=str(job_id),
                    manifest={
                        "engine": "lammps",
                        "kind": kind,
                        "forcefield": outputs["forcefield"],
                        "ensemble": params.ensemble,
                    },
                )
                upload = tar_and_upload_run_dir(
                    bundle, minio_client=minio_client, bucket=DEFAULT_ARTIFACTS_BUCKET,
                )
                extra = dict(getattr(lc.job, "extra_metadata", None) or {})
                extra["artifact"] = {
                    "bucket": upload["bucket"],
                    "key": upload["key"],
                    "tarball": upload["tarball"],
                    "size_bytes": upload["size_bytes"],
                    "uploaded": upload["uploaded"],
                }
                lc.job.extra_metadata = extra  # type: ignore[attr-defined]

                lc.outputs = outputs

        return {
            "job_id": job_id,
            "status": "success",
            "kind": kind,
            "forcefield": outputs["forcefield"],
            "artifact_key": upload["key"],
        }
    finally:
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            pass


@celery_app.task(name="orion.md.nvt", bind=True, acks_late=True)
def run_md_nvt_job(self, job_id: str) -> Dict[str, Any]:
    """NVT (canonical) MD via LAMMPS.

    Ensemble forced to ``nvt_langevin`` unless the caller sets a
    different NVT flavor in ``job.parameters.ensemble``.
    """
    params = {}  # inspected inside _run_lammps_step from the job row
    return _run_lammps_step(self, job_id, kind="md_nvt")


@celery_app.task(name="orion.md.nve", bind=True, acks_late=True)
def run_md_nve_job(self, job_id: str) -> Dict[str, Any]:
    """NVE (microcanonical) MD via LAMMPS."""
    return _run_lammps_step(
        self, job_id, kind="md_nve", ensemble_override="nve",
    )


@celery_app.task(name="orion.md.npt", bind=True, acks_late=True)
def run_md_npt_job(self, job_id: str) -> Dict[str, Any]:
    """NPT (isothermal-isobaric) MD via LAMMPS."""
    return _run_lammps_step(
        self, job_id, kind="md_npt", ensemble_override="npt",
    )


# ---------------------------------------------------------------------------
# Session 5.3 — multiscale scaffolded tasks (loud PendingAnalyzerError)
# ---------------------------------------------------------------------------


def _raise_pending(job_id: str, kind: str, analyzer: str, tracker: str) -> None:
    """Mark the job FAILED with a PendingAnalyzerError and re-raise.

    The workflow tick propagates the FAILED state upward so the
    parent ``WorkflowRun`` also fails loudly. We keep the bookkeeping
    inline here rather than reusing ``_run_lammps_step`` because the
    full LAMMPS lifecycle (MinIO tarball, JobLifecycle with live
    metrics) isn't meaningful for a task that runs zero lines of
    physics.
    """
    from backend.common.reports import PendingAnalyzerError
    from backend.common.workers import JobLifecycle
    from backend.common.workers.events import RedisPubSubEmitter

    engine, Session = _sync_session_for_worker()
    try:
        with Session() as session:
            try:
                lc = JobLifecycle(
                    job_id,
                    session=session,
                    emitter=RedisPubSubEmitter(),
                    worker_id=None,
                )
                lc.__enter__()
            except Exception:  # noqa: BLE001
                # Even if the bookkeeping fails, raise the pending
                # error so the workflow tick sees the step failed.
                raise PendingAnalyzerError(analyzer, tracker=tracker)
            try:
                raise PendingAnalyzerError(analyzer, tracker=tracker)
            finally:
                lc.__exit__(*([None] * 3))  # the raise above bubbles up
    finally:
        try:
            engine.dispose()
        except Exception:  # noqa: BLE001
            pass


@celery_app.task(name="orion.dft.elastic", bind=True, acks_late=True)
def run_dft_elastic_job(self, job_id: str) -> Dict[str, Any]:
    """DFT elastic-tensor scaffolded task. Deferred to Phase 8.

    Raises :class:`~backend.common.reports.PendingAnalyzerError`. The
    multiscale DAG (``dft_to_md_to_continuum``) wires this in so
    submissions fail loudly at execution; Phase 8 will replace this
    with a real ±ε strain DFT sub-workflow + C_ij fit.
    """
    _raise_pending(
        job_id,
        kind="dft_elastic",
        analyzer="dft_elastic (elastic tensor via ±ε strain DFT)",
        tracker="Phase 8",
    )


@celery_app.task(name="orion.md.green_kubo_thermal", bind=True, acks_late=True)
def run_md_green_kubo_thermal_job(self, job_id: str) -> Dict[str, Any]:
    """MD Green-Kubo thermal-conductivity scaffolded task.

    Deferred to the Phase 4 Green-Kubo follow-up (flagged in the
    Phase 4.2 report alongside VACF→vDOS).
    """
    _raise_pending(
        job_id,
        kind="md_green_kubo_thermal",
        analyzer="md_green_kubo_thermal (κ from heat-flux ACF)",
        tracker="Phase 4 follow-up",
    )


@celery_app.task(name="orion.continuum.thermomechanical", bind=True, acks_late=True)
def run_continuum_thermomechanical_job(self, job_id: str) -> Dict[str, Any]:
    """Thermomechanical FEM scaffolded task (Session 5.3b).

    The linear elasticity + steady heat solvers exist (Session 5.1);
    the coupled thermoelastic ``σ = C:ε − α·ΔT·I`` run would combine
    them with the upstream C_ij (Phase 8) and κ (Phase 4 follow-up).
    Until both land, running this task raises PendingAnalyzerError.
    """
    _raise_pending(
        job_id,
        kind="continuum_thermomechanical",
        analyzer="continuum_thermomechanical (thermoelastic FEM)",
        tracker="Session 5.3b (depends on Phase 8 + Phase 4 follow-up)",
    )


# ---------------------------------------------------------------------------
# Workflow tick — Session 2.4
# ---------------------------------------------------------------------------


def _tick_workflow_runs_sync() -> Dict[str, Any]:
    """Advance every non-terminal WorkflowRun by one step.

    Runs under a sync engine so the Celery worker thread model is
    straightforward. Returns a dict suitable for logging / metrics.
    """
    from sqlalchemy import select

    from backend.common.workflows import tick
    from backend.common.workflows.celery_dispatcher import CeleryDispatcher
    from src.api.models import (
        SimulationJob,
        WorkflowRun,
        WorkflowRunStatus,
    )

    engine, Session = _sync_session_for_worker()
    advanced_total = 0
    completed_runs: list[str] = []
    per_run: list[Dict[str, Any]] = []
    try:
        with Session() as session:
            stmt = select(WorkflowRun).where(
                WorkflowRun.status.in_(
                    [
                        WorkflowRunStatus.PENDING.value,
                        WorkflowRunStatus.RUNNING.value,
                    ]
                )
            )
            for run in session.execute(stmt).scalars().all():
                dispatcher = CeleryDispatcher(session)
                result = tick(
                    run,
                    job_lookup=lambda jid: session.get(SimulationJob, jid),
                    dispatcher=dispatcher,
                )
                advanced_total += len(result.advanced)
                per_run.append({"run_id": str(run.id), **result.as_dict()})
                if result.aggregate_status in ("COMPLETED", "FAILED", "CANCELLED"):
                    completed_runs.append(str(run.id))
                    # Write the manifest on terminal.
                    from backend.common.workflows import build_workflow_manifest

                    run.manifest = build_workflow_manifest(
                        workflow_run_id=str(run.id),
                        name=run.name,
                        step_records=[
                            {
                                "step_id": s.step_id,
                                "status": s.status,
                                "job_id": str(s.simulation_job_id) if s.simulation_job_id else None,
                                "outputs": s.outputs,
                                "artifact": (
                                    (session.get(SimulationJob, s.simulation_job_id).extra_metadata or {}).get("artifact")
                                    if s.simulation_job_id else None
                                ),
                            }
                            for s in run.steps
                        ],
                    )
            session.commit()
    finally:
        engine.dispose()
    return {
        "advanced": advanced_total,
        "completed_runs": completed_runs,
        "per_run": per_run,
    }


@celery_app.task(name="orion.workflows.tick", bind=True, ignore_result=False)
def tick_workflow_runs(self) -> Dict[str, Any]:
    """Celery beat entry point: advance every workflow run by one step."""
    try:
        return _tick_workflow_runs_sync()
    except Exception as exc:  # noqa: BLE001
        logger.exception("workflow tick failed: %s", exc)
        return {"advanced": 0, "completed_runs": [], "per_run": [], "error": str(exc)}


@celery_app.task(name="orion.io.reap_stalled_jobs", bind=True, ignore_result=False)
def reap_stalled_jobs(self, stall_seconds: int = 120) -> Dict[str, Any]:
    """
    Scan for RUNNING jobs whose last update is older than *stall_seconds*
    and flip them to FAILED(reason=worker_lost).

    Wired as a beat-scheduled task in :mod:`src.worker.celery_app`;
    runs every 60 s by default.
    """
    try:
        result = _reap_stalled_sync(stall_seconds)
        if result["reaped"]:
            logger.warning("reaped %d stalled jobs: %s", len(result["reaped"]), result["reaped"])
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("reaper failed: %s", exc)
        return {"reaped": [], "errors": [str(exc)]}

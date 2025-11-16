"""
Example integration of Celery worker with FastAPI.

This shows how to integrate the worker tasks into your FastAPI application.
You can use this as a reference when implementing job submission endpoints.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.database import get_db
from src.api.models.simulation import SimulationJob, JobStatus
from src.worker.tasks import (
    run_simulation_job,
    run_dummy_job,
    update_job_status,
    cancel_job,
)

router = APIRouter(prefix="/simulations", tags=["simulations"])


@router.post("/{job_id}/submit")
async def submit_simulation_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a simulation job to the worker queue.

    This endpoint:
    1. Validates the job exists
    2. Checks job is in PENDING state
    3. Submits to Celery worker
    4. Updates job status to QUEUED
    5. Stores Celery task ID

    Args:
        job_id: UUID of the simulation job
        db: Database session

    Returns:
        Job details with Celery task ID
    """
    # Get job from database
    stmt = select(SimulationJob).where(SimulationJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check job is in correct state
    if job.status != JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not in PENDING state (current: {job.status})"
        )

    # Submit to Celery
    task = run_simulation_job.delay(str(job_id))

    # Update job in database
    job.status = JobStatus.QUEUED
    job.celery_task_id = task.id
    await db.commit()
    await db.refresh(job)

    return {
        "job_id": str(job.id),
        "celery_task_id": task.id,
        "status": job.status.value,
        "message": "Job submitted to worker queue",
    }


@router.post("/{job_id}/submit-dummy")
async def submit_dummy_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Submit a dummy simulation job for testing.

    This is useful for Session 1 testing without requiring
    a full simulation setup.

    Args:
        job_id: UUID of the simulation job
        db: Database session

    Returns:
        Job details with Celery task ID
    """
    # Get job from database
    stmt = select(SimulationJob).where(SimulationJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Submit dummy job
    task = run_dummy_job.delay(str(job_id))

    # Update job in database
    job.status = JobStatus.QUEUED
    job.celery_task_id = task.id
    await db.commit()
    await db.refresh(job)

    return {
        "job_id": str(job.id),
        "celery_task_id": task.id,
        "status": job.status.value,
        "message": "Dummy job submitted (for testing)",
    }


@router.post("/{job_id}/cancel")
async def cancel_simulation_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a running simulation job.

    This endpoint:
    1. Checks job exists and is cancellable
    2. Calls Celery task to cancel
    3. Updates job status to CANCELLED

    Args:
        job_id: UUID of the simulation job
        db: Database session

    Returns:
        Cancellation status
    """
    # Get job from database
    stmt = select(SimulationJob).where(SimulationJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check job is cancellable
    if job.is_terminal:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in terminal state: {job.status}"
        )

    # Cancel via Celery task
    # Note: We use delay() here so cancellation happens async
    cancel_task = cancel_job.delay(str(job_id))

    return {
        "job_id": str(job.id),
        "cancel_task_id": cancel_task.id,
        "message": "Cancellation request submitted",
    }


@router.get("/{job_id}/task-status")
async def get_task_status(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get Celery task status for a job.

    This provides the Celery task state, separate from
    the job status in the database.

    Useful for debugging and monitoring.

    Args:
        job_id: UUID of the simulation job
        db: Database session

    Returns:
        Celery task status details
    """
    # Get job from database
    stmt = select(SimulationJob).where(SimulationJob.id == job_id)
    result = await db.execute(stmt)
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.celery_task_id:
        raise HTTPException(
            status_code=400,
            detail="Job has not been submitted to worker yet"
        )

    # Get Celery task status
    from celery.result import AsyncResult
    from src.worker.celery_app import celery_app

    task_result = AsyncResult(job.celery_task_id, app=celery_app)

    return {
        "job_id": str(job.id),
        "celery_task_id": job.celery_task_id,
        "task_state": task_result.state,
        "task_ready": task_result.ready(),
        "task_successful": task_result.successful() if task_result.ready() else None,
        "task_failed": task_result.failed() if task_result.ready() else None,
        "task_result": task_result.result if task_result.ready() else None,
        "task_traceback": task_result.traceback if task_result.failed() else None,
    }


@router.post("/{job_id}/update-status")
async def update_simulation_status(
    job_id: UUID,
    status: str,
    progress: Optional[float] = None,
    current_step: Optional[str] = None,
    error_message: Optional[str] = None,
):
    """
    Update job status via Celery task.

    This is an alternative to direct database updates.
    Useful when you want status updates to go through
    the worker queue for consistency.

    Args:
        job_id: UUID of the simulation job
        status: New status (PENDING, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED)
        progress: Optional progress (0.0 to 1.0)
        current_step: Optional current step description
        error_message: Optional error message

    Returns:
        Update confirmation
    """
    # Build kwargs
    kwargs = {}
    if progress is not None:
        kwargs["progress"] = progress
    if current_step is not None:
        kwargs["current_step"] = current_step
    if error_message is not None:
        kwargs["error_message"] = error_message

    # Submit update task
    task = update_job_status.delay(str(job_id), status, **kwargs)

    return {
        "job_id": str(job_id),
        "update_task_id": task.id,
        "message": "Status update submitted",
    }


# Example: How to include this router in your FastAPI app
"""
# In src/api/app.py or wherever you set up your FastAPI app:

from fastapi import FastAPI
from src.worker.example_integration import router as worker_router

app = FastAPI()

# Include the worker integration router
app.include_router(worker_router, prefix="/api/v1")

# Now you have these endpoints:
# POST /api/v1/simulations/{job_id}/submit
# POST /api/v1/simulations/{job_id}/submit-dummy
# POST /api/v1/simulations/{job_id}/cancel
# GET  /api/v1/simulations/{job_id}/task-status
# POST /api/v1/simulations/{job_id}/update-status
"""


# Example: Testing the integration
"""
# 1. Start the FastAPI app
uvicorn src.api.app:app --reload

# 2. Start the Celery worker
celery -A src.worker.celery_app worker --loglevel=info

# 3. Submit a job via API
curl -X POST http://localhost:8000/api/v1/simulations/{job_id}/submit-dummy

# 4. Check task status
curl http://localhost:8000/api/v1/simulations/{job_id}/task-status

# 5. Monitor in Flower
celery -A src.worker.celery_app flower
# Open http://localhost:5555
"""

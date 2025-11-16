"""
Simulation jobs router for NANO-OS API.

Provides:
- Create and manage simulation jobs
- List jobs with filtering (status, material, structure)
- Get job details and results
- Cancel running jobs
- Retrieve job results
"""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload, joinedload
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, Structure, WorkflowTemplate, SimulationJob, SimulationResult
from ..schemas.simulation import (
    SimulationJobCreate,
    SimulationJobUpdate,
    SimulationJobResponse,
    SimulationResultResponse
)
from ..auth.security import get_current_active_user
from ..exceptions import (
    NotFoundError,
    ValidationError,
    AuthorizationError,
    ConflictError
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/jobs",
    tags=["simulation-jobs"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Job not found"}
    }
)


@router.post(
    "",
    response_model=SimulationJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create simulation job",
    description="""
    Create a new simulation job.

    A simulation job represents a calculation to be performed on a structure
    using a specific workflow template.

    Required:
    - structure_id: Structure to simulate
    - workflow_template_id: Template defining the calculation

    Optional:
    - name: Custom job name
    - priority: Job priority (0-20, default 5)
    - parameters: Override template default parameters
    - resources: Custom resource allocation

    The job is created in PENDING status and must be queued by the worker service.
    """,
    responses={
        201: {
            "description": "Job created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "423e4567-e89b-12d3-a456-426614174000",
                        "structure_id": "223e4567-e89b-12d3-a456-426614174000",
                        "workflow_template_id": "323e4567-e89b-12d3-a456-426614174000",
                        "name": "MoS2 relaxation",
                        "status": "PENDING",
                        "priority": 10,
                        "engine": "QE",
                        "submitted_at": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        403: {"description": "Not authorized to run simulations"},
        404: {"description": "Structure or template not found"}
    }
)
async def create_simulation_job(
    job_data: SimulationJobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> SimulationJobResponse:
    """
    Create a new simulation job.
    """
    logger.info(f"Creating simulation job: {job_data.name or 'Unnamed'}")

    # Check permission
    if not current_user.can_run_simulations():
        raise AuthorizationError("You don't have permission to run simulations")

    # Verify structure exists
    structure = await db.get(Structure, job_data.structure_id)
    if not structure:
        raise NotFoundError("Structure", job_data.structure_id)

    # Verify workflow template exists and is active
    template = await db.get(WorkflowTemplate, job_data.workflow_template_id)
    if not template or template.deleted_at:
        raise NotFoundError("WorkflowTemplate", job_data.workflow_template_id)

    if not template.is_active:
        raise ValidationError(
            f"Workflow template '{template.name}' is inactive",
            details={"template_id": str(template.id)}
        )

    # Merge parameters: template defaults + user overrides
    merged_parameters = {
        **template.default_parameters,
        **(job_data.parameters or {})
    }

    # Use template resources if not specified
    resources = job_data.resources or template.default_resources

    # Create job
    new_job = SimulationJob(
        structure_id=job_data.structure_id,
        workflow_template_id=job_data.workflow_template_id,
        name=job_data.name,
        description=job_data.description,
        status="PENDING",
        priority=job_data.priority,
        engine=template.engine,  # Copy from template
        parameters=merged_parameters,
        resources=resources,
        metadata=job_data.metadata or {},
        submitted_at=datetime.utcnow()
    )

    db.add(new_job)

    # Increment template usage count
    template.increment_usage()

    await db.commit()
    await db.refresh(new_job)

    logger.info(f"Simulation job created: {new_job.id} (status: {new_job.status})")

    return SimulationJobResponse.model_validate(new_job)


@router.post(
    "/{job_id}/submit",
    response_model=SimulationJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit job for execution",
    description="""
    Submit a PENDING job to the execution queue.

    This endpoint:
    - Changes job status from PENDING → QUEUED
    - Enqueues the job to Celery worker for execution
    - Returns the updated job

    Can only submit jobs in PENDING status.
    Jobs that are already QUEUED, RUNNING, or in terminal states cannot be resubmitted.

    The job will be picked up by the next available worker and executed according
    to its priority.
    """,
    responses={
        200: {
            "description": "Job submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "423e4567-e89b-12d3-a456-426614174000",
                        "status": "QUEUED",
                        "submitted_at": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        404: {"description": "Job not found"},
        409: {"description": "Job cannot be submitted in current state"}
    }
)
async def submit_simulation_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> SimulationJobResponse:
    """
    Submit a simulation job to the execution queue.
    """
    logger.info(f"Submitting simulation job: {job_id}")

    # Get job
    job = await db.get(SimulationJob, job_id)
    if not job:
        raise NotFoundError("SimulationJob", job_id)

    # Validate job is in PENDING status
    if job.status != "PENDING":
        raise ConflictError(
            message=f"Job cannot be submitted in current state: {job.status.value}",
            details={
                "job_id": str(job_id),
                "current_status": job.status.value,
                "expected_status": "PENDING"
            }
        )

    # Import here to avoid circular dependency
    from src.worker.tasks import run_simulation_job as run_simulation_task

    # Update job status to QUEUED
    job.status = "QUEUED"
    job.updated_at = datetime.utcnow()

    # Enqueue task to Celery
    task = run_simulation_task.apply_async(args=[str(job_id)])
    job.celery_task_id = task.id

    await db.commit()
    await db.refresh(job)

    logger.info(f"Job {job_id} submitted successfully (task_id: {task.id})")

    return SimulationJobResponse.model_validate(job)


@router.get(
    "/{job_id}/status",
    summary="Get job status",
    description="""
    Get lightweight status information for a simulation job.

    This is a minimal endpoint optimized for polling and monitoring.
    Returns only essential status fields without loading full job details or relationships.

    Use this for:
    - Real-time status monitoring
    - Progress tracking
    - Polling for job completion

    For full job details, use GET /jobs/{job_id}
    """,
    responses={
        200: {
            "description": "Job status",
            "content": {
                "application/json": {
                    "example": {
                        "id": "423e4567-e89b-12d3-a456-426614174000",
                        "status": "RUNNING",
                        "progress": 0.45,
                        "current_step": "SCF iteration",
                        "error_message": None
                    }
                }
            }
        },
        404: {"description": "Job not found"}
    }
)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get lightweight job status information.
    """
    logger.debug(f"Fetching status for job: {job_id}")

    # Query only the fields we need (no joins, no relationships)
    query = select(
        SimulationJob.id,
        SimulationJob.status,
        SimulationJob.progress,
        SimulationJob.current_step,
        SimulationJob.error_message
    ).where(SimulationJob.id == job_id)

    result = await db.execute(query)
    row = result.first()

    if not row:
        raise NotFoundError("SimulationJob", job_id)

    return {
        "id": row[0],
        "status": row[1].value,
        "progress": row[2],
        "current_step": row[3],
        "error_message": row[4]
    }


@router.get(
    "",
    response_model=List[SimulationJobResponse],
    summary="List simulation jobs",
    description="""
    Get list of simulation jobs with optional filtering.

    Filtering options:
    - status: Filter by job status (PENDING, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED)
    - structure_id: Filter by structure
    - material_id: Filter by material (via structure)
    - engine: Filter by simulation engine
    - priority_min: Minimum priority

    Pagination:
    - limit: Maximum number of results (default 100, max 500)
    - offset: Number of results to skip

    Results are sorted by submission time (newest first).
    """,
    responses={
        200: {
            "description": "List of simulation jobs",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "423e4567-e89b-12d3-a456-426614174000",
                            "name": "MoS2 relaxation",
                            "status": "COMPLETED",
                            "priority": 10,
                            "engine": "QE",
                            "submitted_at": "2024-01-15T10:30:00Z",
                            "finished_at": "2024-01-15T11:00:00Z"
                        }
                    ]
                }
            }
        }
    }
)
async def list_simulation_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    structure_id: Optional[uuid.UUID] = Query(None, description="Filter by structure ID"),
    material_id: Optional[uuid.UUID] = Query(None, description="Filter by material ID"),
    engine: Optional[str] = Query(None, description="Filter by engine"),
    priority_min: Optional[int] = Query(None, ge=0, le=20, description="Minimum priority"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[SimulationJobResponse]:
    """
    Get list of simulation jobs with filtering.
    """
    logger.debug(f"Listing simulation jobs: status={status}, limit={limit}")

    # Build query
    query = select(SimulationJob)

    # Apply filters
    if status:
        query = query.where(SimulationJob.status == status.upper())

    if structure_id:
        query = query.where(SimulationJob.structure_id == structure_id)

    if material_id:
        # Join with Structure to filter by material
        query = query.join(Structure).where(Structure.material_id == material_id)

    if engine:
        query = query.where(SimulationJob.engine == engine)

    if priority_min is not None:
        query = query.where(SimulationJob.priority >= priority_min)

    # Sort by submission time (newest first)
    query = query.order_by(SimulationJob.submitted_at.desc())

    # Apply pagination
    query = query.offset(offset).limit(limit)

    # Load relationships
    query = query.options(
        selectinload(SimulationJob.structure),
        selectinload(SimulationJob.workflow_template)
    )

    # Execute
    result = await db.execute(query)
    jobs = result.scalars().all()

    return [SimulationJobResponse.model_validate(j) for j in jobs]


@router.get(
    "/{job_id}",
    response_model=SimulationJobResponse,
    summary="Get simulation job by ID",
    description="""
    Get detailed information about a specific simulation job.

    Includes:
    - Job metadata and status
    - Parameters and resources
    - Timing information
    - Progress tracking
    - Error messages (if failed)
    - Worker information

    To get job results, use GET /jobs/{job_id}/results
    """,
    responses={
        200: {
            "description": "Job details",
            "content": {
                "application/json": {
                    "example": {
                        "id": "423e4567-e89b-12d3-a456-426614174000",
                        "structure_id": "223e4567-e89b-12d3-a456-426614174000",
                        "workflow_template_id": "323e4567-e89b-12d3-a456-426614174000",
                        "name": "MoS2 relaxation",
                        "status": "RUNNING",
                        "priority": 10,
                        "engine": "QE",
                        "progress": 0.45,
                        "current_step": "SCF iteration",
                        "submitted_at": "2024-01-15T10:30:00Z",
                        "started_at": "2024-01-15T10:32:00Z"
                    }
                }
            }
        },
        404: {"description": "Job not found"}
    }
)
async def get_simulation_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> SimulationJobResponse:
    """
    Get simulation job by ID.
    """
    logger.debug(f"Fetching simulation job: {job_id}")

    # Load with relationships
    query = select(SimulationJob).where(
        SimulationJob.id == job_id
    ).options(
        selectinload(SimulationJob.structure),
        selectinload(SimulationJob.workflow_template),
        selectinload(SimulationJob.results)
    )

    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise NotFoundError("SimulationJob", job_id)

    return SimulationJobResponse.model_validate(job)


@router.put(
    "/{job_id}",
    response_model=SimulationJobResponse,
    summary="Update simulation job",
    description="""
    Update simulation job metadata.

    Typically used for:
    - Changing job priority (before it starts)
    - Updating name/description
    - Setting custom metadata

    Note: Cannot change structure, template, or parameters once created.
    Status changes are typically done by the worker service, not users.
    """,
    responses={
        200: {"description": "Job updated successfully"},
        404: {"description": "Job not found"},
        409: {"description": "Job cannot be modified in current state"}
    }
)
async def update_simulation_job(
    job_id: uuid.UUID,
    job_data: SimulationJobUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> SimulationJobResponse:
    """
    Update simulation job metadata.
    """
    logger.info(f"Updating simulation job: {job_id}")

    # Get job
    job = await db.get(SimulationJob, job_id)
    if not job:
        raise NotFoundError("SimulationJob", job_id)

    # Check if job can be modified
    if job.is_terminal:
        raise ConflictError(
            message="Cannot modify job in terminal state",
            details={"status": job.status.value, "job_id": str(job_id)}
        )

    # Update fields
    update_data = job_data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        # Don't allow status changes from users (only workers)
        if field == "status" and not current_user.is_admin:
            continue
        setattr(job, field, value)

    job.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(job)

    logger.info(f"Simulation job updated: {job_id}")

    return SimulationJobResponse.model_validate(job)


@router.delete(
    "/{job_id}/cancel",
    status_code=status.HTTP_200_OK,
    response_model=SimulationJobResponse,
    summary="Cancel simulation job",
    description="""
    Cancel a running or queued simulation job.

    This endpoint:
    - Sets job status to CANCELLED
    - Signals the worker to stop execution
    - Preserves partial results if available

    Can only cancel jobs in PENDING, QUEUED, or RUNNING status.
    Already completed or failed jobs cannot be cancelled.
    """,
    responses={
        200: {
            "description": "Job cancelled successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "423e4567-e89b-12d3-a456-426614174000",
                        "status": "CANCELLED",
                        "finished_at": "2024-01-15T10:45:00Z"
                    }
                }
            }
        },
        404: {"description": "Job not found"},
        409: {"description": "Job cannot be cancelled in current state"}
    }
)
async def cancel_simulation_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> SimulationJobResponse:
    """
    Cancel a simulation job.
    """
    logger.info(f"Cancelling simulation job: {job_id}")

    # Get job
    job = await db.get(SimulationJob, job_id)
    if not job:
        raise NotFoundError("SimulationJob", job_id)

    # Check if job can be cancelled
    if job.is_terminal:
        raise ConflictError(
            message=f"Job already in terminal state: {job.status.value}",
            details={"status": job.status.value, "job_id": str(job_id)}
        )

    # Update job status
    job.status = "CANCELLED"
    job.finished_at = datetime.utcnow()
    job.error_message = f"Cancelled by user {current_user.username}"
    job.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(job)

    logger.info(f"Simulation job cancelled: {job_id}")

    # TODO: Signal worker to stop execution (via Celery, Redis, etc.)

    return SimulationJobResponse.model_validate(job)


@router.get(
    "/{job_id}/results",
    response_model=SimulationResultResponse,
    summary="Get simulation results",
    description="""
    Get results for a completed simulation job.

    Returns:
    - Summary metrics (energy, band gap, etc.)
    - Artifact paths (output files, plots)
    - Convergence information
    - Quality scores

    Only available for jobs with COMPLETED status.
    Failed jobs may have partial results.
    """,
    responses={
        200: {
            "description": "Simulation results",
            "content": {
                "application/json": {
                    "example": {
                        "id": "523e4567-e89b-12d3-a456-426614174000",
                        "simulation_job_id": "423e4567-e89b-12d3-a456-426614174000",
                        "summary": {
                            "total_energy": -123.45,
                            "band_gap": 1.2,
                            "forces_max": 0.001
                        },
                        "convergence_reached": True,
                        "quality_score": 0.95,
                        "created_at": "2024-01-15T11:00:00Z"
                    }
                }
            }
        },
        404: {"description": "Job or results not found"},
        409: {"description": "Job not completed yet"}
    }
)
async def get_simulation_results(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> SimulationResultResponse:
    """
    Get simulation results for a job.
    """
    logger.debug(f"Fetching results for job: {job_id}")

    # Get job with results
    query = select(SimulationJob).where(
        SimulationJob.id == job_id
    ).options(selectinload(SimulationJob.results))

    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise NotFoundError("SimulationJob", job_id)

    # Check if results exist
    if not job.results:
        # Check if job is completed
        if job.status.value == "COMPLETED":
            raise NotFoundError(
                "SimulationResult",
                f"No results found for completed job {job_id}"
            )
        else:
            raise ConflictError(
                message=f"Job not completed yet (status: {job.status.value})",
                details={"job_id": str(job_id), "status": job.status.value}
            )

    return SimulationResultResponse.model_validate(job.results)


@router.get(
    "/status/summary",
    summary="Get job status summary",
    description="""
    Get summary statistics of jobs by status.

    Returns count of jobs in each status:
    - PENDING
    - QUEUED
    - RUNNING
    - COMPLETED
    - FAILED
    - CANCELLED

    Useful for dashboard and monitoring.
    """,
    responses={
        200: {
            "description": "Job status summary",
            "content": {
                "application/json": {
                    "example": {
                        "PENDING": 5,
                        "QUEUED": 10,
                        "RUNNING": 3,
                        "COMPLETED": 142,
                        "FAILED": 8,
                        "CANCELLED": 2
                    }
                }
            }
        }
    }
)
async def get_job_status_summary(
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get summary of jobs by status.
    """
    # Count jobs by status
    result = await db.execute(
        select(
            SimulationJob.status,
            func.count(SimulationJob.id)
        ).group_by(SimulationJob.status)
    )

    summary = {row[0].value: row[1] for row in result.all()}

    # Ensure all statuses are present
    all_statuses = ["PENDING", "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]
    for status in all_statuses:
        if status not in summary:
            summary[status] = 0

    return summary

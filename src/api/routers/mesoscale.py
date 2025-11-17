"""
Mesoscale simulation router for NANO-OS API.

Provides endpoints for:
- Creating mesoscale simulation jobs
- Submitting jobs for execution
- Retrieving job details and status
- Listing jobs with filtering
- Retrieving simulation results

Mesoscale simulations bridge atomistic and continuum scales, modeling
microstructure evolution using phase field, Monte Carlo, or kinetic Monte Carlo methods.
"""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.orm import selectinload
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, MesoscaleSimulationJob, MesoscaleSimulationResult, MultiscaleJobStatus
from ..schemas.multiscale import (
    MesoscaleJobCreate,
    MesoscaleJobResponse,
    MesoscaleResultResponse
)
from ..auth.security import get_current_active_user
from ..exceptions import (
    NotFoundError,
    ValidationError,
    AuthorizationError,
    ConflictError
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/mesoscale",
    tags=["mesoscale-simulations"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Job not found"}
    }
)


@router.post(
    "/jobs",
    response_model=MesoscaleJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create mesoscale simulation job",
    description="""
    Create a new mesoscale simulation job.

    Mesoscale simulations model microstructure evolution at the micrometer to
    millimeter scale using:
    - PHASE_FIELD: Phase field modeling for microstructure evolution
    - MONTE_CARLO: Monte Carlo for grain growth and phase transitions
    - KMC: Kinetic Monte Carlo for diffusion and nucleation processes

    Required:
    - name: Descriptive job name
    - engine_type: Simulation engine (PHASE_FIELD, MONTE_CARLO, KMC)
    - parameters: Simulation parameters (domain size, timesteps, material properties)

    Optional:
    - parent_atomic_job_id: Link to atomistic simulation providing input parameters

    The job is created in PENDING status and must be submitted separately.
    """,
    responses={
        201: {
            "description": "Job created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "423e4567-e89b-12d3-a456-426614174000",
                        "name": "MoS2 grain growth",
                        "engine_type": "MONTE_CARLO",
                        "status": "PENDING",
                        "created_at": "2025-11-17T10:00:00Z"
                    }
                }
            }
        },
        400: {"description": "Invalid parameters"},
        403: {"description": "Not authorized to create simulations"}
    }
)
async def create_mesoscale_job(
    job_data: MesoscaleJobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MesoscaleJobResponse:
    """
    Create a new mesoscale simulation job.
    """
    logger.info(f"Creating mesoscale job: {job_data.name} (engine: {job_data.engine_type})")

    # Validate engine type
    valid_engines = ["PHASE_FIELD", "MONTE_CARLO", "KMC"]
    if job_data.engine_type not in valid_engines:
        raise ValidationError(
            f"Invalid engine_type: {job_data.engine_type}. Must be one of: {', '.join(valid_engines)}",
            details={"valid_engines": valid_engines}
        )

    # Create job
    new_job = MesoscaleSimulationJob(
        owner_id=current_user.id,
        parent_atomic_job_id=job_data.parent_atomic_job_id,
        name=job_data.name,
        engine_type=job_data.engine_type,
        status=MultiscaleJobStatus.PENDING,
        parameters=job_data.parameters,
        created_at=datetime.utcnow()
    )

    db.add(new_job)
    await db.commit()
    await db.refresh(new_job)

    logger.info(f"Mesoscale job created: {new_job.id} (status: {new_job.status})")

    # Build response
    return MesoscaleJobResponse(
        id=new_job.id,
        owner_id=new_job.owner_id,
        parent_atomic_job_id=new_job.parent_atomic_job_id,
        name=new_job.name,
        engine_type=new_job.engine_type.value,
        status=new_job.status.value,
        parameters=new_job.parameters,
        created_at=new_job.created_at,
        started_at=new_job.started_at,
        completed_at=new_job.completed_at,
        celery_task_id=new_job.celery_task_id,
        duration_seconds=None
    )


@router.post(
    "/jobs/{job_id}/submit",
    response_model=MesoscaleJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit mesoscale job for execution",
    description="""
    Submit a PENDING mesoscale job to the execution queue.

    This endpoint:
    - Changes job status from PENDING → QUEUED
    - Enqueues the job to Celery worker for execution (stub for now)
    - Returns the updated job

    Can only submit jobs in PENDING status.
    Jobs that are already QUEUED, RUNNING, or in terminal states cannot be resubmitted.

    The job will be picked up by the next available mesoscale simulation worker.
    """,
    responses={
        200: {
            "description": "Job submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "423e4567-e89b-12d3-a456-426614174000",
                        "status": "QUEUED",
                        "celery_task_id": "task-abc-123"
                    }
                }
            }
        },
        404: {"description": "Job not found"},
        409: {"description": "Job cannot be submitted in current state"}
    }
)
async def submit_mesoscale_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MesoscaleJobResponse:
    """
    Submit a mesoscale job to the execution queue.
    """
    logger.info(f"Submitting mesoscale job: {job_id}")

    # Get job
    job = await db.get(MesoscaleSimulationJob, job_id)
    if not job:
        raise NotFoundError("MesoscaleSimulationJob", job_id)

    # Check ownership (non-admin users can only submit their own jobs)
    if not current_user.is_admin and job.owner_id != current_user.id:
        raise AuthorizationError("You can only submit your own jobs")

    # Validate job is in PENDING status
    if job.status != MultiscaleJobStatus.PENDING:
        raise ConflictError(
            message=f"Job cannot be submitted in current state: {job.status.value}",
            details={
                "job_id": str(job_id),
                "current_status": job.status.value,
                "expected_status": "PENDING"
            }
        )

    # Update job status to QUEUED
    job.status = MultiscaleJobStatus.QUEUED

    # TODO: Enqueue task to Celery
    # For now, just set a stub task ID
    job.celery_task_id = f"mesoscale-task-{job_id}"

    await db.commit()
    await db.refresh(job)

    logger.info(f"Mesoscale job {job_id} submitted successfully (task_id: {job.celery_task_id})")

    # Calculate duration if completed
    duration = None
    if job.started_at and job.completed_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    return MesoscaleJobResponse(
        id=job.id,
        owner_id=job.owner_id,
        parent_atomic_job_id=job.parent_atomic_job_id,
        name=job.name,
        engine_type=job.engine_type.value,
        status=job.status.value,
        parameters=job.parameters,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        celery_task_id=job.celery_task_id,
        duration_seconds=duration
    )


@router.get(
    "/jobs/{job_id}",
    response_model=MesoscaleJobResponse,
    summary="Get mesoscale job by ID",
    description="""
    Get detailed information about a specific mesoscale simulation job.

    Includes:
    - Job metadata and status
    - Parameters and engine configuration
    - Timing information
    - Parent job references
    - Worker information

    To get job results, use GET /mesoscale/jobs/{job_id}/results
    """,
    responses={
        200: {
            "description": "Job details",
            "content": {
                "application/json": {
                    "example": {
                        "id": "423e4567-e89b-12d3-a456-426614174000",
                        "name": "MoS2 grain growth",
                        "engine_type": "MONTE_CARLO",
                        "status": "RUNNING",
                        "created_at": "2025-11-17T10:00:00Z",
                        "started_at": "2025-11-17T10:01:00Z"
                    }
                }
            }
        },
        404: {"description": "Job not found"},
        403: {"description": "Not authorized to view this job"}
    }
)
async def get_mesoscale_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MesoscaleJobResponse:
    """
    Get mesoscale job by ID.
    """
    logger.debug(f"Fetching mesoscale job: {job_id}")

    # Load job with relationships
    query = select(MesoscaleSimulationJob).where(
        MesoscaleSimulationJob.id == job_id
    ).options(
        selectinload(MesoscaleSimulationJob.results)
    )

    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise NotFoundError("MesoscaleSimulationJob", job_id)

    # Check ownership (non-admin users can only view their own jobs)
    if not current_user.is_admin and job.owner_id != current_user.id:
        raise AuthorizationError("You can only view your own jobs")

    # Calculate duration if completed
    duration = None
    if job.started_at and job.completed_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    return MesoscaleJobResponse(
        id=job.id,
        owner_id=job.owner_id,
        parent_atomic_job_id=job.parent_atomic_job_id,
        name=job.name,
        engine_type=job.engine_type.value,
        status=job.status.value,
        parameters=job.parameters,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        celery_task_id=job.celery_task_id,
        duration_seconds=duration
    )


@router.get(
    "/jobs",
    response_model=List[MesoscaleJobResponse],
    summary="List mesoscale simulation jobs",
    description="""
    Get list of mesoscale simulation jobs with optional filtering.

    Filtering options:
    - status: Filter by job status (PENDING, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED)
    - engine_type: Filter by simulation engine (PHASE_FIELD, MONTE_CARLO, KMC)

    Pagination:
    - skip: Number of results to skip (default 0)
    - limit: Maximum number of results (default 100, max 500)

    Authorization:
    - Non-admin users can only see their own jobs
    - Admin users can see all jobs

    Results are sorted by creation time (newest first).
    """,
    responses={
        200: {
            "description": "List of mesoscale jobs",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "423e4567-e89b-12d3-a456-426614174000",
                            "name": "MoS2 grain growth",
                            "engine_type": "MONTE_CARLO",
                            "status": "COMPLETED",
                            "created_at": "2025-11-17T10:00:00Z"
                        }
                    ]
                }
            }
        }
    }
)
async def list_mesoscale_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    engine_type: Optional[str] = Query(None, description="Filter by engine type"),
    skip: int = Query(0, ge=0, description="Results to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[MesoscaleJobResponse]:
    """
    Get list of mesoscale jobs with filtering.
    """
    logger.debug(f"Listing mesoscale jobs: status={status}, engine_type={engine_type}, limit={limit}")

    # Build query
    query = select(MesoscaleSimulationJob)

    # Filter by owner (non-admin users can only see their own jobs)
    if not current_user.is_admin:
        query = query.where(MesoscaleSimulationJob.owner_id == current_user.id)

    # Apply filters
    if status:
        query = query.where(MesoscaleSimulationJob.status == status.upper())

    if engine_type:
        query = query.where(MesoscaleSimulationJob.engine_type == engine_type.upper())

    # Sort by creation time (newest first)
    query = query.order_by(desc(MesoscaleSimulationJob.created_at))

    # Apply pagination
    query = query.offset(skip).limit(limit)

    # Load relationships
    query = query.options(selectinload(MesoscaleSimulationJob.results))

    # Execute
    result = await db.execute(query)
    jobs = result.scalars().all()

    # Build responses
    responses = []
    for job in jobs:
        duration = None
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()

        responses.append(MesoscaleJobResponse(
            id=job.id,
            owner_id=job.owner_id,
            parent_atomic_job_id=job.parent_atomic_job_id,
            name=job.name,
            engine_type=job.engine_type.value,
            status=job.status.value,
            parameters=job.parameters,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            celery_task_id=job.celery_task_id,
            duration_seconds=duration
        ))

    return responses


@router.get(
    "/jobs/{job_id}/results",
    response_model=MesoscaleResultResponse,
    summary="Get mesoscale simulation results",
    description="""
    Get results for a completed mesoscale simulation job.

    Returns:
    - Microstructure metrics (grain size, phase fractions, grain boundary density)
    - Domain information (simulation domain dimensions)
    - Timestep information
    - Output file paths (VTK, HDF5, images)

    Only available for jobs with COMPLETED status.
    Failed jobs may have partial results.
    """,
    responses={
        200: {
            "description": "Simulation results",
            "content": {
                "application/json": {
                    "example": {
                        "id": "623e4567-e89b-12d3-a456-426614174000",
                        "job_id": "423e4567-e89b-12d3-a456-426614174000",
                        "microstructure_metrics": {
                            "average_grain_size": 15.3,
                            "phase_fractions": {"alpha": 0.6, "beta": 0.4}
                        },
                        "timesteps": 10000,
                        "created_at": "2025-11-17T10:30:00Z"
                    }
                }
            }
        },
        404: {"description": "Job or results not found"},
        403: {"description": "Not authorized to view results"},
        409: {"description": "Job not completed yet"}
    }
)
async def get_mesoscale_results(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MesoscaleResultResponse:
    """
    Get mesoscale simulation results for a job.
    """
    logger.debug(f"Fetching results for mesoscale job: {job_id}")

    # Get job with results
    query = select(MesoscaleSimulationJob).where(
        MesoscaleSimulationJob.id == job_id
    ).options(selectinload(MesoscaleSimulationJob.results))

    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise NotFoundError("MesoscaleSimulationJob", job_id)

    # Check ownership (non-admin users can only view their own job results)
    if not current_user.is_admin and job.owner_id != current_user.id:
        raise AuthorizationError("You can only view results for your own jobs")

    # Check if results exist
    if not job.results:
        # Check if job is completed
        if job.status == MultiscaleJobStatus.COMPLETED:
            raise NotFoundError(
                "MesoscaleSimulationResult",
                f"No results found for completed job {job_id}"
            )
        else:
            raise ConflictError(
                message=f"Job not completed yet (status: {job.status.value})",
                details={"job_id": str(job_id), "status": job.status.value}
            )

    return MesoscaleResultResponse(
        id=job.results.id,
        job_id=job.results.job_id,
        microstructure_metrics=job.results.microstructure_metrics,
        domain_size=job.results.domain_size,
        timesteps=job.results.timesteps,
        output_files=job.results.output_files,
        created_at=job.results.created_at
    )

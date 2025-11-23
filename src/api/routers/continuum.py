"""
Continuum simulation router for NANO-OS API.

Provides endpoints for:
- Creating continuum simulation jobs
- Submitting jobs for execution
- Retrieving job details and status
- Listing jobs with filtering
- Retrieving simulation results

Continuum simulations operate at the macroscopic scale, modeling material
behavior using FEM, FVM, or BEM methods.
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
from ..models import User, ContinuumSimulationJob, ContinuumSimulationResult, MultiscaleJobStatus
from ..schemas.multiscale import (
    ContinuumJobCreate,
    ContinuumJobResponse,
    ContinuumResultResponse
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
    prefix="/continuum",
    tags=["continuum-simulations"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Job not found"}
    }
)


@router.post(
    "/jobs",
    response_model=ContinuumJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create continuum simulation job",
    description="""
    Create a new continuum simulation job.

    Continuum simulations model macroscopic behavior at the millimeter to meter
    scale using:
    - FEM: Finite Element Method for structural analysis
    - FVM: Finite Volume Method for fluid dynamics and heat transfer
    - BEM: Boundary Element Method for elasticity and acoustics

    Required:
    - name: Descriptive job name
    - engine_type: Simulation engine (FEM, FVM, BEM)
    - parameters: Simulation parameters (mesh, boundary conditions, material properties)

    Optional:
    - parent_mesoscale_job_id: Link to mesoscale simulation providing material properties

    The job is created in PENDING status and must be submitted separately.
    """,
    responses={
        201: {
            "description": "Job created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "723e4567-e89b-12d3-a456-426614174000",
                        "name": "Tensile test FEM",
                        "engine_type": "FEM",
                        "status": "PENDING",
                        "created_at": "2025-11-17T11:00:00Z"
                    }
                }
            }
        },
        400: {"description": "Invalid parameters"},
        403: {"description": "Not authorized to create simulations"}
    }
)
async def create_continuum_job(
    job_data: ContinuumJobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ContinuumJobResponse:
    """
    Create a new continuum simulation job.
    """
    logger.info(f"Creating continuum job: {job_data.name} (engine: {job_data.engine_type})")

    # Validate engine type
    valid_engines = ["FEM", "FVM", "BEM"]
    if job_data.engine_type not in valid_engines:
        raise ValidationError(
            f"Invalid engine_type: {job_data.engine_type}. Must be one of: {', '.join(valid_engines)}",
            details={"valid_engines": valid_engines}
        )

    # Create job
    new_job = ContinuumSimulationJob(
        owner_id=current_user.id,
        parent_mesoscale_job_id=job_data.parent_mesoscale_job_id,
        name=job_data.name,
        engine_type=job_data.engine_type,
        status=MultiscaleJobStatus.PENDING,
        parameters=job_data.parameters,
        created_at=datetime.utcnow()
    )

    db.add(new_job)
    await db.commit()
    await db.refresh(new_job)

    logger.info(f"Continuum job created: {new_job.id} (status: {new_job.status})")

    # Build response
    return ContinuumJobResponse(
        id=new_job.id,
        owner_id=new_job.owner_id,
        parent_mesoscale_job_id=new_job.parent_mesoscale_job_id,
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
    response_model=ContinuumJobResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit continuum job for execution",
    description="""
    Submit a PENDING continuum job to the execution queue.

    This endpoint:
    - Changes job status from PENDING → QUEUED
    - Enqueues the job to Celery worker for execution (stub for now)
    - Returns the updated job

    Can only submit jobs in PENDING status.
    Jobs that are already QUEUED, RUNNING, or in terminal states cannot be resubmitted.

    The job will be picked up by the next available continuum simulation worker.
    """,
    responses={
        200: {
            "description": "Job submitted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "723e4567-e89b-12d3-a456-426614174000",
                        "status": "QUEUED",
                        "celery_task_id": "task-def-456"
                    }
                }
            }
        },
        404: {"description": "Job not found"},
        409: {"description": "Job cannot be submitted in current state"}
    }
)
async def submit_continuum_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ContinuumJobResponse:
    """
    Submit a continuum job to the execution queue.
    """
    logger.info(f"Submitting continuum job: {job_id}")

    # Get job
    job = await db.get(ContinuumSimulationJob, job_id)
    if not job:
        raise NotFoundError("ContinuumSimulationJob", job_id)

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

    # Enqueue task to Celery
    from src.worker.tasks import run_continuum_simulation

    celery_task = run_continuum_simulation.apply_async(
        args=[str(job_id)],
        task_id=f"continuum-{job_id}"
    )
    job.celery_task_id = celery_task.id

    await db.commit()
    await db.refresh(job)

    logger.info(f"Continuum job {job_id} submitted successfully (task_id: {job.celery_task_id})")

    # Calculate duration if completed
    duration = None
    if job.started_at and job.completed_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    return ContinuumJobResponse(
        id=job.id,
        owner_id=job.owner_id,
        parent_mesoscale_job_id=job.parent_mesoscale_job_id,
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
    response_model=ContinuumJobResponse,
    summary="Get continuum job by ID",
    description="""
    Get detailed information about a specific continuum simulation job.

    Includes:
    - Job metadata and status
    - Parameters and engine configuration
    - Timing information
    - Parent job references
    - Worker information

    To get job results, use GET /continuum/jobs/{job_id}/results
    """,
    responses={
        200: {
            "description": "Job details",
            "content": {
                "application/json": {
                    "example": {
                        "id": "723e4567-e89b-12d3-a456-426614174000",
                        "name": "Tensile test FEM",
                        "engine_type": "FEM",
                        "status": "RUNNING",
                        "created_at": "2025-11-17T11:00:00Z",
                        "started_at": "2025-11-17T11:01:00Z"
                    }
                }
            }
        },
        404: {"description": "Job not found"},
        403: {"description": "Not authorized to view this job"}
    }
)
async def get_continuum_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ContinuumJobResponse:
    """
    Get continuum job by ID.
    """
    logger.debug(f"Fetching continuum job: {job_id}")

    # Load job with relationships
    query = select(ContinuumSimulationJob).where(
        ContinuumSimulationJob.id == job_id
    ).options(
        selectinload(ContinuumSimulationJob.results)
    )

    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise NotFoundError("ContinuumSimulationJob", job_id)

    # Check ownership (non-admin users can only view their own jobs)
    if not current_user.is_admin and job.owner_id != current_user.id:
        raise AuthorizationError("You can only view your own jobs")

    # Calculate duration if completed
    duration = None
    if job.started_at and job.completed_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    return ContinuumJobResponse(
        id=job.id,
        owner_id=job.owner_id,
        parent_mesoscale_job_id=job.parent_mesoscale_job_id,
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
    response_model=List[ContinuumJobResponse],
    summary="List continuum simulation jobs",
    description="""
    Get list of continuum simulation jobs with optional filtering.

    Filtering options:
    - status: Filter by job status (PENDING, QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED)
    - engine_type: Filter by simulation engine (FEM, FVM, BEM)

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
            "description": "List of continuum jobs",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "723e4567-e89b-12d3-a456-426614174000",
                            "name": "Tensile test FEM",
                            "engine_type": "FEM",
                            "status": "COMPLETED",
                            "created_at": "2025-11-17T11:00:00Z"
                        }
                    ]
                }
            }
        }
    }
)
async def list_continuum_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    engine_type: Optional[str] = Query(None, description="Filter by engine type"),
    skip: int = Query(0, ge=0, description="Results to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum results"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[ContinuumJobResponse]:
    """
    Get list of continuum jobs with filtering.
    """
    logger.debug(f"Listing continuum jobs: status={status}, engine_type={engine_type}, limit={limit}")

    # Build query
    query = select(ContinuumSimulationJob)

    # Filter by owner (non-admin users can only see their own jobs)
    if not current_user.is_admin:
        query = query.where(ContinuumSimulationJob.owner_id == current_user.id)

    # Apply filters
    if status:
        query = query.where(ContinuumSimulationJob.status == status.upper())

    if engine_type:
        query = query.where(ContinuumSimulationJob.engine_type == engine_type.upper())

    # Sort by creation time (newest first)
    query = query.order_by(desc(ContinuumSimulationJob.created_at))

    # Apply pagination
    query = query.offset(skip).limit(limit)

    # Load relationships
    query = query.options(selectinload(ContinuumSimulationJob.results))

    # Execute
    result = await db.execute(query)
    jobs = result.scalars().all()

    # Build responses
    responses = []
    for job in jobs:
        duration = None
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()

        responses.append(ContinuumJobResponse(
            id=job.id,
            owner_id=job.owner_id,
            parent_mesoscale_job_id=job.parent_mesoscale_job_id,
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
    response_model=ContinuumResultResponse,
    summary="Get continuum simulation results",
    description="""
    Get results for a completed continuum simulation job.

    Returns:
    - Effective material properties (elastic moduli, thermal conductivity)
    - Mesh information (nodes, elements, mesh quality)
    - Solution fields (displacement, stress, temperature, strain)
    - Output file paths (VTK, ParaView, result databases)

    Only available for jobs with COMPLETED status.
    Failed jobs may have partial results.
    """,
    responses={
        200: {
            "description": "Simulation results",
            "content": {
                "application/json": {
                    "example": {
                        "id": "823e4567-e89b-12d3-a456-426614174000",
                        "job_id": "723e4567-e89b-12d3-a456-426614174000",
                        "effective_properties": {
                            "effective_elastic_modulus": 195e9
                        },
                        "mesh_info": {
                            "num_nodes": 125000,
                            "num_elements": 100000
                        },
                        "created_at": "2025-11-17T11:45:00Z"
                    }
                }
            }
        },
        404: {"description": "Job or results not found"},
        403: {"description": "Not authorized to view results"},
        409: {"description": "Job not completed yet"}
    }
)
async def get_continuum_results(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ContinuumResultResponse:
    """
    Get continuum simulation results for a job.
    """
    logger.debug(f"Fetching results for continuum job: {job_id}")

    # Get job with results
    query = select(ContinuumSimulationJob).where(
        ContinuumSimulationJob.id == job_id
    ).options(selectinload(ContinuumSimulationJob.results))

    result = await db.execute(query)
    job = result.scalar_one_or_none()

    if not job:
        raise NotFoundError("ContinuumSimulationJob", job_id)

    # Check ownership (non-admin users can only view their own job results)
    if not current_user.is_admin and job.owner_id != current_user.id:
        raise AuthorizationError("You can only view results for your own jobs")

    # Check if results exist
    if not job.results:
        # Check if job is completed
        if job.status == MultiscaleJobStatus.COMPLETED:
            raise NotFoundError(
                "ContinuumSimulationResult",
                f"No results found for completed job {job_id}"
            )
        else:
            raise ConflictError(
                message=f"Job not completed yet (status: {job.status.value})",
                details={"job_id": str(job_id), "status": job.status.value}
            )

    return ContinuumResultResponse(
        id=job.results.id,
        job_id=job.results.job_id,
        effective_properties=job.results.effective_properties,
        mesh_info=job.results.mesh_info,
        solution_fields=job.results.solution_fields,
        output_files=job.results.output_files,
        created_at=job.results.created_at
    )

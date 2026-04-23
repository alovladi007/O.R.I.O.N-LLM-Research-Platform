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
from ..models import (
    User,
    Structure,
    WorkflowTemplate,
    SimulationJob,
    SimulationResult,
    JobStatus,
    IllegalJobTransitionError,
)
from ..models.simulation import JobKind
from ..schemas.simulation import (
    SimulationJobCreate,
    SimulationJobDispatch,
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

# Note: app.py includes this router at `prefix="/api/v1/jobs"`, so the
# router must NOT repeat "/jobs" (Session 1.1 caught the same bug on the
# structures router). All paths here are written relative to /api/v1/jobs.
router = APIRouter(
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

    # State machine handles the legality check and raises on illegal.
    # Wrap into a 409 ConflictError; the user saw a terminal job.
    try:
        job.transition_to(
            JobStatus.CANCELLED,
            error_message=f"Cancelled by user {current_user.username}",
            set_finished=True,
        )
    except IllegalJobTransitionError as exc:
        raise ConflictError(
            message=str(exc),
            details={"status": job.status.value, "job_id": str(job_id)},
        ) from exc

    await db.commit()
    await db.refresh(job)

    logger.info(f"Simulation job cancelled: {job_id}")

    # NOTE: actual Celery task termination lands in Session 2.1. Until then
    # the CANCELLED state is authoritative from the API's point of view —
    # any worker already processing the job will see the new status when
    # it next checks in (Phase 2 adds the check-in hook).

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


# ---------------------------------------------------------------------------
# Dispatch path — Session 2.2
# ---------------------------------------------------------------------------


# Task name → Celery task (lazy import; avoids pulling src.worker at module
# load). The mock_static entry is the one Session 2.2 ships; Phase 3+ add
# more.
_DISPATCH_TASKS: dict[str, str] = {
    "mock_static": "orion.mock.static",
    "md_nvt": "orion.md.nvt",                  # Session 4.2
    "md_nve": "orion.md.nve",                  # Session 4.2
    "md_npt": "orion.md.npt",                  # Session 4.2
    "dft_static": "orion.dft.static",          # Session 3.2
    "dft_relax": "orion.dft.relax",            # Session 3.3
    "dft_bands": "orion.dft.bands",            # Session 3.3
    "dft_dos": "orion.dft.dos",                # Session 3.3
    "dft_phonons_gamma": "orion.dft.phonons_gamma",   # Session 3.3
    # Session 5.3 scaffolded tasks — raise PendingAnalyzerError at
    # run time, documented in the session report. Registered so the
    # `dft_to_md_to_continuum` DAG dispatches cleanly (loud failure
    # at execution, not silent at validation).
    "dft_elastic": "orion.dft.elastic",                       # Phase 8
    "md_green_kubo_thermal": "orion.md.green_kubo_thermal",    # Phase 4 follow-up
    "continuum_thermomechanical": "orion.continuum.thermomechanical",  # Session 5.3b
}

# Built-in workflow templates, materialized lazily so the mock dispatch
# path works on a fresh DB without a separate seed step.
_BUILTIN_TEMPLATES: dict[str, dict] = {
    "mock_static": {
        "name": "mock_static_default",
        "display_name": "Mock static (builtin)",
        "description": "Auto-created template for the mock_static dispatch path.",
        "engine": "mock",
        "category": "static",
        "default_parameters": {},
        "default_resources": {"cores": 1, "memory_gb": 1, "walltime_minutes": 5},
        "is_active": True,
        "is_public": True,
    },
    "md_nvt": {
        "name": "md_nvt_default",
        "display_name": "MD NVT (LAMMPS, builtin)",
        "description": "Auto-created template for LAMMPS NVT Langevin runs.",
        "engine": "lammps",
        "category": "md",
        "default_parameters": {
            "ensemble": "nvt_langevin",
            "temperature_k": 300.0,
            "timestep_fs": 1.0,
            "duration_ps": 10.0,
            "thermo_every": 100,
            "dump_every": 500,
        },
        "default_resources": {"cores": 2, "memory_gb": 2, "walltime_minutes": 60},
        "is_active": True,
        "is_public": True,
    },
    "md_nve": {
        "name": "md_nve_default",
        "display_name": "MD NVE (LAMMPS, builtin)",
        "description": "Auto-created template for LAMMPS NVE runs.",
        "engine": "lammps",
        "category": "md",
        "default_parameters": {
            "ensemble": "nve",
            "temperature_k": 300.0,
            "timestep_fs": 1.0,
            "duration_ps": 10.0,
            "thermo_every": 100,
            "dump_every": 500,
        },
        "default_resources": {"cores": 2, "memory_gb": 2, "walltime_minutes": 60},
        "is_active": True,
        "is_public": True,
    },
    "md_npt": {
        "name": "md_npt_default",
        "display_name": "MD NPT (LAMMPS, builtin)",
        "description": "Auto-created template for LAMMPS NPT Parrinello-Rahman runs.",
        "engine": "lammps",
        "category": "md",
        "default_parameters": {
            "ensemble": "npt",
            "temperature_k": 300.0,
            "pressure_bar": 1.0,
            "timestep_fs": 1.0,
            "duration_ps": 20.0,
            "thermo_every": 100,
            "dump_every": 1000,
        },
        "default_resources": {"cores": 2, "memory_gb": 2, "walltime_minutes": 120},
        "is_active": True,
        "is_public": True,
    },
    "dft_static": {
        "name": "dft_static_default",
        "display_name": "DFT static SCF (Quantum Espresso, builtin)",
        "description": "Auto-created template for single-point QE SCF runs.",
        "engine": "qe",
        "category": "static",
        "default_parameters": {
            "calculation": "scf",
            "occupations": "smearing",
            "smearing": "gauss",
            "degauss": 0.01,
        },
        "default_resources": {"cores": 4, "memory_gb": 8, "walltime_minutes": 60},
        "is_active": True,
        "is_public": True,
    },
    "dft_relax": {
        "name": "dft_relax_default",
        "display_name": "DFT variable-cell relax (Quantum Espresso)",
        "description": "Auto-created template for QE vc-relax runs.",
        "engine": "qe",
        "category": "relax",
        "default_parameters": {
            "calculation": "vc-relax",
            "occupations": "smearing",
            "smearing": "gauss",
            "degauss": 0.01,
            "forc_conv_thr": 1.0e-4,
            "press_conv_thr": 0.5,
        },
        "default_resources": {"cores": 4, "memory_gb": 8, "walltime_minutes": 120},
        "is_active": True,
        "is_public": True,
    },
    "dft_bands": {
        "name": "dft_bands_default",
        "display_name": "DFT band structure (Quantum Espresso)",
        "description": "Auto-created template for QE bands runs.",
        "engine": "qe",
        "category": "bands",
        "default_parameters": {
            "calculation": "bands",
            "occupations": "smearing",
            "smearing": "gauss",
            "degauss": 0.01,
        },
        "default_resources": {"cores": 4, "memory_gb": 8, "walltime_minutes": 120},
        "is_active": True,
        "is_public": True,
    },
    "dft_dos": {
        "name": "dft_dos_default",
        "display_name": "DFT DOS (Quantum Espresso + dos.x)",
        "description": "Auto-created template for scf + dos.x runs.",
        "engine": "qe",
        "category": "dos",
        "default_parameters": {
            "calculation": "scf",
            "occupations": "smearing",
            "smearing": "gauss",
            "degauss": 0.01,
            "dos_delta_e_ev": 0.01,
            "dos_emin_ev": -20.0,
            "dos_emax_ev": 20.0,
        },
        "default_resources": {"cores": 4, "memory_gb": 8, "walltime_minutes": 60},
        "is_active": True,
        "is_public": True,
    },
    "dft_phonons_gamma": {
        "name": "dft_phonons_gamma_default",
        "display_name": "DFT phonons at Γ (Quantum Espresso + ph.x)",
        "description": "Auto-created template for scf + ph.x Γ-only runs.",
        "engine": "qe",
        "category": "phonons",
        "default_parameters": {
            "calculation": "scf",
            "occupations": "smearing",
            "smearing": "gauss",
            "degauss": 0.01,
            "tr2_ph": 1.0e-14,
        },
        "default_resources": {"cores": 4, "memory_gb": 8, "walltime_minutes": 180},
        "is_active": True,
        "is_public": True,
    },
    # Session 5.3 deferred builtins. These let the multiscale DAG
    # dispatch cleanly so the 5.3 acceptance test (submit + fail loud
    # at exec) works without hand-wiring templates. Tasks raise
    # PendingAnalyzerError so real runs fail loudly.
    "dft_elastic": {
        "name": "dft_elastic_default",
        "display_name": "DFT elastic tensor (Phase 8, scaffolded)",
        "description": (
            "Scaffolded template for the Phase 8 DFT elastic-tensor "
            "workflow. The Celery task raises PendingAnalyzerError; "
            "the DAG is wired so submissions fail loudly at execution "
            "rather than at validation."
        ),
        "engine": "qe",
        "category": "elastic",
        "default_parameters": {"strain_magnitude": 0.005},
        "default_resources": {"cores": 4, "memory_gb": 8, "walltime_minutes": 180},
        "is_active": True,
        "is_public": True,
    },
    "md_green_kubo_thermal": {
        "name": "md_green_kubo_thermal_default",
        "display_name": "MD Green-Kubo thermal conductivity (Phase 4 follow-up, scaffolded)",
        "description": (
            "Scaffolded template for the MD Green-Kubo κ analyzer. "
            "Deferred alongside VACF→vDOS. Celery task raises "
            "PendingAnalyzerError until the analyzer ships."
        ),
        "engine": "lammps",
        "category": "thermal",
        "default_parameters": {
            "temperature_k": 300.0,
            "duration_ps": 200.0,
        },
        "default_resources": {"cores": 4, "memory_gb": 4, "walltime_minutes": 240},
        "is_active": True,
        "is_public": True,
    },
    "continuum_thermomechanical": {
        "name": "continuum_thermomechanical_default",
        "display_name": "Continuum thermomechanical FEM (scaffolded)",
        "description": (
            "Scaffolded template for the thermoelastic FEM solve that "
            "consumes the upstream DFT C_ij + MD κ. Pending Session "
            "5.3b; Celery task raises PendingAnalyzerError today."
        ),
        "engine": "skfem",
        "category": "continuum",
        "default_parameters": {
            "length_x_m": 1.0e-3,
            "length_y_m": 1.0e-3,
            "length_z_m": 1.0e-3,
        },
        "default_resources": {"cores": 2, "memory_gb": 4, "walltime_minutes": 60},
        "is_active": True,
        "is_public": True,
    },
}


async def _get_or_create_builtin_template(db: AsyncSession, kind: str) -> WorkflowTemplate:
    """Fetch the built-in template for *kind*, creating it on first use.

    The template is keyed by ``name`` (unique) so concurrent submitters
    race gracefully — SQL's unique constraint collapses the duplicates.
    """
    spec = _BUILTIN_TEMPLATES.get(kind)
    if spec is None:
        raise ValidationError(
            f"No built-in template for kind={kind!r}",
            details={"kind": kind, "known_kinds": sorted(_BUILTIN_TEMPLATES)},
        )
    existing = await db.execute(
        select(WorkflowTemplate).where(WorkflowTemplate.name == spec["name"])
    )
    tpl = existing.scalar_one_or_none()
    if tpl is not None:
        return tpl

    tpl = WorkflowTemplate(**spec)
    db.add(tpl)
    try:
        await db.flush()
    except Exception:  # noqa: BLE001 — someone beat us; re-select.
        await db.rollback()
        existing = await db.execute(
            select(WorkflowTemplate).where(WorkflowTemplate.name == spec["name"])
        )
        tpl = existing.scalar_one()
    return tpl


@router.post(
    "/dispatch",
    response_model=SimulationJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Dispatch a job by kind + structure (Session 2.2)",
    description=(
        "Short-form submission path. Body is ``{kind, structure_id}`` — "
        "the router looks up (or creates) a built-in workflow template "
        "for the kind, creates the SimulationJob in QUEUED status, and "
        "enqueues the matching Celery task. Returns the job row."
    ),
)
async def dispatch_simulation_job(
    body: SimulationJobDispatch,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> SimulationJobResponse:
    if not current_user.can_run_simulations():
        raise AuthorizationError("You don't have permission to run simulations")

    # Validate kind against known Celery tasks.
    if body.kind not in _DISPATCH_TASKS:
        raise ValidationError(
            f"Unknown job kind {body.kind!r}",
            details={"kind": body.kind, "known": sorted(_DISPATCH_TASKS)},
        )

    # Structure must exist.
    structure = await db.get(Structure, body.structure_id)
    if structure is None:
        raise NotFoundError("Structure", body.structure_id)

    tpl = await _get_or_create_builtin_template(db, body.kind)

    new_job = SimulationJob(
        owner_id=current_user.id,
        structure_id=body.structure_id,
        workflow_template_id=tpl.id,
        name=body.name or f"{body.kind} · {structure.name or structure.id}",
        status=JobStatus.QUEUED,
        priority=body.priority,
        engine=tpl.engine,
        kind=body.kind,
        parameters={**(tpl.default_parameters or {}), **(body.parameters or {})},
        resources=tpl.default_resources,
        extra_metadata={},
        submitted_at=datetime.utcnow(),
    )
    db.add(new_job)
    if hasattr(tpl, "usage_count") and tpl.usage_count is not None:
        tpl.usage_count = tpl.usage_count + 1
    await db.commit()
    await db.refresh(new_job)

    # Enqueue. Use send_task by name so the API doesn't import src.worker.
    try:
        from src.worker.celery_app import celery_app

        async_result = celery_app.send_task(
            _DISPATCH_TASKS[body.kind],
            args=[str(new_job.id)],
        )
        new_job.celery_task_id = async_result.id
        await db.commit()
        await db.refresh(new_job)
    except Exception as exc:  # noqa: BLE001 — flip back to PENDING so the
        # dispatch failure is visible. The reaper won't pick this up
        # (wrong status), but the user gets a 500.
        logger.exception("dispatch enqueue failed: %s", exc)
        new_job.status = JobStatus.FAILED
        new_job.error_message = f"enqueue_failed: {exc}"
        await db.commit()
        raise

    return SimulationJobResponse.model_validate(new_job)


# ---------------------------------------------------------------------------
# Artifacts endpoint — Session 2.2
# ---------------------------------------------------------------------------


@router.get(
    "/{job_id}/artifacts",
    summary="List artifacts for a job (Session 2.2)",
    description=(
        "Returns the artifact bundle stored on the job's ``extra_metadata`` "
        "plus a presigned GET URL when MinIO is reachable. Jobs that haven't "
        "produced artifacts yet return ``{items: []}``."
    ),
)
async def list_job_artifacts(
    job_id: uuid.UUID,
    expires_seconds: int = Query(3600, ge=60, le=24 * 3600),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> dict:
    job = await db.get(SimulationJob, job_id)
    if job is None:
        raise NotFoundError("SimulationJob", job_id)

    extra = job.extra_metadata or {}
    artifact = extra.get("artifact") if isinstance(extra, dict) else None
    if not artifact:
        return {"job_id": str(job.id), "items": []}

    presigned_url: Optional[str] = None
    error: Optional[str] = None
    try:
        from datetime import timedelta as _td

        from backend.common.jobs import build_minio_client, presign_artifact

        client = build_minio_client()
        presigned_url = presign_artifact(
            client,
            bucket=artifact["bucket"],
            key=artifact["key"],
            expires=_td(seconds=expires_seconds),
        )
    except Exception as exc:  # noqa: BLE001 — MinIO unreachable ≠ error for
        # the metadata view; surface it so the caller can retry later.
        logger.warning("presign failed for %s: %s", job.id, exc)
        error = f"{type(exc).__name__}: {exc}"

    return {
        "job_id": str(job.id),
        "items": [
            {
                "kind": "tarball",
                "bucket": artifact.get("bucket"),
                "key": artifact.get("key"),
                "size_bytes": artifact.get("size_bytes"),
                "uploaded": artifact.get("uploaded"),
                "presigned_url": presigned_url,
                "presigned_expires_seconds": expires_seconds if presigned_url else None,
                "error": error,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Live observability endpoints (Session 1.4 MVP)
# ---------------------------------------------------------------------------


@router.get(
    "/{job_id}/events",
    summary="Stream job state transitions as Server-Sent Events",
    description=(
        "Returns a text/event-stream that emits a JSON event every time the "
        "job row changes status, until the job reaches a terminal state or "
        "the client disconnects. Implemented as polling in Session 1.4 "
        "(2 s cadence). Session 10 replaces the polling loop with a Redis "
        "pub/sub push from Celery task hooks."
    ),
)
async def stream_job_events(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from fastapi.responses import StreamingResponse
    import asyncio
    import json

    job = await db.get(SimulationJob, job_id)
    if not job:
        raise NotFoundError("SimulationJob", job_id)

    async def _poll_and_stream():
        last_status = None
        last_progress = None
        # Initial snapshot so the client doesn't have to wait for a change.
        yield _sse_event(
            "snapshot",
            {
                "job_id": str(job_id),
                "status": job.status.value if hasattr(job.status, "value") else job.status,
                "progress": job.progress,
                "current_step": job.current_step,
            },
        )

        while True:
            # Re-read the row — bypass the Session's identity cache.
            current = await db.get(SimulationJob, job_id)
            if current is None:
                yield _sse_event("error", {"reason": "job_disappeared"})
                return

            status_val = (
                current.status.value if hasattr(current.status, "value") else current.status
            )
            if status_val != last_status or current.progress != last_progress:
                yield _sse_event(
                    "status",
                    {
                        "job_id": str(job_id),
                        "status": status_val,
                        "progress": current.progress,
                        "current_step": current.current_step,
                        "updated_at": (
                            current.updated_at.isoformat() if current.updated_at else None
                        ),
                    },
                )
                last_status = status_val
                last_progress = current.progress

            if current.is_terminal:
                yield _sse_event(
                    "terminal",
                    {
                        "job_id": str(job_id),
                        "final_status": status_val,
                        "finished_at": (
                            current.finished_at.isoformat() if current.finished_at else None
                        ),
                    },
                )
                return

            await asyncio.sleep(2.0)

    return StreamingResponse(
        _poll_and_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_event(event: str, data: dict) -> str:
    """Format one Server-Sent Event payload."""
    import json

    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.get(
    "/{job_id}/logs",
    summary="Tail worker logs for this job",
    description=(
        "Returns the most recent worker log lines for this job as text/plain. "
        "Session 1.4 returns a placeholder derived from `error_message` + "
        "`current_step` because the MinIO-backed log pipeline lands in "
        "Session 2.1."
    ),
)
async def get_job_logs(
    job_id: uuid.UUID,
    tail: int = Query(200, ge=1, le=10_000, description="Number of lines to return."),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from fastapi.responses import PlainTextResponse

    job = await db.get(SimulationJob, job_id)
    if not job:
        raise NotFoundError("SimulationJob", job_id)

    # Placeholder: compose what we know about the job. Session 2.1 replaces
    # this with a MinIO presigned URL or a real tail via the worker's log
    # pipeline.
    lines = [
        f"# ORION job log placeholder (Session 1.4 — real tail lands in Session 2.1)",
        f"job_id={job.id}",
        f"status={job.status.value if hasattr(job.status, 'value') else job.status}",
        f"engine={job.engine}",
        f"kind={getattr(job, 'kind', None)}",
        f"worker_id={job.worker_id}",
        f"worker_hostname={job.worker_hostname}",
        f"current_step={job.current_step}",
        f"progress={job.progress}",
        f"submitted_at={job.submitted_at}",
        f"started_at={job.started_at}",
        f"finished_at={job.finished_at}",
    ]
    if job.error_message:
        lines.append(f"error_message={job.error_message}")
    body = "\n".join(lines[-tail:]) + "\n"
    return PlainTextResponse(body)

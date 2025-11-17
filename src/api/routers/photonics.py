"""
Photonics CRUD router for NANO-OS API.

Session 22: Photonics Vertical
Provides CRUD operations for photonics structures and simulation jobs.
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, PhotonicsStructure, PhotonicsSimulationJob
from ..schemas.photonics import (
    PhotonicsStructureCreate,
    PhotonicsStructureUpdate,
    PhotonicsStructureResponse,
    PhotonicsSimulationJobCreate,
    PhotonicsSimulationJobResponse,
)
from ..auth.security import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/photonics",
    tags=["photonics"],
    dependencies=[Depends(get_current_active_user)],
)


# ==============================================================================
# Photonics Structure Endpoints
# ==============================================================================

@router.post(
    "/structures",
    response_model=PhotonicsStructureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create photonics structure"
)
async def create_photonics_structure(
    structure_data: PhotonicsStructureCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> PhotonicsStructureResponse:
    """Create a new photonics structure."""
    logger.info(f"Creating photonics structure: {structure_data.type}")

    structure = PhotonicsStructure(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        base_structure_id=structure_data.base_structure_id,
        type=structure_data.type,
        geometry_params=structure_data.geometry_params,
        wavelength_range=structure_data.wavelength_range,
        metadata=structure_data.metadata or {},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(structure)
    await db.commit()
    await db.refresh(structure)

    return structure


@router.get(
    "/structures",
    response_model=List[PhotonicsStructureResponse],
    summary="List photonics structures"
)
async def list_photonics_structures(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[PhotonicsStructureResponse]:
    """List photonics structures with optional filtering."""
    query = select(PhotonicsStructure).where(
        PhotonicsStructure.owner_id == current_user.id
    )

    if type:
        query = query.where(PhotonicsStructure.type == type)

    query = query.offset(skip).limit(limit).order_by(PhotonicsStructure.created_at.desc())

    result = await db.execute(query)
    structures = result.scalars().all()

    return structures


@router.get(
    "/structures/{structure_id}",
    response_model=PhotonicsStructureResponse,
    summary="Get photonics structure"
)
async def get_photonics_structure(
    structure_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> PhotonicsStructureResponse:
    """Get a specific photonics structure by ID."""
    result = await db.execute(
        select(PhotonicsStructure).where(
            PhotonicsStructure.id == structure_id,
            PhotonicsStructure.owner_id == current_user.id
        )
    )
    structure = result.scalar_one_or_none()

    if not structure:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photonics structure {structure_id} not found"
        )

    return structure


@router.patch(
    "/structures/{structure_id}",
    response_model=PhotonicsStructureResponse,
    summary="Update photonics structure"
)
async def update_photonics_structure(
    structure_id: uuid.UUID,
    update_data: PhotonicsStructureUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> PhotonicsStructureResponse:
    """Update a photonics structure."""
    result = await db.execute(
        select(PhotonicsStructure).where(
            PhotonicsStructure.id == structure_id,
            PhotonicsStructure.owner_id == current_user.id
        )
    )
    structure = result.scalar_one_or_none()

    if not structure:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photonics structure {structure_id} not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(structure, field, value)

    structure.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(structure)

    return structure


@router.delete(
    "/structures/{structure_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete photonics structure"
)
async def delete_photonics_structure(
    structure_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a photonics structure."""
    result = await db.execute(
        select(PhotonicsStructure).where(
            PhotonicsStructure.id == structure_id,
            PhotonicsStructure.owner_id == current_user.id
        )
    )
    structure = result.scalar_one_or_none()

    if not structure:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photonics structure {structure_id} not found"
        )

    await db.delete(structure)
    await db.commit()


# ==============================================================================
# Photonics Simulation Job Endpoints
# ==============================================================================

@router.post(
    "/simulations",
    response_model=PhotonicsSimulationJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create photonics simulation job"
)
async def create_photonics_simulation_job(
    job_data: PhotonicsSimulationJobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> PhotonicsSimulationJobResponse:
    """Create a new photonics simulation job."""
    logger.info(f"Creating photonics simulation job for structure {job_data.photonics_structure_id}")

    job = PhotonicsSimulationJob(
        id=uuid.uuid4(),
        photonics_structure_id=job_data.photonics_structure_id,
        engine=job_data.engine,
        parameters=job_data.parameters,
        results=None,
        status="CREATED",
        metadata=job_data.metadata or {},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    return job


@router.get(
    "/simulations",
    response_model=List[PhotonicsSimulationJobResponse],
    summary="List photonics simulation jobs"
)
async def list_photonics_simulation_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[PhotonicsSimulationJobResponse]:
    """List photonics simulation jobs."""
    query = select(PhotonicsSimulationJob)

    if status:
        query = query.where(PhotonicsSimulationJob.status == status)

    query = query.offset(skip).limit(limit).order_by(PhotonicsSimulationJob.created_at.desc())

    result = await db.execute(query)
    jobs = result.scalars().all()

    return jobs


@router.get(
    "/simulations/{job_id}",
    response_model=PhotonicsSimulationJobResponse,
    summary="Get photonics simulation job"
)
async def get_photonics_simulation_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> PhotonicsSimulationJobResponse:
    """Get a specific photonics simulation job by ID."""
    result = await db.execute(
        select(PhotonicsSimulationJob).where(PhotonicsSimulationJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photonics simulation job {job_id} not found"
        )

    return job


@router.delete(
    "/simulations/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete photonics simulation job"
)
async def delete_photonics_simulation_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a photonics simulation job."""
    result = await db.execute(
        select(PhotonicsSimulationJob).where(PhotonicsSimulationJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Photonics simulation job {job_id} not found"
        )

    await db.delete(job)
    await db.commit()

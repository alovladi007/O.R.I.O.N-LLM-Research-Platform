"""
Metamaterials CRUD router for NANO-OS API.

Session 25: Metamaterials Vertical
Provides CRUD operations for metamaterial unit cells and simulation jobs.
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, MetamaterialUnitCell, MetamaterialSimulationJob
from ..schemas.metamaterial import (
    MetamaterialUnitCellCreate,
    MetamaterialUnitCellUpdate,
    MetamaterialUnitCellResponse,
    MetamaterialSimulationJobCreate,
    MetamaterialSimulationJobResponse,
)
from ..auth.security import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/metamaterials",
    tags=["metamaterials"],
    dependencies=[Depends(get_current_active_user)],
)


# ==============================================================================
# Metamaterial Unit Cell Endpoints
# ==============================================================================

@router.post(
    "/cells",
    response_model=MetamaterialUnitCellResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create metamaterial unit cell"
)
async def create_metamaterial_unit_cell(
    cell_data: MetamaterialUnitCellCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MetamaterialUnitCellResponse:
    """Create a new metamaterial unit cell."""
    logger.info(f"Creating metamaterial unit cell: {cell_data.name} ({cell_data.type})")

    cell = MetamaterialUnitCell(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        base_structure_id=cell_data.base_structure_id,
        name=cell_data.name,
        type=cell_data.type,
        lattice_vectors=cell_data.lattice_vectors,
        fill_pattern=cell_data.fill_pattern,
        target_frequency_range=cell_data.target_frequency_range,
        metadata=cell_data.metadata or {},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(cell)
    await db.commit()
    await db.refresh(cell)

    return cell


@router.get(
    "/cells",
    response_model=List[MetamaterialUnitCellResponse],
    summary="List metamaterial unit cells"
)
async def list_metamaterial_unit_cells(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[MetamaterialUnitCellResponse]:
    """List metamaterial unit cells with optional filtering."""
    query = select(MetamaterialUnitCell).where(
        MetamaterialUnitCell.owner_id == current_user.id
    )

    if type:
        query = query.where(MetamaterialUnitCell.type == type)

    query = query.offset(skip).limit(limit).order_by(MetamaterialUnitCell.created_at.desc())

    result = await db.execute(query)
    cells = result.scalars().all()

    return cells


@router.get(
    "/cells/{cell_id}",
    response_model=MetamaterialUnitCellResponse,
    summary="Get metamaterial unit cell"
)
async def get_metamaterial_unit_cell(
    cell_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MetamaterialUnitCellResponse:
    """Get a specific metamaterial unit cell by ID."""
    result = await db.execute(
        select(MetamaterialUnitCell).where(
            MetamaterialUnitCell.id == cell_id,
            MetamaterialUnitCell.owner_id == current_user.id
        )
    )
    cell = result.scalar_one_or_none()

    if not cell:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metamaterial unit cell {cell_id} not found"
        )

    return cell


@router.patch(
    "/cells/{cell_id}",
    response_model=MetamaterialUnitCellResponse,
    summary="Update metamaterial unit cell"
)
async def update_metamaterial_unit_cell(
    cell_id: uuid.UUID,
    update_data: MetamaterialUnitCellUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MetamaterialUnitCellResponse:
    """Update a metamaterial unit cell."""
    result = await db.execute(
        select(MetamaterialUnitCell).where(
            MetamaterialUnitCell.id == cell_id,
            MetamaterialUnitCell.owner_id == current_user.id
        )
    )
    cell = result.scalar_one_or_none()

    if not cell:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metamaterial unit cell {cell_id} not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(cell, field, value)

    cell.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(cell)

    return cell


@router.delete(
    "/cells/{cell_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete metamaterial unit cell"
)
async def delete_metamaterial_unit_cell(
    cell_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a metamaterial unit cell."""
    result = await db.execute(
        select(MetamaterialUnitCell).where(
            MetamaterialUnitCell.id == cell_id,
            MetamaterialUnitCell.owner_id == current_user.id
        )
    )
    cell = result.scalar_one_or_none()

    if not cell:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metamaterial unit cell {cell_id} not found"
        )

    await db.delete(cell)
    await db.commit()


# ==============================================================================
# Metamaterial Simulation Job Endpoints
# ==============================================================================

@router.post(
    "/simulations",
    response_model=MetamaterialSimulationJobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create metamaterial simulation job"
)
async def create_metamaterial_simulation_job(
    job_data: MetamaterialSimulationJobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MetamaterialSimulationJobResponse:
    """Create a new metamaterial simulation job."""
    logger.info(f"Creating metamaterial simulation job for unit cell {job_data.unit_cell_id}")

    job = MetamaterialSimulationJob(
        id=uuid.uuid4(),
        unit_cell_id=job_data.unit_cell_id,
        name=job_data.name,
        status="CREATED",
        engine=job_data.engine,
        parameters=job_data.parameters,
        results=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    return job


@router.get(
    "/simulations",
    response_model=List[MetamaterialSimulationJobResponse],
    summary="List metamaterial simulation jobs"
)
async def list_metamaterial_simulation_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[MetamaterialSimulationJobResponse]:
    """List metamaterial simulation jobs."""
    query = select(MetamaterialSimulationJob)

    if status:
        query = query.where(MetamaterialSimulationJob.status == status)

    query = query.offset(skip).limit(limit).order_by(MetamaterialSimulationJob.created_at.desc())

    result = await db.execute(query)
    jobs = result.scalars().all()

    return jobs


@router.get(
    "/simulations/{job_id}",
    response_model=MetamaterialSimulationJobResponse,
    summary="Get metamaterial simulation job"
)
async def get_metamaterial_simulation_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MetamaterialSimulationJobResponse:
    """Get a specific metamaterial simulation job by ID."""
    result = await db.execute(
        select(MetamaterialSimulationJob).where(MetamaterialSimulationJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metamaterial simulation job {job_id} not found"
        )

    return job


@router.delete(
    "/simulations/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete metamaterial simulation job"
)
async def delete_metamaterial_simulation_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a metamaterial simulation job."""
    result = await db.execute(
        select(MetamaterialSimulationJob).where(MetamaterialSimulationJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metamaterial simulation job {job_id} not found"
        )

    await db.delete(job)
    await db.commit()

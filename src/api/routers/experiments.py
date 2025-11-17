"""
Experiments and Instruments CRUD router for NANO-OS API.

Session 21: Lab Integration & Experiment Management
Provides CRUD operations for instruments and experiment runs.
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, Instrument, ExperimentRun
from ..schemas.experiment import (
    InstrumentCreate,
    InstrumentUpdate,
    InstrumentResponse,
    ExperimentCreate,
    ExperimentUpdate,
    ExperimentResponse,
)
from ..auth.security import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/experiments",
    tags=["experiments"],
    dependencies=[Depends(get_current_active_user)],
)


# ==============================================================================
# Instrument Endpoints
# ==============================================================================

@router.post(
    "/instruments",
    response_model=InstrumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create instrument"
)
async def create_instrument(
    instrument_data: InstrumentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> InstrumentResponse:
    """Create a new instrument."""
    logger.info(f"Creating instrument: {instrument_data.name} ({instrument_data.adapter_type})")

    instrument = Instrument(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        name=instrument_data.name,
        description=instrument_data.description,
        adapter_type=instrument_data.adapter_type,
        connection_info=instrument_data.connection_info,
        capabilities=instrument_data.capabilities or [],
        status=instrument_data.status,
        location=instrument_data.location,
        metadata=instrument_data.metadata or {},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        last_used_at=None,
    )

    db.add(instrument)
    await db.commit()
    await db.refresh(instrument)

    return instrument


@router.get(
    "/instruments",
    response_model=List[InstrumentResponse],
    summary="List instruments"
)
async def list_instruments(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    adapter_type: Optional[str] = None,
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[InstrumentResponse]:
    """List instruments with optional filtering."""
    query = select(Instrument).where(
        Instrument.owner_id == current_user.id
    )

    if adapter_type:
        query = query.where(Instrument.adapter_type == adapter_type)
    if status:
        query = query.where(Instrument.status == status)

    query = query.offset(skip).limit(limit).order_by(Instrument.created_at.desc())

    result = await db.execute(query)
    instruments = result.scalars().all()

    return instruments


@router.get(
    "/instruments/{instrument_id}",
    response_model=InstrumentResponse,
    summary="Get instrument"
)
async def get_instrument(
    instrument_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> InstrumentResponse:
    """Get a specific instrument by ID."""
    result = await db.execute(
        select(Instrument).where(
            Instrument.id == instrument_id,
            Instrument.owner_id == current_user.id
        )
    )
    instrument = result.scalar_one_or_none()

    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instrument {instrument_id} not found"
        )

    return instrument


@router.patch(
    "/instruments/{instrument_id}",
    response_model=InstrumentResponse,
    summary="Update instrument"
)
async def update_instrument(
    instrument_id: uuid.UUID,
    update_data: InstrumentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> InstrumentResponse:
    """Update an instrument."""
    result = await db.execute(
        select(Instrument).where(
            Instrument.id == instrument_id,
            Instrument.owner_id == current_user.id
        )
    )
    instrument = result.scalar_one_or_none()

    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instrument {instrument_id} not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(instrument, field, value)

    instrument.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(instrument)

    return instrument


@router.delete(
    "/instruments/{instrument_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete instrument"
)
async def delete_instrument(
    instrument_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete an instrument."""
    result = await db.execute(
        select(Instrument).where(
            Instrument.id == instrument_id,
            Instrument.owner_id == current_user.id
        )
    )
    instrument = result.scalar_one_or_none()

    if not instrument:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instrument {instrument_id} not found"
        )

    await db.delete(instrument)
    await db.commit()


# ==============================================================================
# Experiment Run Endpoints
# ==============================================================================

@router.post(
    "/runs",
    response_model=ExperimentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create experiment run"
)
async def create_experiment_run(
    experiment_data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ExperimentResponse:
    """Create a new experiment run."""
    logger.info(f"Creating experiment run: {experiment_data.name} ({experiment_data.type})")

    experiment = ExperimentRun(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        instrument_id=experiment_data.instrument_id,
        linked_structure_id=experiment_data.linked_structure_id,
        name=experiment_data.name,
        description=experiment_data.description,
        type=experiment_data.type,
        status="PLANNED",
        parameters=experiment_data.parameters,
        results={},
        started_at=None,
        completed_at=None,
        error_message=None,
        external_job_id=None,
        metadata=experiment_data.metadata or {},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)

    return experiment


@router.get(
    "/runs",
    response_model=List[ExperimentResponse],
    summary="List experiment runs"
)
async def list_experiment_runs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    type: Optional[str] = None,
    status: Optional[str] = None,
    instrument_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[ExperimentResponse]:
    """List experiment runs with optional filtering."""
    query = select(ExperimentRun).where(
        ExperimentRun.owner_id == current_user.id
    )

    if type:
        query = query.where(ExperimentRun.type == type)
    if status:
        query = query.where(ExperimentRun.status == status)
    if instrument_id:
        query = query.where(ExperimentRun.instrument_id == instrument_id)

    query = query.offset(skip).limit(limit).order_by(ExperimentRun.created_at.desc())

    result = await db.execute(query)
    experiments = result.scalars().all()

    return experiments


@router.get(
    "/runs/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Get experiment run"
)
async def get_experiment_run(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ExperimentResponse:
    """Get a specific experiment run by ID."""
    result = await db.execute(
        select(ExperimentRun).where(
            ExperimentRun.id == experiment_id,
            ExperimentRun.owner_id == current_user.id
        )
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment run {experiment_id} not found"
        )

    return experiment


@router.patch(
    "/runs/{experiment_id}",
    response_model=ExperimentResponse,
    summary="Update experiment run"
)
async def update_experiment_run(
    experiment_id: uuid.UUID,
    update_data: ExperimentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ExperimentResponse:
    """Update an experiment run."""
    result = await db.execute(
        select(ExperimentRun).where(
            ExperimentRun.id == experiment_id,
            ExperimentRun.owner_id == current_user.id
        )
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment run {experiment_id} not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(experiment, field, value)

    experiment.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(experiment)

    return experiment


@router.delete(
    "/runs/{experiment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete experiment run"
)
async def delete_experiment_run(
    experiment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete an experiment run."""
    result = await db.execute(
        select(ExperimentRun).where(
            ExperimentRun.id == experiment_id,
            ExperimentRun.owner_id == current_user.id
        )
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment run {experiment_id} not found"
        )

    await db.delete(experiment)
    await db.commit()

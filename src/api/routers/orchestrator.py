"""
Orchestrator management API endpoints.

Session 30: Control Plane for Nanomaterials AGI
"""

import logging
from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.api.database import get_db
from src.api.models import (
    OrchestratorState,
    OrchestratorRun,
    OrchestratorMode,
    User,
)
from src.api.schemas.orchestrator import (
    OrchestratorConfigUpdate,
    OrchestratorStateResponse,
    OrchestratorRunResponse,
    OrchestratorRunRequest,
)
from src.api.dependencies.auth import get_current_user
from backend.orchestrator import (
    run_orchestrator_step,
    collect_orchestrator_stats,
    get_default_config,
    get_or_create_orchestrator,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrator", tags=["orchestrator"])


@router.get("/state", response_model=OrchestratorStateResponse)
async def get_orchestrator_state(
    name: str = "default",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get orchestrator state.

    Args:
        name: Orchestrator name
        db: Database session
        current_user: Current authenticated user

    Returns:
        Orchestrator state
    """
    # Get or create orchestrator
    orchestrator = get_or_create_orchestrator(db, name=name)

    return OrchestratorStateResponse.from_orm(orchestrator)


@router.post("/config", response_model=OrchestratorStateResponse)
async def update_orchestrator_config(
    config_update: OrchestratorConfigUpdate,
    name: str = "default",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update orchestrator configuration.

    Args:
        config_update: Configuration updates
        name: Orchestrator name
        db: Database session
        current_user: Current authenticated user

    Returns:
        Updated orchestrator state
    """
    # Get orchestrator
    result = await db.execute(
        select(OrchestratorState).where(OrchestratorState.name == name)
    )
    orchestrator = result.scalar_one_or_none()

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Orchestrator '{name}' not found"
        )

    # Update config
    current_config = orchestrator.config or get_default_config()

    # Apply updates
    update_dict = config_update.dict(exclude_unset=True)
    for key, value in update_dict.items():
        if value is not None:
            current_config[key] = value

    orchestrator.config = current_config

    await db.commit()
    await db.refresh(orchestrator)

    logger.info(f"Updated orchestrator config: {list(update_dict.keys())}")

    return OrchestratorStateResponse.from_orm(orchestrator)


@router.post("/run_once", response_model=OrchestratorRunResponse)
async def run_orchestrator_once(
    request: OrchestratorRunRequest = OrchestratorRunRequest(),
    name: str = "default",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Run orchestrator step once.

    This manually triggers a single orchestrator step.

    Args:
        request: Run request with optional trigger context
        name: Orchestrator name
        db: Database session
        current_user: Current authenticated user

    Returns:
        Orchestrator run record
    """
    # Get orchestrator
    result = await db.execute(
        select(OrchestratorState).where(OrchestratorState.name == name)
    )
    orchestrator = result.scalar_one_or_none()

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Orchestrator '{name}' not found"
        )

    if not orchestrator.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Orchestrator '{name}' is not active"
        )

    # Run orchestrator step
    try:
        run = run_orchestrator_step(
            db=db,
            orchestrator_id=orchestrator.id,
            triggered_by="manual",
            trigger_context=request.trigger_context or {"user_id": str(current_user.id)}
        )

        logger.info(
            f"Orchestrator step completed manually: "
            f"{len(run.actions.get('campaigns_advanced', []))} campaigns advanced"
        )

        return OrchestratorRunResponse.from_orm(run)

    except Exception as e:
        logger.error(f"Orchestrator step failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestrator step failed: {str(e)}"
        )


@router.get("/runs", response_model=List[OrchestratorRunResponse])
async def list_orchestrator_runs(
    name: str = "default",
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List orchestrator runs.

    Args:
        name: Orchestrator name
        limit: Maximum number of runs to return
        offset: Number of runs to skip
        db: Database session
        current_user: Current authenticated user

    Returns:
        List of orchestrator runs
    """
    # Get orchestrator
    result = await db.execute(
        select(OrchestratorState).where(OrchestratorState.name == name)
    )
    orchestrator = result.scalar_one_or_none()

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Orchestrator '{name}' not found"
        )

    # Get runs
    result = await db.execute(
        select(OrchestratorRun)
        .where(OrchestratorRun.orchestrator_id == orchestrator.id)
        .order_by(OrchestratorRun.started_at.desc())
        .limit(limit)
        .offset(offset)
    )
    runs = result.scalars().all()

    return [OrchestratorRunResponse.from_orm(run) for run in runs]


@router.get("/stats")
async def get_orchestrator_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get current system statistics.

    Args:
        db: Database session
        current_user: Current authenticated user

    Returns:
        System statistics
    """
    stats = collect_orchestrator_stats(db)
    return stats


@router.post("/activate", response_model=OrchestratorStateResponse)
async def activate_orchestrator(
    name: str = "default",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Activate orchestrator.

    Args:
        name: Orchestrator name
        db: Database session
        current_user: Current authenticated user

    Returns:
        Updated orchestrator state
    """
    result = await db.execute(
        select(OrchestratorState).where(OrchestratorState.name == name)
    )
    orchestrator = result.scalar_one_or_none()

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Orchestrator '{name}' not found"
        )

    orchestrator.is_active = True
    await db.commit()
    await db.refresh(orchestrator)

    logger.info(f"Activated orchestrator: {name}")

    return OrchestratorStateResponse.from_orm(orchestrator)


@router.post("/deactivate", response_model=OrchestratorStateResponse)
async def deactivate_orchestrator(
    name: str = "default",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Deactivate orchestrator.

    Args:
        name: Orchestrator name
        db: Database session
        current_user: Current authenticated user

    Returns:
        Updated orchestrator state
    """
    result = await db.execute(
        select(OrchestratorState).where(OrchestratorState.name == name)
    )
    orchestrator = result.scalar_one_or_none()

    if not orchestrator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Orchestrator '{name}' not found"
        )

    orchestrator.is_active = False
    await db.commit()
    await db.refresh(orchestrator)

    logger.info(f"Deactivated orchestrator: {name}")

    return OrchestratorStateResponse.from_orm(orchestrator)

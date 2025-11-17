"""
Execution Profiles CRUD router for NANO-OS API.

Session 27: HPC and Cloud Scaling
Provides CRUD operations for execution profiles.
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, ExecutionProfile
from ..schemas.execution import (
    ExecutionProfileCreate,
    ExecutionProfileUpdate,
    ExecutionProfileResponse,
)
from ..auth.security import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/execution",
    tags=["execution"],
    dependencies=[Depends(get_current_active_user)],
)


@router.post(
    "/profiles",
    response_model=ExecutionProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create execution profile"
)
async def create_execution_profile(
    profile_data: ExecutionProfileCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ExecutionProfileResponse:
    """Create a new execution profile."""
    logger.info(f"Creating execution profile: {profile_data.name} ({profile_data.backend_type})")

    profile = ExecutionProfile(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        name=profile_data.name,
        description=profile_data.description,
        backend_type=profile_data.backend_type,
        config=profile_data.config,
        default_resources=profile_data.default_resources,
        is_active=profile_data.is_active,
        is_default=profile_data.is_default,
        metadata=profile_data.metadata or {},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(profile)
    await db.commit()
    await db.refresh(profile)

    return profile


@router.get(
    "/profiles",
    response_model=List[ExecutionProfileResponse],
    summary="List execution profiles"
)
async def list_execution_profiles(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    backend_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[ExecutionProfileResponse]:
    """List execution profiles with optional filtering."""
    query = select(ExecutionProfile).where(
        ExecutionProfile.owner_id == current_user.id
    )

    if backend_type:
        query = query.where(ExecutionProfile.backend_type == backend_type)
    if is_active is not None:
        query = query.where(ExecutionProfile.is_active == is_active)

    query = query.offset(skip).limit(limit).order_by(ExecutionProfile.created_at.desc())

    result = await db.execute(query)
    profiles = result.scalars().all()

    return profiles


@router.get(
    "/profiles/{profile_id}",
    response_model=ExecutionProfileResponse,
    summary="Get execution profile"
)
async def get_execution_profile(
    profile_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ExecutionProfileResponse:
    """Get a specific execution profile by ID."""
    result = await db.execute(
        select(ExecutionProfile).where(
            ExecutionProfile.id == profile_id,
            ExecutionProfile.owner_id == current_user.id
        )
    )
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution profile {profile_id} not found"
        )

    return profile


@router.patch(
    "/profiles/{profile_id}",
    response_model=ExecutionProfileResponse,
    summary="Update execution profile"
)
async def update_execution_profile(
    profile_id: uuid.UUID,
    update_data: ExecutionProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ExecutionProfileResponse:
    """Update an execution profile."""
    result = await db.execute(
        select(ExecutionProfile).where(
            ExecutionProfile.id == profile_id,
            ExecutionProfile.owner_id == current_user.id
        )
    )
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution profile {profile_id} not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(profile, field, value)

    profile.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(profile)

    return profile


@router.delete(
    "/profiles/{profile_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete execution profile"
)
async def delete_execution_profile(
    profile_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete an execution profile."""
    result = await db.execute(
        select(ExecutionProfile).where(
            ExecutionProfile.id == profile_id,
            ExecutionProfile.owner_id == current_user.id
        )
    )
    profile = result.scalar_one_or_none()

    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Execution profile {profile_id} not found"
        )

    await db.delete(profile)
    await db.commit()

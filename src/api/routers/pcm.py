"""
Phase-Change Memory (PCM) CRUD router for NANO-OS API.

Session 26: PCM Vertical
Provides CRUD operations for PCM materials.
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, PCMMaterial
from ..schemas.pcm import (
    PCMMaterialCreate,
    PCMMaterialUpdate,
    PCMMaterialResponse,
)
from ..auth.security import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/pcm",
    tags=["pcm"],
    dependencies=[Depends(get_current_active_user)],
)


@router.post(
    "/materials",
    response_model=PCMMaterialResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create PCM material"
)
async def create_pcm_material(
    material_data: PCMMaterialCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> PCMMaterialResponse:
    """Create a new PCM material."""
    logger.info(f"Creating PCM material: {material_data.name} ({material_data.composition})")

    material = PCMMaterial(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        structure_crystalline_id=material_data.structure_crystalline_id,
        structure_amorphous_id=material_data.structure_amorphous_id,
        name=material_data.name,
        composition=material_data.composition,
        description=material_data.description,
        metadata=material_data.metadata or {},
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(material)
    await db.commit()
    await db.refresh(material)

    return material


@router.get(
    "/materials",
    response_model=List[PCMMaterialResponse],
    summary="List PCM materials"
)
async def list_pcm_materials(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    composition: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[PCMMaterialResponse]:
    """List PCM materials with optional filtering."""
    query = select(PCMMaterial).where(
        PCMMaterial.owner_id == current_user.id
    )

    if composition:
        query = query.where(PCMMaterial.composition == composition)

    query = query.offset(skip).limit(limit).order_by(PCMMaterial.created_at.desc())

    result = await db.execute(query)
    materials = result.scalars().all()

    return materials


@router.get(
    "/materials/{material_id}",
    response_model=PCMMaterialResponse,
    summary="Get PCM material"
)
async def get_pcm_material(
    material_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> PCMMaterialResponse:
    """Get a specific PCM material by ID."""
    result = await db.execute(
        select(PCMMaterial).where(
            PCMMaterial.id == material_id,
            PCMMaterial.owner_id == current_user.id
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PCM material {material_id} not found"
        )

    return material


@router.patch(
    "/materials/{material_id}",
    response_model=PCMMaterialResponse,
    summary="Update PCM material"
)
async def update_pcm_material(
    material_id: uuid.UUID,
    update_data: PCMMaterialUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> PCMMaterialResponse:
    """Update a PCM material."""
    result = await db.execute(
        select(PCMMaterial).where(
            PCMMaterial.id == material_id,
            PCMMaterial.owner_id == current_user.id
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PCM material {material_id} not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(material, field, value)

    material.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(material)

    return material


@router.delete(
    "/materials/{material_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete PCM material"
)
async def delete_pcm_material(
    material_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a PCM material."""
    result = await db.execute(
        select(PCMMaterial).where(
            PCMMaterial.id == material_id,
            PCMMaterial.owner_id == current_user.id
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PCM material {material_id} not found"
        )

    await db.delete(material)
    await db.commit()

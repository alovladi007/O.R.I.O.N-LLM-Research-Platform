"""
Battery Materials CRUD router for NANO-OS API.

Session 23: Battery Materials Vertical
Provides CRUD operations for battery materials.
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, BatteryMaterial
from ..schemas.battery import (
    BatteryMaterialCreate,
    BatteryMaterialUpdate,
    BatteryMaterialResponse,
)
from ..auth.security import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/batteries",
    tags=["batteries"],
    dependencies=[Depends(get_current_active_user)],
)


@router.post(
    "/materials",
    response_model=BatteryMaterialResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create battery material"
)
async def create_battery_material(
    material_data: BatteryMaterialCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> BatteryMaterialResponse:
    """Create a new battery material."""
    logger.info(f"Creating battery material: {material_data.name} ({material_data.role})")

    material = BatteryMaterial(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        structure_id=material_data.structure_id,
        name=material_data.name,
        role=material_data.role,
        redox_species=material_data.redox_species,
        reference_state_info=material_data.reference_state_info,
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
    response_model=List[BatteryMaterialResponse],
    summary="List battery materials"
)
async def list_battery_materials(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    role: Optional[str] = None,
    redox_species: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[BatteryMaterialResponse]:
    """List battery materials with optional filtering."""
    query = select(BatteryMaterial).where(
        BatteryMaterial.owner_id == current_user.id
    )

    if role:
        query = query.where(BatteryMaterial.role == role)
    if redox_species:
        query = query.where(BatteryMaterial.redox_species == redox_species)

    query = query.offset(skip).limit(limit).order_by(BatteryMaterial.created_at.desc())

    result = await db.execute(query)
    materials = result.scalars().all()

    return materials


@router.get(
    "/materials/{material_id}",
    response_model=BatteryMaterialResponse,
    summary="Get battery material"
)
async def get_battery_material(
    material_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> BatteryMaterialResponse:
    """Get a specific battery material by ID."""
    result = await db.execute(
        select(BatteryMaterial).where(
            BatteryMaterial.id == material_id,
            BatteryMaterial.owner_id == current_user.id
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Battery material {material_id} not found"
        )

    return material


@router.patch(
    "/materials/{material_id}",
    response_model=BatteryMaterialResponse,
    summary="Update battery material"
)
async def update_battery_material(
    material_id: uuid.UUID,
    update_data: BatteryMaterialUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> BatteryMaterialResponse:
    """Update a battery material."""
    result = await db.execute(
        select(BatteryMaterial).where(
            BatteryMaterial.id == material_id,
            BatteryMaterial.owner_id == current_user.id
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Battery material {material_id} not found"
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
    summary="Delete battery material"
)
async def delete_battery_material(
    material_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a battery material."""
    result = await db.execute(
        select(BatteryMaterial).where(
            BatteryMaterial.id == material_id,
            BatteryMaterial.owner_id == current_user.id
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Battery material {material_id} not found"
        )

    await db.delete(material)
    await db.commit()

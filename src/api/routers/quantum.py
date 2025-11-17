"""
Quantum Materials CRUD router for NANO-OS API.

Session 24: Quantum Materials Vertical
Provides CRUD operations for quantum materials and Wannier setups.
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, QuantumMaterial, WannierSetup
from ..schemas.quantum import (
    QuantumMaterialCreate,
    QuantumMaterialUpdate,
    QuantumMaterialResponse,
    WannierSetupCreate,
    WannierSetupUpdate,
    WannierSetupResponse,
)
from ..auth.security import get_current_active_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/quantum",
    tags=["quantum"],
    dependencies=[Depends(get_current_active_user)],
)


# ==============================================================================
# Quantum Material Endpoints
# ==============================================================================

@router.post(
    "/materials",
    response_model=QuantumMaterialResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create quantum material"
)
async def create_quantum_material(
    material_data: QuantumMaterialCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> QuantumMaterialResponse:
    """Create a new quantum material."""
    logger.info(f"Creating quantum material: {material_data.name} ({material_data.focus})")

    material = QuantumMaterial(
        id=uuid.uuid4(),
        owner_id=current_user.id,
        structure_id=material_data.structure_id,
        name=material_data.name,
        focus=material_data.focus,
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
    response_model=List[QuantumMaterialResponse],
    summary="List quantum materials"
)
async def list_quantum_materials(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    focus: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[QuantumMaterialResponse]:
    """List quantum materials with optional filtering."""
    query = select(QuantumMaterial).where(
        QuantumMaterial.owner_id == current_user.id
    )

    if focus:
        query = query.where(QuantumMaterial.focus == focus)

    query = query.offset(skip).limit(limit).order_by(QuantumMaterial.created_at.desc())

    result = await db.execute(query)
    materials = result.scalars().all()

    return materials


@router.get(
    "/materials/{material_id}",
    response_model=QuantumMaterialResponse,
    summary="Get quantum material"
)
async def get_quantum_material(
    material_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> QuantumMaterialResponse:
    """Get a specific quantum material by ID."""
    result = await db.execute(
        select(QuantumMaterial).where(
            QuantumMaterial.id == material_id,
            QuantumMaterial.owner_id == current_user.id
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quantum material {material_id} not found"
        )

    return material


@router.patch(
    "/materials/{material_id}",
    response_model=QuantumMaterialResponse,
    summary="Update quantum material"
)
async def update_quantum_material(
    material_id: uuid.UUID,
    update_data: QuantumMaterialUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> QuantumMaterialResponse:
    """Update a quantum material."""
    result = await db.execute(
        select(QuantumMaterial).where(
            QuantumMaterial.id == material_id,
            QuantumMaterial.owner_id == current_user.id
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quantum material {material_id} not found"
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
    summary="Delete quantum material"
)
async def delete_quantum_material(
    material_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a quantum material."""
    result = await db.execute(
        select(QuantumMaterial).where(
            QuantumMaterial.id == material_id,
            QuantumMaterial.owner_id == current_user.id
        )
    )
    material = result.scalar_one_or_none()

    if not material:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quantum material {material_id} not found"
        )

    await db.delete(material)
    await db.commit()


# ==============================================================================
# Wannier Setup Endpoints
# ==============================================================================

@router.post(
    "/wannier",
    response_model=WannierSetupResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Wannier setup"
)
async def create_wannier_setup(
    setup_data: WannierSetupCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> WannierSetupResponse:
    """Create a new Wannier setup."""
    logger.info(f"Creating Wannier setup: {setup_data.name}")

    setup = WannierSetup(
        id=uuid.uuid4(),
        structure_id=setup_data.structure_id,
        name=setup_data.name,
        target_bands_range=setup_data.target_bands_range,
        initial_projections=setup_data.initial_projections,
        metadata=setup_data.metadata or {},
        created_at=datetime.utcnow(),
    )

    db.add(setup)
    await db.commit()
    await db.refresh(setup)

    return setup


@router.get(
    "/wannier",
    response_model=List[WannierSetupResponse],
    summary="List Wannier setups"
)
async def list_wannier_setups(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    structure_id: Optional[uuid.UUID] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> List[WannierSetupResponse]:
    """List Wannier setups with optional filtering."""
    query = select(WannierSetup)

    if structure_id:
        query = query.where(WannierSetup.structure_id == structure_id)

    query = query.offset(skip).limit(limit).order_by(WannierSetup.created_at.desc())

    result = await db.execute(query)
    setups = result.scalars().all()

    return setups


@router.get(
    "/wannier/{setup_id}",
    response_model=WannierSetupResponse,
    summary="Get Wannier setup"
)
async def get_wannier_setup(
    setup_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> WannierSetupResponse:
    """Get a specific Wannier setup by ID."""
    result = await db.execute(
        select(WannierSetup).where(WannierSetup.id == setup_id)
    )
    setup = result.scalar_one_or_none()

    if not setup:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Wannier setup {setup_id} not found"
        )

    return setup


@router.patch(
    "/wannier/{setup_id}",
    response_model=WannierSetupResponse,
    summary="Update Wannier setup"
)
async def update_wannier_setup(
    setup_id: uuid.UUID,
    update_data: WannierSetupUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> WannierSetupResponse:
    """Update a Wannier setup."""
    result = await db.execute(
        select(WannierSetup).where(WannierSetup.id == setup_id)
    )
    setup = result.scalar_one_or_none()

    if not setup:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Wannier setup {setup_id} not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(setup, field, value)

    await db.commit()
    await db.refresh(setup)

    return setup


@router.delete(
    "/wannier/{setup_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Wannier setup"
)
async def delete_wannier_setup(
    setup_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a Wannier setup."""
    result = await db.execute(
        select(WannierSetup).where(WannierSetup.id == setup_id)
    )
    setup = result.scalar_one_or_none()

    if not setup:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Wannier setup {setup_id} not found"
        )

    await db.delete(setup)
    await db.commit()

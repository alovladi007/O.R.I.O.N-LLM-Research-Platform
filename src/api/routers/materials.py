"""
Materials CRUD router for NANO-OS API.

Provides:
- Create, read, update, delete (CRUD) operations for materials
- List materials with pagination and filtering
- Search by name, formula, tags
- Material metadata management
"""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_
from sqlalchemy.orm import selectinload
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, Material
from ..schemas.material import (
    MaterialCreate,
    MaterialUpdate,
    MaterialResponse,
    MaterialListResponse
)
from ..auth.security import get_current_active_user
from ..exceptions import NotFoundError, ConflictError, ValidationError
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/materials",
    tags=["materials"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Material not found"}
    }
)


@router.post(
    "",
    response_model=MaterialResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create material",
    description="""
    Create a new material entry.

    A material represents a chemical compound that can have multiple
    atomic structures (e.g., MoS2 can have 1H, 1T, 2H polymorphs).

    Required fields:
    - name: Human-readable material name
    - formula: Chemical formula

    Optional fields:
    - description: Detailed description
    - tags: List of tags for categorization
    - composition: Element composition dict (e.g., {"Mo": 1, "S": 2})
    - source: Data source (e.g., "Materials Project", "User Upload")
    - external_id: ID in external database
    """,
    responses={
        201: {
            "description": "Material created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "name": "Molybdenum Disulfide",
                        "formula": "MoS2",
                        "description": "2D semiconductor",
                        "tags": ["2D material", "semiconductor"],
                        "composition": {"Mo": 1, "S": 2},
                        "structure_count": 0,
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                }
            }
        }
    }
)
async def create_material(
    material_data: MaterialCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MaterialResponse:
    """
    Create a new material.
    """
    logger.info(f"Creating material: {material_data.name} ({material_data.formula})")

    # Check for permission (only researchers and admins can create)
    if not current_user.can_create_materials():
        from ..exceptions import AuthorizationError
        raise AuthorizationError("You don't have permission to create materials")

    # Create material
    new_material = Material(
        name=material_data.name,
        formula=material_data.formula,
        description=material_data.description,
        tags=material_data.tags or [],
        composition=material_data.composition,
        metadata=material_data.metadata or {},
        source=material_data.source,
        external_id=material_data.external_id
    )

    db.add(new_material)
    await db.commit()
    await db.refresh(new_material)

    logger.info(f"Material created: {new_material.id}")

    return MaterialResponse.model_validate(new_material)


@router.get(
    "",
    response_model=MaterialListResponse,
    summary="List materials",
    description="""
    Get paginated list of materials with optional filtering.

    Filtering options:
    - name: Search in material name (case-insensitive)
    - formula: Filter by exact chemical formula
    - tags: Filter by tags (comma-separated, matches any)
    - source: Filter by data source

    Pagination:
    - page: Page number (1-based)
    - page_size: Number of items per page (default 50, max 100)

    Results are sorted by creation date (newest first).
    """,
    responses={
        200: {
            "description": "List of materials",
            "content": {
                "application/json": {
                    "example": {
                        "items": [
                            {
                                "id": "123e4567-e89b-12d3-a456-426614174000",
                                "name": "Molybdenum Disulfide",
                                "formula": "MoS2",
                                "tags": ["2D material"],
                                "structure_count": 3,
                                "created_at": "2024-01-15T10:30:00Z"
                            }
                        ],
                        "total": 42,
                        "page": 1,
                        "page_size": 50,
                        "pages": 1
                    }
                }
            }
        }
    }
)
async def list_materials(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    name: Optional[str] = Query(None, description="Filter by name (partial match)"),
    formula: Optional[str] = Query(None, description="Filter by formula (exact match)"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    source: Optional[str] = Query(None, description="Filter by source"),
    db: AsyncSession = Depends(get_db)
) -> MaterialListResponse:
    """
    Get paginated list of materials with filtering.
    """
    logger.debug(f"Listing materials: page={page}, page_size={page_size}")

    # Build query with filters
    query = select(Material).where(Material.deleted_at.is_(None))

    # Apply filters
    if name:
        query = query.where(Material.name.ilike(f"%{name}%"))

    if formula:
        query = query.where(Material.formula == formula)

    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        # Match any tag in the list
        query = query.where(Material.tags.overlap(tag_list))

    if source:
        query = query.where(Material.source == source)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    result = await db.execute(count_query)
    total = result.scalar_one()

    # Apply pagination and sorting
    query = query.order_by(Material.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)

    # Load relationships
    query = query.options(selectinload(Material.structures))

    # Execute query
    result = await db.execute(query)
    materials = result.scalars().all()

    # Calculate pagination info
    pages = (total + page_size - 1) // page_size if total > 0 else 0

    return MaterialListResponse(
        items=[MaterialResponse.model_validate(m) for m in materials],
        total=total,
        page=page,
        page_size=page_size,
        pages=pages
    )


@router.get(
    "/{material_id}",
    response_model=MaterialResponse,
    summary="Get material by ID",
    description="""
    Get detailed information about a specific material.

    Includes:
    - All material metadata
    - Structure count
    - Creation and update timestamps
    """,
    responses={
        200: {
            "description": "Material details",
            "content": {
                "application/json": {
                    "example": {
                        "id": "123e4567-e89b-12d3-a456-426614174000",
                        "name": "Molybdenum Disulfide",
                        "formula": "MoS2",
                        "description": "2D semiconductor",
                        "tags": ["2D material", "semiconductor"],
                        "composition": {"Mo": 1, "S": 2},
                        "structure_count": 3,
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        404: {"description": "Material not found"}
    }
)
async def get_material(
    material_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> MaterialResponse:
    """
    Get material by ID.
    """
    logger.debug(f"Fetching material: {material_id}")

    # Load material with structures
    query = select(Material).where(
        and_(
            Material.id == material_id,
            Material.deleted_at.is_(None)
        )
    ).options(selectinload(Material.structures))

    result = await db.execute(query)
    material = result.scalar_one_or_none()

    if not material:
        raise NotFoundError("Material", material_id)

    return MaterialResponse.model_validate(material)


@router.put(
    "/{material_id}",
    response_model=MaterialResponse,
    summary="Update material",
    description="""
    Update an existing material.

    All fields are optional. Only provided fields will be updated.

    Note: Formula changes should be done carefully as they may
    invalidate associated structures.
    """,
    responses={
        200: {
            "description": "Material updated successfully"
        },
        404: {"description": "Material not found"}
    }
)
async def update_material(
    material_id: uuid.UUID,
    material_data: MaterialUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> MaterialResponse:
    """
    Update material by ID.
    """
    logger.info(f"Updating material: {material_id}")

    # Check permission
    if not current_user.can_create_materials():
        from ..exceptions import AuthorizationError
        raise AuthorizationError("You don't have permission to update materials")

    # Get material
    material = await db.get(Material, material_id)
    if not material or material.deleted_at:
        raise NotFoundError("Material", material_id)

    # Update fields (only if provided)
    update_data = material_data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(material, field, value)

    material.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(material)

    logger.info(f"Material updated: {material_id}")

    return MaterialResponse.model_validate(material)


@router.delete(
    "/{material_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete material",
    description="""
    Soft delete a material.

    This performs a soft delete (sets deleted_at timestamp).
    The material and its structures are hidden from queries but
    preserved in the database for audit purposes.

    To permanently delete, database maintenance is required.
    """,
    responses={
        204: {"description": "Material deleted successfully"},
        404: {"description": "Material not found"}
    }
)
async def delete_material(
    material_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> None:
    """
    Soft delete material by ID.
    """
    logger.info(f"Deleting material: {material_id}")

    # Check permission (only admins can delete)
    if not current_user.is_admin:
        from ..exceptions import AuthorizationError
        raise AuthorizationError("Only admins can delete materials")

    # Get material
    material = await db.get(Material, material_id)
    if not material or material.deleted_at:
        raise NotFoundError("Material", material_id)

    # Soft delete
    material.deleted_at = datetime.utcnow()

    await db.commit()

    logger.info(f"Material deleted: {material_id}")

    return None

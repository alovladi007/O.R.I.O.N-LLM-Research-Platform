"""
Workflow templates CRUD router for NANO-OS API.

Provides:
- Create, read, update, delete (CRUD) operations for workflow templates
- List templates with filtering
- Template management (admin only for creation/deletion)
"""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, WorkflowTemplate
from ..schemas.workflow import (
    WorkflowTemplateCreate,
    WorkflowTemplateUpdate,
    WorkflowTemplateResponse
)
from ..auth.security import get_current_active_user, get_current_superuser
from ..exceptions import NotFoundError, ConflictError, ValidationError, AuthorizationError
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workflows",
    tags=["workflows"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Workflow template not found"}
    }
)


@router.post(
    "",
    response_model=WorkflowTemplateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create workflow template",
    description="""
    Create a new workflow template (admin only).

    A workflow template defines:
    - Simulation engine (QE, VASP, LAMMPS, MOCK)
    - Default parameters
    - Resource requirements
    - Category (DFT, MD, Phonon, etc.)

    Templates are used to create simulation jobs with consistent settings.

    **Admin only**: Only administrators can create workflow templates.
    """,
    responses={
        201: {
            "description": "Workflow template created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "323e4567-e89b-12d3-a456-426614174000",
                        "name": "DFT_relaxation_QE",
                        "display_name": "DFT Relaxation (Quantum ESPRESSO)",
                        "engine": "QE",
                        "category": "DFT",
                        "default_parameters": {
                            "ecutwfc": 50,
                            "k_points": [4, 4, 4]
                        },
                        "is_active": True,
                        "is_public": True,
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        403: {"description": "Not authorized (admin only)"},
        409: {"description": "Template name already exists"}
    },
    dependencies=[Depends(get_current_superuser)]
)
async def create_workflow_template(
    template_data: WorkflowTemplateCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser)
) -> WorkflowTemplateResponse:
    """
    Create a new workflow template (admin only).
    """
    logger.info(f"Creating workflow template: {template_data.name}")

    # Check if name already exists
    result = await db.execute(
        select(WorkflowTemplate).where(
            and_(
                WorkflowTemplate.name == template_data.name,
                WorkflowTemplate.deleted_at.is_(None)
            )
        )
    )
    if result.scalar_one_or_none():
        raise ConflictError(
            message=f"Workflow template '{template_data.name}' already exists",
            details={"field": "name", "value": template_data.name}
        )

    # Create template
    new_template = WorkflowTemplate(
        name=template_data.name,
        display_name=template_data.display_name,
        description=template_data.description,
        engine=template_data.engine,
        engine_version=template_data.engine_version,
        category=template_data.category,
        default_parameters=template_data.default_parameters or {},
        default_resources=template_data.default_resources,
        is_active=True,
        is_public=template_data.is_public,
        documentation_url=template_data.documentation_url,
        usage_count=0
    )

    db.add(new_template)
    await db.commit()
    await db.refresh(new_template)

    logger.info(f"Workflow template created: {new_template.id}")

    return WorkflowTemplateResponse.model_validate(new_template)


@router.get(
    "",
    response_model=List[WorkflowTemplateResponse],
    summary="List workflow templates",
    description="""
    Get list of available workflow templates.

    Filtering options:
    - engine: Filter by simulation engine (QE, VASP, LAMMPS, MOCK)
    - category: Filter by category (DFT, MD, Phonon, etc.)
    - is_active: Show only active templates (default: true)
    - is_public: Show only public templates (default: true)

    Results are sorted by usage count (most used first), then by name.
    """,
    responses={
        200: {
            "description": "List of workflow templates",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "323e4567-e89b-12d3-a456-426614174000",
                            "name": "DFT_relaxation_QE",
                            "display_name": "DFT Relaxation (Quantum ESPRESSO)",
                            "engine": "QE",
                            "category": "DFT",
                            "is_active": True,
                            "is_public": True,
                            "usage_count": 42
                        }
                    ]
                }
            }
        }
    }
)
async def list_workflow_templates(
    engine: Optional[str] = Query(None, description="Filter by engine"),
    category: Optional[str] = Query(None, description="Filter by category"),
    is_active: Optional[bool] = Query(True, description="Filter by active status"),
    is_public: Optional[bool] = Query(True, description="Filter by public status"),
    db: AsyncSession = Depends(get_db)
) -> List[WorkflowTemplateResponse]:
    """
    Get list of workflow templates.
    """
    logger.debug(f"Listing workflow templates: engine={engine}, category={category}")

    # Build query
    query = select(WorkflowTemplate).where(
        WorkflowTemplate.deleted_at.is_(None)
    )

    # Apply filters
    if engine:
        query = query.where(WorkflowTemplate.engine == engine)

    if category:
        query = query.where(WorkflowTemplate.category == category)

    if is_active is not None:
        query = query.where(WorkflowTemplate.is_active == is_active)

    if is_public is not None:
        query = query.where(WorkflowTemplate.is_public == is_public)

    # Sort by usage count (most used first), then by name
    query = query.order_by(
        WorkflowTemplate.usage_count.desc(),
        WorkflowTemplate.name
    )

    # Execute
    result = await db.execute(query)
    templates = result.scalars().all()

    return [WorkflowTemplateResponse.model_validate(t) for t in templates]


@router.get(
    "/{template_id}",
    response_model=WorkflowTemplateResponse,
    summary="Get workflow template by ID",
    description="""
    Get detailed information about a specific workflow template.

    Includes:
    - All template configuration
    - Default parameters
    - Resource requirements
    - Usage statistics
    """,
    responses={
        200: {"description": "Workflow template details"},
        404: {"description": "Template not found"}
    }
)
async def get_workflow_template(
    template_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> WorkflowTemplateResponse:
    """
    Get workflow template by ID.
    """
    logger.debug(f"Fetching workflow template: {template_id}")

    # Get template
    result = await db.execute(
        select(WorkflowTemplate).where(
            and_(
                WorkflowTemplate.id == template_id,
                WorkflowTemplate.deleted_at.is_(None)
            )
        )
    )
    template = result.scalar_one_or_none()

    if not template:
        raise NotFoundError("WorkflowTemplate", template_id)

    return WorkflowTemplateResponse.model_validate(template)


@router.put(
    "/{template_id}",
    response_model=WorkflowTemplateResponse,
    summary="Update workflow template",
    description="""
    Update an existing workflow template (admin only).

    All fields are optional. Only provided fields will be updated.

    Note: Changing template parameters affects only future jobs,
    not existing jobs created from this template.

    **Admin only**: Only administrators can update templates.
    """,
    responses={
        200: {"description": "Template updated successfully"},
        403: {"description": "Not authorized (admin only)"},
        404: {"description": "Template not found"}
    },
    dependencies=[Depends(get_current_superuser)]
)
async def update_workflow_template(
    template_id: uuid.UUID,
    template_data: WorkflowTemplateUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser)
) -> WorkflowTemplateResponse:
    """
    Update workflow template (admin only).
    """
    logger.info(f"Updating workflow template: {template_id}")

    # Get template
    template = await db.get(WorkflowTemplate, template_id)
    if not template or template.deleted_at:
        raise NotFoundError("WorkflowTemplate", template_id)

    # Update fields
    update_data = template_data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(template, field, value)

    template.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(template)

    logger.info(f"Workflow template updated: {template_id}")

    return WorkflowTemplateResponse.model_validate(template)


@router.delete(
    "/{template_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete workflow template",
    description="""
    Soft delete a workflow template (admin only).

    This performs a soft delete (sets deleted_at timestamp).
    The template is hidden from queries but preserved for
    jobs that already use it.

    **Admin only**: Only administrators can delete templates.
    """,
    responses={
        204: {"description": "Template deleted successfully"},
        403: {"description": "Not authorized (admin only)"},
        404: {"description": "Template not found"}
    },
    dependencies=[Depends(get_current_superuser)]
)
async def delete_workflow_template(
    template_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_superuser)
) -> None:
    """
    Soft delete workflow template (admin only).
    """
    logger.info(f"Deleting workflow template: {template_id}")

    # Get template
    template = await db.get(WorkflowTemplate, template_id)
    if not template or template.deleted_at:
        raise NotFoundError("WorkflowTemplate", template_id)

    # Soft delete
    template.deleted_at = datetime.utcnow()

    await db.commit()

    logger.info(f"Workflow template deleted: {template_id}")

    return None


@router.get(
    "/engines/list",
    response_model=List[str],
    summary="List available engines",
    description="""
    Get list of all simulation engines available in templates.

    Returns unique list of engine names (e.g., QE, VASP, LAMMPS, MOCK).
    """,
    responses={
        200: {
            "description": "List of available engines",
            "content": {
                "application/json": {
                    "example": ["QE", "VASP", "LAMMPS", "MOCK"]
                }
            }
        }
    }
)
async def list_engines(
    db: AsyncSession = Depends(get_db)
) -> List[str]:
    """
    Get list of available simulation engines.
    """
    # Get distinct engines from active templates
    result = await db.execute(
        select(WorkflowTemplate.engine).distinct().where(
            and_(
                WorkflowTemplate.is_active == True,
                WorkflowTemplate.deleted_at.is_(None)
            )
        )
    )

    engines = [row[0] for row in result.all()]
    return sorted(engines)


@router.get(
    "/categories/list",
    response_model=List[str],
    summary="List available categories",
    description="""
    Get list of all workflow categories available in templates.

    Returns unique list of category names (e.g., DFT, MD, Phonon).
    """,
    responses={
        200: {
            "description": "List of available categories",
            "content": {
                "application/json": {
                    "example": ["DFT", "MD", "Phonon", "Band Structure"]
                }
            }
        }
    }
)
async def list_categories(
    db: AsyncSession = Depends(get_db)
) -> List[str]:
    """
    Get list of available workflow categories.
    """
    # Get distinct categories from active templates
    result = await db.execute(
        select(WorkflowTemplate.category).distinct().where(
            and_(
                WorkflowTemplate.category.isnot(None),
                WorkflowTemplate.is_active == True,
                WorkflowTemplate.deleted_at.is_(None)
            )
        )
    )

    categories = [row[0] for row in result.all() if row[0]]
    return sorted(categories)

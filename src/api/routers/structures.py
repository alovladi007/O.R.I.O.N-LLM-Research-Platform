"""
Atomic structures CRUD router for NANO-OS API.

Provides:
- Create, read, update, delete (CRUD) operations for structures
- Structure parsing from multiple formats (CIF, POSCAR, XYZ)
- Structure export to different formats
- List structures with filtering
"""

from fastapi import APIRouter, Depends, Query, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import selectinload, joinedload
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, Material, Structure
from ..schemas.structure import (
    StructureCreate,
    StructureUpdate,
    StructureResponse,
    StructureParseRequest,
    StructureParseResponse
)
from ..auth.security import get_current_active_user
from ..exceptions import NotFoundError, ValidationError, ParsingError
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/structures",
    tags=["structures"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Structure not found"}
    }
)


async def parse_structure_file(text: str, format: str) -> dict:
    """
    Parse structure file and extract atomic information.

    This is a stub - actual parsing will be implemented with proper parsers.
    For now, returns basic placeholder data.

    Args:
        text: Raw structure file content
        format: File format (CIF, POSCAR, XYZ, etc.)

    Returns:
        Dictionary with parsed structure data
    """
    # TODO: Import and use actual parsers from backend.common.structures.parsers
    # For now, return mock data to make the API functional

    logger.warning(f"Using stub parser for format: {format}")

    # Extract some basic info (this is a placeholder)
    num_lines = len(text.split('\n'))

    return {
        "formula": "Unknown",  # Would be extracted by real parser
        "num_atoms": 0,
        "dimensionality": 3,
        "lattice": {
            "vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "volume": 1.0
        },
        "atoms": [],
        "lattice_parameters": {
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "volume": 1.0
        }
    }


async def export_structure(structure: Structure, export_format: str) -> str:
    """
    Export structure to specified format.

    This is a stub - actual export will be implemented with proper converters.

    Args:
        structure: Structure model
        export_format: Target format (CIF, POSCAR, XYZ)

    Returns:
        Structure file content as string
    """
    # TODO: Implement actual structure export
    logger.warning(f"Using stub exporter for format: {export_format}")

    # For now, return the raw text if available
    if structure.raw_text:
        return structure.raw_text

    # Otherwise return a placeholder
    return f"# Structure {structure.id}\n# Export format: {export_format}\n# Not yet implemented\n"


@router.post(
    "",
    response_model=StructureResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create structure",
    description="""
    Create a new atomic structure.

    The structure file is parsed automatically to extract:
    - Chemical formula
    - Number of atoms
    - Lattice parameters
    - Atomic positions

    Supported formats:
    - CIF (Crystallographic Information File)
    - POSCAR (VASP format)
    - XYZ (XYZ coordinate file)
    - INTERNAL (JSON format)

    The structure must be associated with an existing material.
    """,
    responses={
        201: {
            "description": "Structure created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "223e4567-e89b-12d3-a456-426614174000",
                        "material_id": "123e4567-e89b-12d3-a456-426614174000",
                        "name": "MoS2 1H polymorph",
                        "format": "CIF",
                        "formula": "MoS2",
                        "num_atoms": 3,
                        "dimensionality": 2,
                        "created_at": "2024-01-15T10:30:00Z"
                    }
                }
            }
        },
        404: {"description": "Material not found"},
        422: {"description": "Structure parsing failed"}
    }
)
async def create_structure(
    structure_data: StructureCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> StructureResponse:
    """
    Create a new structure.
    """
    logger.info(f"Creating structure for material: {structure_data.material_id}")

    # Check permission
    if not current_user.can_create_materials():
        from ..exceptions import AuthorizationError
        raise AuthorizationError("You don't have permission to create structures")

    # Verify material exists
    material = await db.get(Material, structure_data.material_id)
    if not material or material.deleted_at:
        raise NotFoundError("Material", structure_data.material_id)

    # Parse structure file
    try:
        parsed = await parse_structure_file(
            structure_data.raw_text,
            structure_data.format
        )
    except Exception as e:
        logger.error(f"Structure parsing failed: {e}")
        raise ParsingError(
            file_format=structure_data.format,
            message=str(e)
        )

    # Create structure
    new_structure = Structure(
        material_id=structure_data.material_id,
        name=structure_data.name,
        description=structure_data.description,
        format=structure_data.format,
        source=structure_data.source,
        raw_text=structure_data.raw_text,
        # Parsed data
        lattice=parsed.get("lattice"),
        atoms=parsed.get("atoms"),
        formula=parsed.get("formula"),
        num_atoms=parsed.get("num_atoms"),
        dimensionality=parsed.get("dimensionality"),
        # Lattice parameters
        a=parsed.get("lattice_parameters", {}).get("a"),
        b=parsed.get("lattice_parameters", {}).get("b"),
        c=parsed.get("lattice_parameters", {}).get("c"),
        alpha=parsed.get("lattice_parameters", {}).get("alpha"),
        beta=parsed.get("lattice_parameters", {}).get("beta"),
        gamma=parsed.get("lattice_parameters", {}).get("gamma"),
        volume=parsed.get("lattice_parameters", {}).get("volume"),
        metadata=structure_data.metadata or {}
    )

    db.add(new_structure)
    await db.commit()
    await db.refresh(new_structure)

    logger.info(f"Structure created: {new_structure.id}")

    return StructureResponse.model_validate(new_structure)


@router.get(
    "",
    response_model=List[StructureResponse],
    summary="List structures",
    description="""
    Get list of structures with optional filtering.

    Filtering options:
    - material_id: Filter by parent material
    - format: Filter by file format
    - formula: Filter by chemical formula
    - dimensionality: Filter by dimensionality (0=molecule, 2=2D, 3=bulk)

    Results are sorted by creation date (newest first).
    """,
    responses={
        200: {
            "description": "List of structures",
            "content": {
                "application/json": {
                    "example": [
                        {
                            "id": "223e4567-e89b-12d3-a456-426614174000",
                            "material_id": "123e4567-e89b-12d3-a456-426614174000",
                            "name": "MoS2 1H",
                            "format": "CIF",
                            "formula": "MoS2",
                            "num_atoms": 3,
                            "dimensionality": 2
                        }
                    ]
                }
            }
        }
    }
)
async def list_structures(
    material_id: Optional[uuid.UUID] = Query(None, description="Filter by material ID"),
    format: Optional[str] = Query(None, description="Filter by format"),
    formula: Optional[str] = Query(None, description="Filter by formula"),
    dimensionality: Optional[int] = Query(None, ge=0, le=3, description="Filter by dimensionality"),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db)
) -> List[StructureResponse]:
    """
    Get list of structures with filtering.
    """
    logger.debug(f"Listing structures: material_id={material_id}")

    # Build query
    query = select(Structure)

    # Apply filters
    if material_id:
        query = query.where(Structure.material_id == material_id)

    if format:
        query = query.where(Structure.format == format)

    if formula:
        query = query.where(Structure.formula == formula)

    if dimensionality is not None:
        query = query.where(Structure.dimensionality == dimensionality)

    # Sort and limit
    query = query.order_by(Structure.created_at.desc()).limit(limit)

    # Load relationships
    query = query.options(selectinload(Structure.material))

    # Execute
    result = await db.execute(query)
    structures = result.scalars().all()

    return [StructureResponse.model_validate(s) for s in structures]


@router.get(
    "/{structure_id}",
    response_model=StructureResponse,
    summary="Get structure by ID",
    description="""
    Get detailed information about a specific structure.

    Includes:
    - All structural data (lattice, atoms, parameters)
    - Metadata
    - Creation timestamp
    """,
    responses={
        200: {"description": "Structure details"},
        404: {"description": "Structure not found"}
    }
)
async def get_structure(
    structure_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
) -> StructureResponse:
    """
    Get structure by ID.
    """
    logger.debug(f"Fetching structure: {structure_id}")

    # Load with material
    query = select(Structure).where(
        Structure.id == structure_id
    ).options(selectinload(Structure.material))

    result = await db.execute(query)
    structure = result.scalar_one_or_none()

    if not structure:
        raise NotFoundError("Structure", structure_id)

    return StructureResponse.model_validate(structure)


@router.put(
    "/{structure_id}",
    response_model=StructureResponse,
    summary="Update structure",
    description="""
    Update structure metadata.

    Note: This endpoint updates metadata only (name, description).
    The structural data (lattice, atoms) is immutable.
    To modify structure, create a new structure entry.
    """,
    responses={
        200: {"description": "Structure updated successfully"},
        404: {"description": "Structure not found"}
    }
)
async def update_structure(
    structure_id: uuid.UUID,
    structure_data: StructureUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> StructureResponse:
    """
    Update structure metadata.
    """
    logger.info(f"Updating structure: {structure_id}")

    # Check permission
    if not current_user.can_create_materials():
        from ..exceptions import AuthorizationError
        raise AuthorizationError("You don't have permission to update structures")

    # Get structure
    structure = await db.get(Structure, structure_id)
    if not structure:
        raise NotFoundError("Structure", structure_id)

    # Update fields
    update_data = structure_data.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        setattr(structure, field, value)

    structure.updated_at = datetime.utcnow()

    await db.commit()
    await db.refresh(structure)

    logger.info(f"Structure updated: {structure_id}")

    return StructureResponse.model_validate(structure)


@router.delete(
    "/{structure_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete structure",
    description="""
    Delete a structure.

    This permanently deletes the structure from the database.
    Any simulation jobs using this structure will fail.

    Use with caution!
    """,
    responses={
        204: {"description": "Structure deleted successfully"},
        404: {"description": "Structure not found"}
    }
)
async def delete_structure(
    structure_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> None:
    """
    Delete structure by ID.
    """
    logger.info(f"Deleting structure: {structure_id}")

    # Check permission
    if not current_user.is_admin:
        from ..exceptions import AuthorizationError
        raise AuthorizationError("Only admins can delete structures")

    # Get structure
    structure = await db.get(Structure, structure_id)
    if not structure:
        raise NotFoundError("Structure", structure_id)

    # Delete (cascade will handle related jobs)
    await db.delete(structure)
    await db.commit()

    logger.info(f"Structure deleted: {structure_id}")

    return None


@router.post(
    "/parse",
    response_model=StructureParseResponse,
    summary="Parse structure file",
    description="""
    Parse a structure file without saving it to the database.

    Useful for:
    - Validating structure files before upload
    - Extracting structure information
    - Preview before creating a structure entry

    Returns parsed structural data including:
    - Chemical formula
    - Number of atoms
    - Lattice parameters
    - Atomic positions
    """,
    responses={
        200: {
            "description": "Structure parsed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "formula": "MoS2",
                        "num_atoms": 3,
                        "dimensionality": 2,
                        "lattice": {
                            "vectors": [[3.16, 0, 0], [0, 3.16, 0], [0, 0, 12.3]]
                        },
                        "atoms": [
                            {"species": "Mo", "position": [0, 0, 0]},
                            {"species": "S", "position": [0.33, 0.33, 0.1]}
                        ],
                        "lattice_parameters": {
                            "a": 3.16,
                            "b": 3.16,
                            "c": 12.3,
                            "alpha": 90.0,
                            "beta": 90.0,
                            "gamma": 120.0,
                            "volume": 106.5
                        }
                    }
                }
            }
        },
        422: {"description": "Parsing failed"}
    }
)
async def parse_structure(
    parse_request: StructureParseRequest
) -> StructureParseResponse:
    """
    Parse structure file without saving.
    """
    logger.info(f"Parsing structure: format={parse_request.format}")

    try:
        parsed = await parse_structure_file(
            parse_request.text,
            parse_request.format
        )

        return StructureParseResponse(
            formula=parsed["formula"],
            num_atoms=parsed["num_atoms"],
            dimensionality=parsed["dimensionality"],
            lattice=parsed["lattice"],
            atoms=parsed["atoms"],
            lattice_parameters=parsed["lattice_parameters"]
        )

    except Exception as e:
        logger.error(f"Structure parsing failed: {e}")
        raise ParsingError(
            file_format=parse_request.format,
            message=str(e)
        )


@router.get(
    "/{structure_id}/export",
    summary="Export structure",
    description="""
    Export structure to specified format.

    Supported export formats:
    - CIF: Crystallographic Information File
    - POSCAR: VASP input format
    - XYZ: XYZ coordinate file

    Returns the structure file as plain text.
    """,
    responses={
        200: {
            "description": "Structure file content",
            "content": {
                "text/plain": {
                    "example": "data_MoS2\n_cell_length_a 3.16\n..."
                }
            }
        },
        404: {"description": "Structure not found"}
    }
)
async def export_structure_file(
    structure_id: uuid.UUID,
    format: str = Query(..., description="Export format (CIF, POSCAR, XYZ)"),
    db: AsyncSession = Depends(get_db)
) -> Response:
    """
    Export structure to specified format.
    """
    logger.info(f"Exporting structure {structure_id} to {format}")

    # Get structure
    structure = await db.get(Structure, structure_id)
    if not structure:
        raise NotFoundError("Structure", structure_id)

    # Export to format
    try:
        content = await export_structure(structure, format.upper())

        return Response(
            content=content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="structure_{structure_id}.{format.lower()}"'
            }
        )

    except Exception as e:
        logger.error(f"Structure export failed: {e}")
        raise ValidationError(f"Export to {format} failed: {str(e)}")

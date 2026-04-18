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

# Note: app.py includes this router at `prefix="/api/v1/structures"`, so the
# router itself must NOT repeat "/structures". (The pre-refactor code did —
# which routed every endpoint to /api/v1/structures/structures/... — fixed in
# Session 1.1.)
router = APIRouter(
    tags=["structures"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Structure not found"}
    }
)


MIN_ATOM_SEPARATION_A = 0.5  # Å; rejects obviously-degenerate structures.


async def parse_structure_file(
    text: str, format: str, *, symprec: float = 0.01,
) -> dict:
    """
    Parse a structure file with ``backend.common.structures``.

    This is the canonical parse path — all of CRUD, the /parse endpoint,
    and the Session 1.5 seed loader go through here.

    On top of the raw parser output we compute (when possible):

    - **spacegroup symbol + number + crystal system** via
      :class:`pymatgen.symmetry.analyzer.SpacegroupAnalyzer` at the given
      ``symprec`` (Å).
    - **density** = (formula mass × n_formula_units) / volume, reported in
      g/cm³.
    - **structure_hash** (64-hex SHA-256) from
      :func:`backend.common.structures.hashing.structure_hash`. Uses the
      pymatgen-aware path when a :class:`pymatgen.core.Structure` is
      reachable (CIF / POSCAR); falls back to the raw
      species + fractional-coords hash for XYZ molecular inputs.

    The function also enforces one always-on physical sanity check:

    - **No two atoms closer than 0.5 Å.** Anything closer is a parsing
      / upload error rather than a real structure; we raise
      :class:`ParsingError` which maps to HTTP 422.
    """
    from backend.common.structures import StructureFormat, parse_structure
    from backend.common.structures.hashing import structure_hash as _hash_structure

    fmt_upper = format.upper()
    try:
        fmt_enum = StructureFormat(fmt_upper)
    except ValueError as exc:
        raise ParsingError(
            file_format=format,
            message=f"unsupported format; expected one of "
            f"{[f.value for f in StructureFormat]}",
        ) from exc

    try:
        internal = parse_structure(text, fmt_enum)
    except Exception as exc:  # noqa: BLE001 — parser re-raises a broad shape
        raise ParsingError(file_format=format, message=str(exc)) from exc

    # Convert to pymatgen Structure for symmetry / density / hashing where
    # it makes sense. XYZ is treated as molecular; skip symmetry.
    pmg_struct = None
    if fmt_enum in (StructureFormat.CIF, StructureFormat.POSCAR):
        try:
            from pymatgen.core import Lattice, Structure as PmgStructure

            pmg_struct = PmgStructure(
                lattice=Lattice(internal.lattice_vectors),
                species=internal.atomic_species,
                coords=internal.atomic_positions,
                coords_are_cartesian=False,
            )
        except Exception as exc:  # pragma: no cover — pymatgen absent
            logger.warning("pymatgen unavailable for symmetry analysis: %s", exc)

    # --- site distance sanity check ---------------------------------------
    min_sep = _minimum_pair_distance_A(internal, pmg_struct)
    if min_sep is not None and min_sep < MIN_ATOM_SEPARATION_A:
        raise ParsingError(
            file_format=format,
            message=(
                f"Two atoms are {min_sep:.3f} Å apart (below the "
                f"{MIN_ATOM_SEPARATION_A} Å cutoff). This typically indicates "
                "overlapping sites or a parsing error — upload the input "
                "file separately and investigate."
            ),
        )

    # --- symmetry analysis -------------------------------------------------
    space_group_symbol: Optional[str] = None
    space_group_number: Optional[int] = None
    crystal_system: Optional[str] = None
    density_g_cm3: Optional[float] = None

    if pmg_struct is not None:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            sga = SpacegroupAnalyzer(pmg_struct, symprec=symprec, angle_tolerance=5.0)
            space_group_symbol = sga.get_space_group_symbol()
            space_group_number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
        except Exception as exc:  # noqa: BLE001 — spacegroup failures are non-fatal
            logger.info(
                "Spacegroup analysis failed for %s at symprec=%s: %s",
                format, symprec, exc,
            )
        try:
            # pymatgen's density is g/cm³ already
            density_g_cm3 = float(pmg_struct.density)
        except Exception as exc:  # noqa: BLE001
            logger.info("Density computation failed: %s", exc)

    # --- hash --------------------------------------------------------------
    try:
        shash = _hash_structure(
            pmg_structure=pmg_struct,
            lattice=internal.lattice_vectors if pmg_struct is None else None,
            atoms=(
                [
                    {"species": sp, "position": pos}
                    for sp, pos in zip(internal.atomic_species, internal.atomic_positions)
                ]
                if pmg_struct is None
                else None
            ),
        )
    except Exception as exc:  # noqa: BLE001 — hashing should be robust
        logger.error("structure_hash computation failed: %s", exc)
        raise ParsingError(
            file_format=format,
            message=f"unable to compute canonical structure hash: {exc}",
        ) from exc

    # --- assemble payload the router persists / returns --------------------
    return {
        "formula": internal.formula,
        "num_atoms": internal.num_atoms,
        "dimensionality": internal.dimensionality,
        "lattice": {
            "vectors": internal.lattice_vectors,
            "a": internal.a,
            "b": internal.b,
            "c": internal.c,
            "alpha": internal.alpha,
            "beta": internal.beta,
            "gamma": internal.gamma,
            "volume": internal.volume,
        },
        "atoms": [
            {"species": sp, "position": list(pos)}
            for sp, pos in zip(internal.atomic_species, internal.atomic_positions)
        ],
        "lattice_parameters": {
            "a": internal.a,
            "b": internal.b,
            "c": internal.c,
            "alpha": internal.alpha,
            "beta": internal.beta,
            "gamma": internal.gamma,
            "volume": internal.volume,
        },
        "space_group": space_group_symbol,
        "space_group_number": space_group_number,
        "crystal_system": crystal_system,
        "density": density_g_cm3,
        "structure_hash": shash,
    }


def _minimum_pair_distance_A(internal, pmg_struct) -> Optional[float]:
    """
    Return the minimum pairwise atom–atom distance in Å.

    Uses pymatgen's distance_matrix (with PBC) when available; falls back
    to a quadratic scan of Cartesian coords otherwise. Returns ``None``
    when the structure has <2 atoms.
    """
    if pmg_struct is not None:
        try:
            import numpy as np

            dm = pmg_struct.distance_matrix  # Nx N, zero diagonal
            mask = ~np.eye(dm.shape[0], dtype=bool)
            if not mask.any():
                return None
            return float(dm[mask].min())
        except Exception:  # noqa: BLE001
            pass

    # Fallback: direct scan without PBC.
    n = len(internal.atomic_positions)
    if n < 2:
        return None
    # Fractional → cartesian
    import numpy as np

    lat = np.asarray(internal.lattice_vectors, dtype=float)
    frac = np.asarray(internal.atomic_positions, dtype=float)
    cart = frac @ lat
    diffs = cart[:, None, :] - cart[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    # Mask self-pairs
    np.fill_diagonal(dists, np.inf)
    return float(dists.min())


async def export_structure(structure: Structure, export_format: str) -> str:
    """
    Serialize a stored :class:`Structure` to CIF / POSCAR / XYZ.

    Delegates to ``backend.common.structures.{to_cif, to_poscar, to_xyz}``
    after rehydrating an :class:`InternalStructureModel` from the DB row.
    Raises :class:`ParsingError` if the row is missing the fields needed
    for serialization (pre-parse legacy rows without lattice/atoms).
    """
    from backend.common.structures import (
        InternalStructureModel,
        StructureFormat,
        to_cif,
        to_poscar,
        to_xyz,
    )

    fmt_upper = export_format.upper()
    try:
        fmt_enum = StructureFormat(fmt_upper)
    except ValueError as exc:
        raise ParsingError(
            file_format=export_format,
            message="unsupported export format; expected CIF, POSCAR, or XYZ",
        ) from exc

    atoms = structure.atoms or []
    if not atoms or not structure.lattice:
        # Pre-parse legacy rows: no normalized atoms/lattice captured.
        if structure.raw_text and fmt_enum.value == structure.format.value:
            return structure.raw_text
        raise ParsingError(
            file_format=export_format,
            message="structure has no parsed lattice/atoms available for export",
        )

    species = [a.get("species") or a.get("element") or a.get("symbol") for a in atoms]
    positions = [a.get("position") or a.get("coords") for a in atoms]

    lattice_vectors = structure.lattice.get("vectors") if isinstance(structure.lattice, dict) else None
    if lattice_vectors is None:
        # Reconstruct lattice from a,b,c,α,β,γ if vectors aren't stored.
        from pymatgen.core import Lattice

        lattice_vectors = Lattice.from_parameters(
            a=structure.a, b=structure.b, c=structure.c,
            alpha=structure.alpha, beta=structure.beta, gamma=structure.gamma,
        ).matrix.tolist()

    internal = InternalStructureModel(
        lattice_vectors=lattice_vectors,
        atomic_species=species,
        atomic_positions=positions,
        dimensionality=structure.dimensionality or 3,
        formula=structure.formula or "X",
        a=structure.a or 0.0,
        b=structure.b or 0.0,
        c=structure.c or 0.0,
        alpha=structure.alpha or 90.0,
        beta=structure.beta or 90.0,
        gamma=structure.gamma or 90.0,
        volume=structure.volume or 0.0,
        num_atoms=structure.num_atoms or len(species),
        space_group=structure.space_group,
        space_group_number=structure.space_group_number,
    )

    try:
        if fmt_enum == StructureFormat.CIF:
            return to_cif(internal)
        if fmt_enum == StructureFormat.POSCAR:
            return to_poscar(internal)
        if fmt_enum == StructureFormat.XYZ:
            return to_xyz(internal)
    except Exception as exc:  # noqa: BLE001 — re-raise as 422
        raise ParsingError(file_format=export_format, message=str(exc)) from exc

    raise ParsingError(
        file_format=export_format,
        message=f"export helper not registered for {fmt_enum}",
    )


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

    # Create structure — every derived field comes from the canonical parser.
    new_structure = Structure(
        material_id=structure_data.material_id,
        owner_id=current_user.id,
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
        # Lattice parameters (scalar columns for easy filtering)
        a=parsed.get("lattice_parameters", {}).get("a"),
        b=parsed.get("lattice_parameters", {}).get("b"),
        c=parsed.get("lattice_parameters", {}).get("c"),
        alpha=parsed.get("lattice_parameters", {}).get("alpha"),
        beta=parsed.get("lattice_parameters", {}).get("beta"),
        gamma=parsed.get("lattice_parameters", {}).get("gamma"),
        volume=parsed.get("lattice_parameters", {}).get("volume"),
        # Symmetry + derived
        space_group=parsed.get("space_group"),
        space_group_number=parsed.get("space_group_number"),
        crystal_system=parsed.get("crystal_system"),
        density=parsed.get("density"),
        structure_hash=parsed.get("structure_hash"),
        extra_metadata=structure_data.metadata or {},
    )

    db.add(new_structure)
    try:
        await db.commit()
    except Exception as exc:  # likely unique-hash conflict
        await db.rollback()
        from ..exceptions import ConflictError

        raise ConflictError(
            f"Structure with hash {parsed.get('structure_hash')[:12]}… already exists."
        ) from exc
    await db.refresh(new_structure)

    logger.info(
        "Structure created: id=%s hash=%s spacegroup=%s",
        new_structure.id, (new_structure.structure_hash or "")[:12],
        new_structure.space_group_number,
    )

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
    logger.info(
        "Parsing structure: format=%s symprec=%s",
        parse_request.format, parse_request.symprec,
    )

    parsed = await parse_structure_file(
        parse_request.text,
        parse_request.format,
        symprec=parse_request.symprec,
    )

    return StructureParseResponse(
        formula=parsed["formula"],
        num_atoms=parsed["num_atoms"],
        dimensionality=parsed["dimensionality"],
        lattice=parsed["lattice"],
        atoms=parsed["atoms"],
        lattice_parameters=parsed["lattice_parameters"],
        space_group=parsed.get("space_group"),
        space_group_number=parsed.get("space_group_number"),
        structure_hash=parsed["structure_hash"],
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

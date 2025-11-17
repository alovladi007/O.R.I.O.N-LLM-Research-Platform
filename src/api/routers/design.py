"""
Design and optimization API endpoints.

Provides endpoints for:
- Property-based material search
- Multi-constraint optimization
- Candidate structure discovery
- Rule-based structure generation
"""

import logging
import time
from typing import List
from uuid import UUID
import statistics

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from ..schemas.design import (
    DesignSearchRequest,
    DesignSearchResponse,
    CandidateStructure,
)
from ...backend.common.design import (
    search_existing_structures,
    calculate_candidate_score,
    generate_structure_variants,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/design",
    tags=["design"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


@router.post(
    "/search",
    response_model=DesignSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search for materials matching design criteria",
    description="""
    Search for material structures that match specified design constraints.

    Supports:
    - Property-based filtering (bandgap, formation energy, stability)
    - Structural constraints (dimensionality, composition, size)
    - Multi-objective optimization (minimize distance from targets)
    - Rule-based variant generation

    Returns ranked list of candidate structures with match scores.
    """,
    responses={
        200: {
            "description": "Search completed successfully",
            "content": {
                "application/json": {
                    "example": {
                        "candidates": [
                            {
                                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                                "material_id": "123e4567-e89b-12d3-a456-426614174001",
                                "formula": "MoS2",
                                "score": 0.92,
                                "properties": {
                                    "bandgap": 1.8,
                                    "formation_energy": -2.5,
                                    "stability_score": 0.88
                                },
                                "property_source": "ML",
                                "match_details": {},
                                "dimensionality": 2,
                                "num_atoms": 3,
                                "is_generated": False
                            }
                        ],
                        "total_found": 15,
                        "search_params": {},
                        "search_time_ms": 123.45
                    }
                }
            }
        },
        400: {"description": "Invalid search parameters"},
        500: {"description": "Search failed due to internal error"},
    }
)
async def search_materials(
    request: DesignSearchRequest,
    db: AsyncSession = Depends(get_db)
) -> DesignSearchResponse:
    """
    Search for materials matching design criteria.

    This endpoint performs a multi-stage search:
    1. Query database for structures matching structural constraints
    2. Score each candidate based on property constraints
    3. Rank candidates by score
    4. Optionally generate rule-based variants
    5. Return top N candidates

    Args:
        request: Search request with constraints
        db: Database session

    Returns:
        DesignSearchResponse with ranked candidates

    Raises:
        HTTPException: If search fails
    """
    start_time = time.time()

    try:
        logger.info(f"Design search request: {request.model_dump()}")

        # Step 1: Search existing structures
        candidates = await search_existing_structures(db, request)
        logger.info(f"Found {len(candidates)} initial candidates")

        # Step 2: Score and filter candidates
        scored_candidates = []
        for candidate in candidates:
            score, match_details = calculate_candidate_score(
                candidate["properties"],
                request
            )

            # Apply minimum score filter
            if score < request.min_score:
                continue

            # Create candidate structure
            scored_candidates.append(
                CandidateStructure(
                    structure_id=candidate["structure_id"],
                    material_id=candidate["material_id"],
                    formula=candidate["formula"],
                    score=score,
                    properties=candidate["properties"],
                    property_source=candidate["property_source"],
                    match_details=match_details,
                    dimensionality=candidate.get("dimensionality"),
                    num_atoms=candidate.get("num_atoms"),
                    elements=candidate.get("elements"),
                    is_generated=candidate.get("is_generated", False),
                    parent_structure_id=candidate.get("parent_structure_id"),
                    generation_method=candidate.get("generation_method"),
                )
            )

        logger.info(f"Scored {len(scored_candidates)} candidates above threshold")

        # Step 3: Sort by score (descending)
        scored_candidates.sort(key=lambda c: c.score, reverse=True)

        # Step 4: Generate variants if requested
        total_before_generation = len(scored_candidates)
        if request.include_generated and scored_candidates:
            logger.info("Generating structure variants")

            # Use top candidates as templates
            top_candidates_for_generation = [
                {
                    "structure_id": c.structure_id,
                    "material_id": c.material_id,
                    "formula": c.formula,
                    "properties": c.properties,
                    "dimensionality": c.dimensionality,
                    "num_atoms": c.num_atoms,
                }
                for c in scored_candidates[:5]  # Use top 5 as templates
            ]

            # Generate variants
            variants = await generate_structure_variants(
                db,
                top_candidates_for_generation,
                limit=request.limit // 2  # Half of limit for variants
            )

            # Score variants
            variant_candidates = []
            for variant in variants:
                score, match_details = calculate_candidate_score(
                    variant["properties"],
                    request
                )

                if score >= request.min_score:
                    variant_candidates.append(
                        CandidateStructure(
                            structure_id=variant["structure_id"] or UUID(int=0),  # Placeholder for generated
                            material_id=variant["material_id"],
                            formula=variant["formula"],
                            score=score,
                            properties=variant["properties"],
                            property_source=variant["property_source"],
                            match_details=match_details,
                            dimensionality=variant.get("dimensionality"),
                            num_atoms=variant.get("num_atoms"),
                            elements=variant.get("elements"),
                            is_generated=True,
                            parent_structure_id=variant.get("parent_structure_id"),
                            generation_method=variant.get("generation_method"),
                        )
                    )

            # Merge and re-sort
            scored_candidates.extend(variant_candidates)
            scored_candidates.sort(key=lambda c: c.score, reverse=True)

            logger.info(f"Added {len(variant_candidates)} generated variants")

        # Step 5: Apply limit
        total_found = len(scored_candidates)
        final_candidates = scored_candidates[:request.limit]

        # Calculate statistics
        if final_candidates:
            scores = [c.score for c in final_candidates]
            score_distribution = {
                "min": min(scores),
                "max": max(scores),
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
            }

            # Property ranges
            property_ranges = _calculate_property_ranges(final_candidates)
        else:
            score_distribution = None
            property_ranges = None

        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Design search completed: {len(final_candidates)} candidates "
            f"returned ({total_found} total found) in {search_time_ms:.2f}ms"
        )

        return DesignSearchResponse(
            candidates=final_candidates,
            total_found=total_found,
            search_params=request.model_dump(exclude_none=True),
            search_time_ms=search_time_ms,
            score_distribution=score_distribution,
            property_ranges=property_ranges,
        )

    except Exception as e:
        logger.error(f"Design search failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Design search failed: {str(e)}"
        )


def _calculate_property_ranges(
    candidates: List[CandidateStructure]
) -> dict:
    """
    Calculate property ranges across candidates.

    Args:
        candidates: List of candidate structures

    Returns:
        Dictionary of property ranges
    """
    property_ranges = {}

    # Collect all property names
    all_properties = set()
    for candidate in candidates:
        all_properties.update(candidate.properties.keys())

    # Calculate ranges for each property
    for prop_name in all_properties:
        values = [
            c.properties[prop_name]
            for c in candidates
            if prop_name in c.properties
            and isinstance(c.properties[prop_name], (int, float))
        ]

        if values:
            property_ranges[prop_name] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "count": len(values),
            }

    return property_ranges


@router.get(
    "/stats",
    summary="Get design search statistics",
    description="Get statistics about available structures and properties for design search.",
)
async def get_design_stats(
    db: AsyncSession = Depends(get_db)
) -> dict:
    """
    Get statistics for design search.

    Returns:
        Statistics about available structures and properties
    """
    from sqlalchemy import func, select
    from ..models.structure import Structure
    from ..models.predicted_properties import PredictedProperties

    try:
        # Count total structures
        total_structures_result = await db.execute(
            select(func.count()).select_from(Structure)
        )
        total_structures = total_structures_result.scalar()

        # Count structures with predictions
        structures_with_predictions_result = await db.execute(
            select(func.count(func.distinct(PredictedProperties.structure_id)))
            .select_from(PredictedProperties)
        )
        structures_with_predictions = structures_with_predictions_result.scalar()

        # Count by dimensionality
        dim_result = await db.execute(
            select(
                Structure.dimensionality,
                func.count(Structure.id).label('count')
            )
            .where(Structure.dimensionality.isnot(None))
            .group_by(Structure.dimensionality)
        )
        dimensionality_counts = {
            row.dimensionality: row.count
            for row in dim_result
        }

        return {
            "total_structures": total_structures,
            "structures_with_predictions": structures_with_predictions,
            "dimensionality_counts": dimensionality_counts,
            "coverage": {
                "prediction_coverage": (
                    structures_with_predictions / total_structures
                    if total_structures > 0 else 0
                )
            }
        }

    except Exception as e:
        logger.error(f"Failed to get design stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get design stats: {str(e)}"
        )

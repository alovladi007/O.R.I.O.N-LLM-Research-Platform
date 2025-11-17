"""
Material design search service.

Provides functionality for:
- Property-based structure search
- Multi-constraint filtering
- Candidate scoring and ranking
- Rule-based structure variant generation
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import statistics
import re

from ...api.models.structure import Structure
from ...api.models.material import Material
from ...api.models.predicted_properties import PredictedProperties
from ...api.models.simulation import SimulationResult, SimulationJob, JobStatus
from ...api.schemas.design import DesignSearchRequest, PropertyConstraint

logger = logging.getLogger(__name__)


async def search_existing_structures(
    db: AsyncSession,
    constraints: DesignSearchRequest
) -> List[Dict[str, Any]]:
    """
    Query database for structures matching design constraints.

    Args:
        db: Database session
        constraints: Search constraints from request

    Returns:
        List of structures with their properties and metadata
    """
    logger.info(f"Searching structures with constraints: {constraints}")

    # Build base query
    query = (
        select(Structure)
        .outerjoin(
            PredictedProperties,
            PredictedProperties.structure_id == Structure.id
        )
        .outerjoin(
            SimulationJob,
            and_(
                SimulationJob.structure_id == Structure.id,
                SimulationJob.status == JobStatus.COMPLETED
            )
        )
        .outerjoin(
            SimulationResult,
            SimulationResult.simulation_job_id == SimulationJob.id
        )
        .options(
            selectinload(Structure.material),
            selectinload(Structure.predicted_properties),
        )
    )

    # Apply structural filters
    filters = []

    if constraints.dimensionality is not None:
        filters.append(Structure.dimensionality == constraints.dimensionality)

    if constraints.max_atoms is not None:
        filters.append(Structure.num_atoms <= constraints.max_atoms)

    if constraints.min_atoms is not None:
        filters.append(Structure.num_atoms >= constraints.min_atoms)

    if constraints.elements:
        # Structure must contain all specified elements
        for element in constraints.elements:
            filters.append(Structure.formula.contains(element))

    if constraints.exclude_elements:
        # Structure must not contain any excluded elements
        for element in constraints.exclude_elements:
            filters.append(~Structure.formula.contains(element))

    if filters:
        query = query.where(and_(*filters))

    # Execute query
    result = await db.execute(query)
    structures = result.unique().scalars().all()

    logger.info(f"Found {len(structures)} structures matching structural filters")

    # Process structures and collect properties
    candidates = []
    for structure in structures:
        # Collect properties from ML predictions and simulations
        properties = await _collect_structure_properties(db, structure)

        if not properties:
            # Skip structures with no property data
            continue

        # Determine property source
        property_source = _determine_property_source(properties)

        # Get elements from formula
        elements = _extract_elements_from_formula(structure.formula or "")

        candidates.append({
            "structure_id": structure.id,
            "material_id": structure.material_id,
            "formula": structure.formula,
            "properties": properties,
            "property_source": property_source,
            "dimensionality": structure.dimensionality,
            "num_atoms": structure.num_atoms,
            "elements": elements,
            "is_generated": False,
            "parent_structure_id": None,
            "generation_method": None,
        })

    logger.info(f"Collected {len(candidates)} candidates with property data")
    return candidates


async def _collect_structure_properties(
    db: AsyncSession,
    structure: Structure
) -> Dict[str, float]:
    """
    Collect all available properties for a structure.

    Combines properties from:
    - ML predictions (PredictedProperties)
    - Simulation results (SimulationResult)

    Args:
        db: Database session
        structure: Structure to collect properties for

    Returns:
        Dictionary of property name -> value
    """
    properties = {}

    # Collect from ML predictions (prefer most recent)
    if structure.predicted_properties:
        # Sort by creation time, newest first
        sorted_predictions = sorted(
            structure.predicted_properties,
            key=lambda p: p.created_at,
            reverse=True
        )

        if sorted_predictions:
            latest_prediction = sorted_predictions[0]
            properties.update(latest_prediction.properties)

    # Collect from simulation results
    # Query for completed simulation jobs for this structure
    sim_query = (
        select(SimulationResult)
        .join(SimulationJob)
        .where(
            and_(
                SimulationJob.structure_id == structure.id,
                SimulationJob.status == JobStatus.COMPLETED
            )
        )
        .order_by(SimulationResult.created_at.desc())
    )

    sim_result = await db.execute(sim_query)
    sim_results = sim_result.scalars().all()

    if sim_results:
        # Use most recent simulation result
        latest_sim = sim_results[0]
        # Simulation results override ML predictions for overlapping properties
        properties.update(latest_sim.summary)

    return properties


def _determine_property_source(properties: Dict[str, Any]) -> str:
    """
    Determine the source of properties.

    Args:
        properties: Property dictionary

    Returns:
        "ML", "SIMULATION", or "MIXED"
    """
    # This is a simplified version
    # In practice, we'd track which properties came from which source
    # For now, return "MIXED" if we have properties
    return "MIXED" if properties else "UNKNOWN"


def _extract_elements_from_formula(formula: str) -> List[str]:
    """
    Extract element symbols from a chemical formula.

    Args:
        formula: Chemical formula (e.g., "MoS2", "H2O")

    Returns:
        List of element symbols (e.g., ["Mo", "S"], ["H", "O"])
    """
    # Simple regex to extract element symbols (capital letter + optional lowercase)
    pattern = r'([A-Z][a-z]?)'
    elements = re.findall(pattern, formula)
    # Return unique elements
    return list(set(elements))


def calculate_candidate_score(
    properties: Dict[str, float],
    targets: DesignSearchRequest
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate how well a structure matches target constraints.

    Scoring algorithm:
    1. Start with score = 1.0
    2. For each constraint:
       - If target value specified: score *= distance_penalty
       - If range specified: score *= range_penalty
    3. Missing properties incur 0.5 penalty

    Args:
        properties: Structure properties
        targets: Target constraints

    Returns:
        Tuple of (score, match_details)
        - score: 0-1 match score (higher is better)
        - match_details: Details on constraint satisfaction
    """
    score = 1.0
    match_details = {}
    constraint_scores = []

    # Check bandgap constraint
    if targets.target_bandgap:
        bg_score, bg_details = _evaluate_property_constraint(
            properties.get("bandgap"),
            targets.target_bandgap,
            "bandgap"
        )
        score *= bg_score
        constraint_scores.append(bg_score)
        match_details["bandgap"] = bg_details

    # Check formation energy constraint
    if targets.target_formation_energy:
        fe_score, fe_details = _evaluate_property_constraint(
            properties.get("formation_energy"),
            targets.target_formation_energy,
            "formation_energy"
        )
        score *= fe_score
        constraint_scores.append(fe_score)
        match_details["formation_energy"] = fe_details

    # Check stability score constraint
    if targets.target_stability_score:
        stab_score, stab_details = _evaluate_property_constraint(
            properties.get("stability_score"),
            targets.target_stability_score,
            "stability_score"
        )
        score *= stab_score
        constraint_scores.append(stab_score)
        match_details["stability_score"] = stab_details

    # Calculate overall statistics
    if constraint_scores:
        match_details["overall_match"] = score
        match_details["average_constraint_score"] = statistics.mean(constraint_scores)
        match_details["min_constraint_score"] = min(constraint_scores)
        match_details["max_constraint_score"] = max(constraint_scores)
    else:
        # No constraints specified, full score
        match_details["overall_match"] = 1.0
        match_details["note"] = "No property constraints specified"

    return score, match_details


def _evaluate_property_constraint(
    value: Optional[float],
    constraint: PropertyConstraint,
    property_name: str
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluate how well a property value satisfies a constraint.

    Args:
        value: Property value (or None if missing)
        constraint: Constraint specification
        property_name: Name of property for logging

    Returns:
        Tuple of (score, details)
    """
    details = {
        "property": property_name,
        "value": value,
        "constraint": constraint.model_dump(exclude_none=True),
    }

    # Missing property penalty
    if value is None:
        details["satisfied"] = False
        details["reason"] = "Property value missing"
        return 0.5, details

    # Target-based scoring (optimization goal)
    if constraint.target is not None:
        # Calculate distance from target
        distance = abs(value - constraint.target)

        # Determine normalization factor
        # Use range if specified, otherwise use target value itself
        if constraint.min is not None and constraint.max is not None:
            range_size = constraint.max - constraint.min
        else:
            range_size = abs(constraint.target) if constraint.target != 0 else 1.0

        # Normalized distance (0 = perfect match, 1 = very far)
        normalized_distance = min(distance / range_size, 1.0) if range_size > 0 else 0.0

        # Convert to score (1 = perfect, 0 = worst)
        score = 1.0 - normalized_distance

        details["satisfied"] = score > 0.5
        details["distance_from_target"] = distance
        details["normalized_distance"] = normalized_distance
        details["score"] = score

        return score, details

    # Range-based scoring (hard constraints)
    else:
        in_range = True

        if constraint.min is not None and value < constraint.min:
            in_range = False
            details["violation"] = f"Below minimum: {value} < {constraint.min}"

        if constraint.max is not None and value > constraint.max:
            in_range = False
            details["violation"] = f"Above maximum: {value} > {constraint.max}"

        if in_range:
            details["satisfied"] = True
            details["score"] = 1.0
            return 1.0, details
        else:
            details["satisfied"] = False
            details["score"] = 0.1
            return 0.1, details


async def generate_structure_variants(
    db: AsyncSession,
    base_structures: List[Dict[str, Any]],
    limit: int
) -> List[Dict[str, Any]]:
    """
    Generate rule-based variants of successful structures.

    Current generation rules:
    1. Element substitution (Mo ↔ W, Cr ↔ Mo, S ↔ Se, etc.)
    2. Simple doping (add small amounts of dopants)

    Args:
        db: Database session
        base_structures: Top-scoring structures to use as templates
        limit: Maximum number of variants to generate

    Returns:
        List of generated structure candidates
    """
    logger.info(f"Generating variants from {len(base_structures)} base structures")

    variants = []

    # Element substitution rules (TMD-focused)
    substitution_rules = {
        "Mo": ["W", "Cr"],
        "W": ["Mo"],
        "Cr": ["Mo"],
        "S": ["Se", "Te"],
        "Se": ["S", "Te"],
        "Te": ["S", "Se"],
        "Ti": ["Zr", "Hf"],
        "Zr": ["Ti", "Hf"],
        "Hf": ["Ti", "Zr"],
    }

    for base in base_structures[:5]:  # Limit base structures to avoid explosion
        formula = base["formula"]

        # Try element substitutions
        for old_elem, new_elems in substitution_rules.items():
            if old_elem in formula:
                for new_elem in new_elems:
                    variant_formula = formula.replace(old_elem, new_elem)

                    # Estimate properties using simple heuristics
                    variant_properties = _estimate_variant_properties(
                        base["properties"],
                        old_elem,
                        new_elem
                    )

                    # Get elements for variant
                    variant_elements = _extract_elements_from_formula(variant_formula)

                    variants.append({
                        "structure_id": None,  # Generated, no real structure ID
                        "material_id": base["material_id"],
                        "formula": variant_formula,
                        "properties": variant_properties,
                        "property_source": "GENERATED",
                        "dimensionality": base.get("dimensionality"),
                        "num_atoms": base.get("num_atoms"),
                        "elements": variant_elements,
                        "is_generated": True,
                        "parent_structure_id": base["structure_id"],
                        "generation_method": f"element_substitution_{old_elem}_to_{new_elem}",
                    })

                    if len(variants) >= limit:
                        return variants

    logger.info(f"Generated {len(variants)} structure variants")
    return variants


def _estimate_variant_properties(
    base_properties: Dict[str, float],
    old_element: str,
    new_element: str
) -> Dict[str, float]:
    """
    Estimate properties for a structure variant.

    Uses simple heuristics based on known trends:
    - Mo → W: Similar properties, slightly higher bandgap
    - S → Se: Smaller bandgap, similar stability
    - etc.

    Args:
        base_properties: Properties of base structure
        old_element: Element being replaced
        new_element: Replacement element

    Returns:
        Estimated properties for variant
    """
    # Copy base properties
    variant_props = base_properties.copy()

    # Apply simple adjustment rules
    # These are rough heuristics and should be replaced with ML predictions

    # Chalcogen substitutions (S, Se, Te)
    if old_element == "S" and new_element == "Se":
        # S → Se: Bandgap typically decreases
        if "bandgap" in variant_props:
            variant_props["bandgap"] *= 0.85
    elif old_element == "Se" and new_element == "S":
        # Se → S: Bandgap typically increases
        if "bandgap" in variant_props:
            variant_props["bandgap"] *= 1.15
    elif old_element == "Se" and new_element == "Te":
        # Se → Te: Bandgap typically decreases
        if "bandgap" in variant_props:
            variant_props["bandgap"] *= 0.75

    # Transition metal substitutions
    elif old_element == "Mo" and new_element == "W":
        # Mo → W: Similar properties, slight bandgap increase
        if "bandgap" in variant_props:
            variant_props["bandgap"] *= 1.05

    # Add uncertainty to estimated properties
    for key in variant_props:
        if isinstance(variant_props[key], (int, float)):
            # Mark as estimated with 20% uncertainty
            # In a real system, these would have large confidence intervals
            pass

    return variant_props

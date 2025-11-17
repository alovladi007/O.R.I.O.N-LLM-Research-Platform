"""
Design and optimization schemas for material discovery.

These schemas support:
- Property-based material search (bandgap, stability, etc.)
- Multi-constraint optimization
- Candidate ranking and scoring
- Rule-based structure generation
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class PropertyConstraint(BaseModel):
    """
    Constraint for a single material property.

    Supports both range-based and target-based constraints:
    - Range: min <= value <= max
    - Target: value should be as close as possible to target
    """
    min: Optional[float] = Field(None, description="Minimum acceptable value")
    max: Optional[float] = Field(None, description="Maximum acceptable value")
    target: Optional[float] = Field(None, description="Exact target value (optimization goal)")

    class Config:
        json_schema_extra = {
            "example": {
                "min": 1.0,
                "max": 3.0,
                "target": 2.0
            }
        }


class DesignSearchRequest(BaseModel):
    """
    Request schema for design-based material search.

    Supports filtering by:
    - Material properties (bandgap, formation energy, stability)
    - Structural features (dimensionality, composition, size)
    - Generated variants (rule-based substitutions)
    """
    # Property constraints
    target_bandgap: Optional[PropertyConstraint] = Field(
        None,
        description="Bandgap constraint in eV"
    )
    target_formation_energy: Optional[PropertyConstraint] = Field(
        None,
        description="Formation energy constraint in eV/atom"
    )
    target_stability_score: Optional[PropertyConstraint] = Field(
        None,
        description="Stability score constraint (0-1, higher is more stable)"
    )

    # Structural constraints
    dimensionality: Optional[int] = Field(
        None,
        ge=0,
        le=3,
        description="Material dimensionality: 0=molecule, 1=1D, 2=2D, 3=bulk"
    )
    elements: Optional[List[str]] = Field(
        None,
        description="Must contain these elements (e.g., ['Mo', 'S'])"
    )
    exclude_elements: Optional[List[str]] = Field(
        None,
        description="Must NOT contain these elements"
    )
    max_atoms: Optional[int] = Field(
        None,
        gt=0,
        description="Maximum number of atoms in structure"
    )
    min_atoms: Optional[int] = Field(
        None,
        gt=0,
        description="Minimum number of atoms in structure"
    )

    # Search parameters
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of candidates to return"
    )
    include_generated: bool = Field(
        default=False,
        description="Include rule-based structure variants"
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum match score threshold (0-1)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "target_bandgap": {
                    "min": 1.5,
                    "max": 3.0,
                    "target": 2.0
                },
                "target_stability_score": {
                    "min": 0.7
                },
                "dimensionality": 2,
                "elements": ["Mo", "S"],
                "max_atoms": 50,
                "limit": 20,
                "include_generated": False,
                "min_score": 0.5
            }
        }


class CandidateStructure(BaseModel):
    """
    A candidate structure matching search criteria.

    Includes:
    - Structure and material identifiers
    - Match score and ranking
    - Predicted/simulated properties
    - Details on constraint satisfaction
    """
    structure_id: uuid.UUID = Field(..., description="Structure unique ID")
    material_id: uuid.UUID = Field(..., description="Parent material ID")
    formula: str = Field(..., description="Chemical formula")
    score: float = Field(..., ge=0.0, le=1.0, description="Match score (0-1, higher is better)")

    # Properties
    properties: dict = Field(
        default_factory=dict,
        description="Material properties (bandgap, formation_energy, stability_score, etc.)"
    )
    property_source: str = Field(
        ...,
        description="Source of properties: ML, SIMULATION, or MIXED"
    )

    # Match details
    match_details: dict = Field(
        default_factory=dict,
        description="Details on which constraints are satisfied and by how much"
    )

    # Structural information
    dimensionality: Optional[int] = Field(None, description="0D/1D/2D/3D")
    num_atoms: Optional[int] = Field(None, description="Number of atoms")
    elements: Optional[List[str]] = Field(None, description="Elements in structure")

    # Generation info (if applicable)
    is_generated: bool = Field(
        default=False,
        description="True if this is a rule-based variant"
    )
    parent_structure_id: Optional[uuid.UUID] = Field(
        None,
        description="Parent structure ID if generated"
    )
    generation_method: Optional[str] = Field(
        None,
        description="Method used to generate variant (e.g., 'element_substitution')"
    )

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
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
                "match_details": {
                    "bandgap_match": 0.95,
                    "stability_match": 0.88,
                    "overall_match": 0.92
                },
                "dimensionality": 2,
                "num_atoms": 3,
                "elements": ["Mo", "S"],
                "is_generated": False
            }
        }


class DesignSearchResponse(BaseModel):
    """
    Response schema for design search.

    Returns:
    - List of candidate structures ranked by score
    - Total number of matches found
    - Search parameters used
    - Performance metrics
    """
    candidates: List[CandidateStructure] = Field(
        default_factory=list,
        description="Ranked list of candidate structures"
    )
    total_found: int = Field(
        ...,
        ge=0,
        description="Total number of structures matching criteria (before limit)"
    )
    search_params: dict = Field(
        default_factory=dict,
        description="Search parameters used (for reproducibility)"
    )
    search_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time taken to perform search in milliseconds"
    )

    # Summary statistics
    score_distribution: Optional[dict] = Field(
        None,
        description="Statistics on score distribution (min, max, mean, median)"
    )
    property_ranges: Optional[dict] = Field(
        None,
        description="Ranges of properties in results"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "candidates": [],
                "total_found": 15,
                "search_params": {
                    "target_bandgap": {"min": 1.5, "max": 3.0},
                    "dimensionality": 2
                },
                "search_time_ms": 123.45,
                "score_distribution": {
                    "min": 0.65,
                    "max": 0.95,
                    "mean": 0.82,
                    "median": 0.84
                }
            }
        }

"""
Design and optimization services for material discovery.

This module provides:
- Property-based structure search
- Multi-constraint optimization
- Candidate scoring and ranking
- Rule-based structure generation
"""

from .search import (
    search_existing_structures,
    calculate_candidate_score,
    generate_structure_variants,
)

__all__ = [
    "search_existing_structures",
    "calculate_candidate_score",
    "generate_structure_variants",
]

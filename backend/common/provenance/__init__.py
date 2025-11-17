"""
Provenance tracking utilities.

This module provides functions for recording and querying provenance data,
enabling complete audit trails and lineage tracking for all entities in NANO-OS.

Key features:
- Record events with detailed context
- Query provenance chains for entities
- Collect system information for reproducibility
- Track code versions via git
"""

from .tracker import (
    record_provenance,
    get_provenance_chain,
    get_system_info,
    get_code_version,
)

__all__ = [
    "record_provenance",
    "get_provenance_chain",
    "get_system_info",
    "get_code_version",
]

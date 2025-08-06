"""
ORION Knowledge Graph Module
===========================

Manages the materials knowledge graph with Neo4j backend.
"""

from .manager import KnowledgeGraphManager
from .schema import MaterialNode, ProcessNode, PropertyNode, MethodNode
from .queries import GraphQuery, GraphQueryBuilder
from .etl import ETLPipeline
from .conflict_resolution import (
    SourceMetadata,
    ProvenanceWeightedConsensus,
    ConflictResolutionService
)

__all__ = [
    # Core components
    "KnowledgeGraphManager",
    
    # Schema
    "MaterialNode",
    "ProcessNode", 
    "PropertyNode",
    "MethodNode",
    
    # Queries
    "GraphQuery",
    "GraphQueryBuilder",
    
    # ETL
    "ETLPipeline",
    
    # Conflict resolution
    "SourceMetadata",
    "ProvenanceWeightedConsensus",
    "ConflictResolutionService",
]
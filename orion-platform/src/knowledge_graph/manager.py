"""
Knowledge Graph Manager
======================

Manages the Neo4j knowledge graph for ORION.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import json
import asyncio
from contextlib import asynccontextmanager

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
    from neo4j.exceptions import Neo4jError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncDriver = AsyncSession = None

from ..core.exceptions import (
    KnowledgeGraphError,
    GraphConnectionError,
    GraphQueryError,
    ResourceNotFoundError,
    ResourceExistsError
)
from ..core.config_manager import KnowledgeGraphConfig

from .schema import (
    MaterialNode, ProcessNode, PropertyNode, MethodNode,
    EquipmentNode, PublicationNode, Relationship, RelationshipType,
    SchemaValidator
)
from .queries import GraphQueryBuilder

logger = logging.getLogger(__name__)


class KnowledgeGraphManager:
    """
    Manages the ORION knowledge graph using Neo4j.
    
    Features:
    - Node and relationship CRUD operations
    - Complex graph queries
    - Schema validation
    - Transaction support
    - Performance optimization
    """
    
    def __init__(self, config: KnowledgeGraphConfig):
        """
        Initialize knowledge graph manager.
        
        Args:
            config: Knowledge graph configuration
        """
        self.config = config
        self.driver: Optional[AsyncDriver] = None
        self.query_builder = GraphQueryBuilder()
        self.validator = SchemaValidator()
        self._initialized = False
        
        if not NEO4J_AVAILABLE:
            logger.warning("Neo4j driver not available. Using mock implementation.")
    
    async def initialize(self):
        """Initialize connection to Neo4j"""
        if self._initialized:
            return
        
        try:
            if NEO4J_AVAILABLE:
                self.driver = AsyncGraphDatabase.driver(
                    self.config.neo4j.uri,
                    auth=(self.config.neo4j.user, self.config.neo4j.password),
                    encrypted=self.config.neo4j.encrypted,
                    trust=self.config.neo4j.trust
                )
                
                # Verify connection
                async with self.driver.session() as session:
                    await session.run("RETURN 1")
                
                # Create constraints and indexes
                await self._create_schema_constraints()
                
                logger.info("Connected to Neo4j successfully")
            else:
                logger.info("Using mock Neo4j implementation")
            
            self._initialized = True
            
        except Exception as e:
            raise GraphConnectionError(f"Failed to connect to Neo4j: {e}")
    
    async def shutdown(self):
        """Close Neo4j connection"""
        if self.driver:
            await self.driver.close()
        self._initialized = False
        logger.info("Neo4j connection closed")
    
    @asynccontextmanager
    async def session(self):
        """Get a Neo4j session"""
        if not self._initialized:
            raise GraphConnectionError("Knowledge graph not initialized")
        
        if NEO4J_AVAILABLE and self.driver:
            async with self.driver.session() as session:
                yield session
        else:
            # Mock session for testing
            yield None
    
    async def _create_schema_constraints(self):
        """Create schema constraints and indexes"""
        if not NEO4J_AVAILABLE:
            return
        
        constraints = self.config.schema.constraints
        
        async with self.session() as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                    logger.info(f"Created constraint: {constraint}")
                except Neo4jError as e:
                    if "already exists" not in str(e):
                        logger.error(f"Failed to create constraint: {e}")
            
            # Create additional indexes
            indexes = [
                "CREATE INDEX material_formula IF NOT EXISTS FOR (m:Material) ON (m.formula)",
                "CREATE INDEX property_name IF NOT EXISTS FOR (p:Property) ON (p.name)",
                "CREATE INDEX process_type IF NOT EXISTS FOR (p:Process) ON (p.process_type)",
                "CREATE INDEX publication_doi IF NOT EXISTS FOR (p:Publication) ON (p.doi)",
            ]
            
            for index in indexes:
                try:
                    await session.run(index)
                    logger.info(f"Created index: {index}")
                except Neo4jError as e:
                    if "already exists" not in str(e):
                        logger.error(f"Failed to create index: {e}")
    
    # Node Operations
    
    async def create_material(self, material: MaterialNode) -> str:
        """Create a material node"""
        self.validator.validate_material(material)
        
        query = """
        CREATE (m:Material $properties)
        RETURN m.id as id
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, properties=material.to_dict())
                record = await result.single()
                logger.info(f"Created material: {material.formula}")
                return record["id"]
            except Neo4jError as e:
                if "already exists" in str(e):
                    raise ResourceExistsError(f"Material {material.id} already exists")
                raise GraphQueryError(f"Failed to create material: {e}")
    
    async def create_process(self, process: ProcessNode) -> str:
        """Create a process node"""
        self.validator.validate_process(process)
        
        query = """
        CREATE (p:Process $properties)
        RETURN p.id as id
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, properties=process.to_dict())
                record = await result.single()
                logger.info(f"Created process: {process.name}")
                return record["id"]
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to create process: {e}")
    
    async def create_property(self, property_node: PropertyNode) -> str:
        """Create a property node"""
        self.validator.validate_property(property_node)
        
        query = """
        CREATE (p:Property $properties)
        RETURN p.id as id
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, properties=property_node.to_dict())
                record = await result.single()
                logger.info(f"Created property: {property_node.name}")
                return record["id"]
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to create property: {e}")
    
    async def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship between nodes"""
        self.validator.validate_relationship(relationship)
        
        query = f"""
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        CREATE (a)-[r:{relationship.relationship_type.value} $properties]->(b)
        RETURN r
        """
        
        async with self.session() as session:
            try:
                result = await session.run(
                    query,
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    properties=relationship.to_dict()
                )
                await result.single()
                logger.info(f"Created relationship: {relationship.source_id} -> {relationship.target_id}")
                return True
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to create relationship: {e}")
    
    # Query Operations
    
    async def get_material(self, material_id: str) -> Optional[Dict[str, Any]]:
        """Get a material by ID"""
        query = """
        MATCH (m:Material {id: $id})
        RETURN m
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, id=material_id)
                record = await result.single()
                if record:
                    return dict(record["m"])
                return None
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to get material: {e}")
    
    async def find_materials_by_formula(self, formula: str) -> List[Dict[str, Any]]:
        """Find materials by chemical formula"""
        query = """
        MATCH (m:Material)
        WHERE m.formula = $formula OR m.formula CONTAINS $formula
        RETURN m
        ORDER BY m.created_at DESC
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, formula=formula)
                materials = []
                async for record in result:
                    materials.append(dict(record["m"]))
                return materials
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to find materials: {e}")
    
    async def get_material_properties(self, material_id: str) -> List[Dict[str, Any]]:
        """Get all properties of a material"""
        query = """
        MATCH (m:Material {id: $id})-[:HAS_PROPERTY]->(p:Property)
        RETURN p
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, id=material_id)
                properties = []
                async for record in result:
                    properties.append(dict(record["p"]))
                return properties
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to get material properties: {e}")
    
    async def get_synthesis_methods(self, material_id: str) -> List[Dict[str, Any]]:
        """Get synthesis methods for a material"""
        query = """
        MATCH (m:Material {id: $id})-[:SYNTHESIZED_BY]->(p:Process)
        OPTIONAL MATCH (p)-[:USES_METHOD]->(method:Method)
        OPTIONAL MATCH (p)-[:REQUIRES_EQUIPMENT]->(eq:Equipment)
        RETURN p, collect(DISTINCT method) as methods, collect(DISTINCT eq) as equipment
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, id=material_id)
                synthesis_methods = []
                async for record in result:
                    process = dict(record["p"])
                    process["methods"] = [dict(m) for m in record["methods"] if m]
                    process["equipment"] = [dict(e) for e in record["equipment"] if e]
                    synthesis_methods.append(process)
                return synthesis_methods
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to get synthesis methods: {e}")
    
    async def find_similar_materials(self, material_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Find materials similar to a given material"""
        query = """
        MATCH (m1:Material {id: $id})-[:HAS_PROPERTY]->(p:Property)<-[:HAS_PROPERTY]-(m2:Material)
        WHERE m1 <> m2
        WITH m2, count(DISTINCT p) as common_properties
        ORDER BY common_properties DESC
        LIMIT $limit
        RETURN m2, common_properties
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, id=material_id, limit=limit)
                similar_materials = []
                async for record in result:
                    material = dict(record["m2"])
                    material["similarity_score"] = record["common_properties"]
                    similar_materials.append(material)
                return similar_materials
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to find similar materials: {e}")
    
    async def search(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph with natural language query.
        
        Args:
            query: Search query
            filters: Optional filters (node_type, property_range, etc.)
            
        Returns:
            List of matching nodes
        """
        # Build search query based on filters
        cypher_query = self.query_builder.build_search_query(query, filters)
        
        async with self.session() as session:
            try:
                result = await session.run(cypher_query, query=query, **(filters or {}))
                results = []
                async for record in result:
                    results.append(dict(record))
                return results
            except Neo4jError as e:
                raise GraphQueryError(f"Search failed: {e}")
    
    # Batch Operations
    
    async def batch_create_nodes(self, nodes: List[Union[MaterialNode, ProcessNode, PropertyNode]]) -> List[str]:
        """Create multiple nodes in a batch"""
        created_ids = []
        
        async with self.session() as session:
            async with session.begin_transaction() as tx:
                try:
                    for node in nodes:
                        if isinstance(node, MaterialNode):
                            query = "CREATE (m:Material $properties) RETURN m.id as id"
                            self.validator.validate_material(node)
                        elif isinstance(node, ProcessNode):
                            query = "CREATE (p:Process $properties) RETURN p.id as id"
                            self.validator.validate_process(node)
                        elif isinstance(node, PropertyNode):
                            query = "CREATE (p:Property $properties) RETURN p.id as id"
                            self.validator.validate_property(node)
                        else:
                            continue
                        
                        result = await tx.run(query, properties=node.to_dict())
                        record = await result.single()
                        created_ids.append(record["id"])
                    
                    await tx.commit()
                    logger.info(f"Batch created {len(created_ids)} nodes")
                    return created_ids
                    
                except Exception as e:
                    await tx.rollback()
                    raise GraphQueryError(f"Batch creation failed: {e}")
    
    async def batch_create_relationships(self, relationships: List[Relationship]) -> int:
        """Create multiple relationships in a batch"""
        created_count = 0
        
        async with self.session() as session:
            async with session.begin_transaction() as tx:
                try:
                    for rel in relationships:
                        self.validator.validate_relationship(rel)
                        
                        query = f"""
                        MATCH (a {{id: $source_id}})
                        MATCH (b {{id: $target_id}})
                        CREATE (a)-[r:{rel.relationship_type.value} $properties]->(b)
                        RETURN r
                        """
                        
                        await tx.run(
                            query,
                            source_id=rel.source_id,
                            target_id=rel.target_id,
                            properties=rel.to_dict()
                        )
                        created_count += 1
                    
                    await tx.commit()
                    logger.info(f"Batch created {created_count} relationships")
                    return created_count
                    
                except Exception as e:
                    await tx.rollback()
                    raise GraphQueryError(f"Batch relationship creation failed: {e}")
    
    # Update Operations
    
    async def update_material(self, material_id: str, updates: Dict[str, Any]) -> bool:
        """Update a material node"""
        set_clause = ", ".join([f"m.{k} = ${k}" for k in updates.keys()])
        query = f"""
        MATCH (m:Material {{id: $id}})
        SET {set_clause}, m.updated_at = $updated_at
        RETURN m
        """
        
        params = {"id": material_id, "updated_at": datetime.now().isoformat()}
        params.update(updates)
        
        async with self.session() as session:
            try:
                result = await session.run(query, **params)
                record = await result.single()
                if record:
                    logger.info(f"Updated material: {material_id}")
                    return True
                return False
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to update material: {e}")
    
    # Delete Operations
    
    async def delete_node(self, node_id: str) -> bool:
        """Delete a node and its relationships"""
        query = """
        MATCH (n {id: $id})
        DETACH DELETE n
        RETURN count(n) as deleted
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, id=node_id)
                record = await result.single()
                deleted = record["deleted"] > 0
                if deleted:
                    logger.info(f"Deleted node: {node_id}")
                return deleted
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to delete node: {e}")
    
    # Statistics and Analytics
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        query = """
        MATCH (m:Material) WITH count(m) as material_count
        MATCH (p:Process) WITH material_count, count(p) as process_count
        MATCH (prop:Property) WITH material_count, process_count, count(prop) as property_count
        MATCH (pub:Publication) WITH material_count, process_count, property_count, count(pub) as publication_count
        MATCH ()-[r]->() WITH material_count, process_count, property_count, publication_count, count(r) as relationship_count
        RETURN {
            materials: material_count,
            processes: process_count,
            properties: property_count,
            publications: publication_count,
            relationships: relationship_count,
            total_nodes: material_count + process_count + property_count + publication_count
        } as stats
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query)
                record = await result.single()
                return record["stats"] if record else {}
            except Neo4jError as e:
                logger.error(f"Failed to get stats: {e}")
                return {}
    
    async def get_property_distribution(self, property_name: str) -> List[Dict[str, Any]]:
        """Get distribution of a property across materials"""
        query = """
        MATCH (m:Material)-[:HAS_PROPERTY]->(p:Property {name: $property_name})
        WHERE p.value_type = 'numeric'
        RETURN m.formula as material, toFloat(p.value) as value, p.unit as unit
        ORDER BY value DESC
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, property_name=property_name)
                distribution = []
                async for record in result:
                    distribution.append({
                        "material": record["material"],
                        "value": record["value"],
                        "unit": record["unit"]
                    })
                return distribution
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to get property distribution: {e}")
    
    # Data Export/Import
    
    async def export_subgraph(self, root_id: str, depth: int = 2) -> Dict[str, Any]:
        """Export a subgraph starting from a root node"""
        query = """
        MATCH path = (root {id: $id})-[*0..$depth]-(connected)
        WITH collect(DISTINCT nodes(path)) as node_lists, collect(DISTINCT relationships(path)) as rel_lists
        UNWIND node_lists as nodes
        UNWIND nodes as node
        WITH collect(DISTINCT node) as all_nodes, rel_lists
        UNWIND rel_lists as rels
        UNWIND rels as rel
        WITH all_nodes, collect(DISTINCT rel) as all_rels
        RETURN all_nodes, all_rels
        """
        
        async with self.session() as session:
            try:
                result = await session.run(query, id=root_id, depth=depth)
                record = await result.single()
                
                if record:
                    nodes = [dict(node) for node in record["all_nodes"]]
                    relationships = []
                    for rel in record["all_rels"]:
                        rel_dict = dict(rel)
                        rel_dict["source"] = rel.start_node["id"]
                        rel_dict["target"] = rel.end_node["id"]
                        rel_dict["type"] = rel.type
                        relationships.append(rel_dict)
                    
                    return {
                        "nodes": nodes,
                        "relationships": relationships,
                        "root_id": root_id,
                        "export_date": datetime.now().isoformat()
                    }
                return {"nodes": [], "relationships": []}
                
            except Neo4jError as e:
                raise GraphQueryError(f"Failed to export subgraph: {e}")
    
    async def update_from_ingestion(self, ingested_data: Dict[str, Any]):
        """Update knowledge graph from ingested data"""
        # This would be implemented based on the data ingestion format
        # For now, it's a placeholder
        logger.info(f"Updating knowledge graph with {len(ingested_data)} items")
        pass
"""
Knowledge Graph Query Builder
============================

Builds complex Cypher queries for the knowledge graph.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class QueryType(str, Enum):
    """Types of graph queries"""
    SEARCH = "search"
    PATH = "path"
    PATTERN = "pattern"
    AGGREGATE = "aggregate"
    RECOMMENDATION = "recommendation"


@dataclass
class GraphQuery:
    """Represents a graph query"""
    query_type: QueryType
    cypher: str
    parameters: Dict[str, Any]
    description: str = ""
    
    def __str__(self) -> str:
        return self.cypher


class GraphQueryBuilder:
    """
    Builds complex Cypher queries for various graph operations.
    
    Features:
    - Natural language to Cypher translation
    - Query optimization
    - Pattern matching
    - Path finding
    - Aggregation queries
    """
    
    def __init__(self):
        """Initialize query builder"""
        self.query_templates = self._init_query_templates()
        self.property_mappings = self._init_property_mappings()
    
    def _init_query_templates(self) -> Dict[str, str]:
        """Initialize common query templates"""
        return {
            "find_by_property": """
                MATCH (n:{node_type})
                WHERE n.{property} {operator} $value
                RETURN n
                ORDER BY n.created_at DESC
                LIMIT $limit
            """,
            
            "find_by_relationship": """
                MATCH (n:{node_type})-[:{rel_type}]->(target:{target_type})
                WHERE {conditions}
                RETURN n, target
                LIMIT $limit
            """,
            
            "shortest_path": """
                MATCH path = shortestPath(
                    (start:{start_type} {{id: $start_id}})-[*..{max_depth}]-
                    (end:{end_type} {{id: $end_id}})
                )
                RETURN path
            """,
            
            "pattern_match": """
                MATCH {pattern}
                WHERE {conditions}
                RETURN {return_clause}
                ORDER BY {order_by}
                LIMIT $limit
            """,
            
            "aggregate_properties": """
                MATCH (n:{node_type})-[:HAS_PROPERTY]->(p:Property {{name: $property_name}})
                WHERE p.value_type = 'numeric'
                RETURN 
                    avg(toFloat(p.value)) as avg_value,
                    min(toFloat(p.value)) as min_value,
                    max(toFloat(p.value)) as max_value,
                    stdev(toFloat(p.value)) as stdev_value,
                    count(p) as count,
                    p.unit as unit
            """,
            
            "recommend_materials": """
                MATCH (target:Material {{id: $material_id}})
                MATCH (target)-[:HAS_PROPERTY]->(p1:Property)<-[:HAS_PROPERTY]-(similar:Material)
                WHERE target <> similar
                WITH similar, count(DISTINCT p1) as common_props
                
                OPTIONAL MATCH (similar)-[:HAS_PROPERTY]->(p2:Property)
                WHERE NOT EXISTS((target)-[:HAS_PROPERTY]->(p2))
                
                RETURN 
                    similar,
                    common_props,
                    collect(DISTINCT p2) as unique_props
                ORDER BY common_props DESC
                LIMIT $limit
            """,
        }
    
    def _init_property_mappings(self) -> Dict[str, str]:
        """Initialize property name mappings"""
        return {
            # Common property aliases
            "bandgap": "band_gap",
            "band gap": "band_gap",
            "eg": "band_gap",
            "melting point": "melting_temperature",
            "mp": "melting_temperature",
            "boiling point": "boiling_temperature",
            "bp": "boiling_temperature",
            "thermal conductivity": "thermal_conductivity",
            "electrical conductivity": "electrical_conductivity",
            "young's modulus": "youngs_modulus",
            "elastic modulus": "youngs_modulus",
            "hardness": "hardness",
            "density": "density",
            "refractive index": "refractive_index",
            "dielectric constant": "dielectric_constant",
        }
    
    def build_search_query(self, search_text: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a search query from natural language.
        
        Args:
            search_text: Natural language search query
            filters: Optional filters to apply
            
        Returns:
            Cypher query string
        """
        # Extract search intent
        intent = self._extract_search_intent(search_text)
        
        # Build base query
        if intent["type"] == "material_by_property":
            query = self._build_property_search(intent, filters)
        elif intent["type"] == "material_by_formula":
            query = self._build_formula_search(intent, filters)
        elif intent["type"] == "process_by_type":
            query = self._build_process_search(intent, filters)
        elif intent["type"] == "similar_materials":
            query = self._build_similarity_search(intent, filters)
        else:
            query = self._build_general_search(search_text, filters)
        
        return query
    
    def _extract_search_intent(self, search_text: str) -> Dict[str, Any]:
        """Extract search intent from natural language"""
        search_lower = search_text.lower()
        
        # Check for property searches
        property_patterns = [
            r"materials? with (\w+) (greater|less|equal|between|above|below) than? ([\d.]+)",
            r"find materials? where (\w+) is ([\d.]+)",
            r"(\w+) of ([\d.]+)",
        ]
        
        for pattern in property_patterns:
            match = re.search(pattern, search_lower)
            if match:
                return {
                    "type": "material_by_property",
                    "property": self._normalize_property_name(match.group(1)),
                    "operator": self._parse_operator(match.group(2) if len(match.groups()) > 2 else "="),
                    "value": float(match.group(-1))
                }
        
        # Check for formula searches
        formula_pattern = r"[A-Z][a-z]?\d*"
        if re.search(formula_pattern, search_text):
            formulas = re.findall(formula_pattern, search_text)
            if formulas:
                return {
                    "type": "material_by_formula",
                    "formula": "".join(formulas)
                }
        
        # Check for process searches
        process_keywords = ["synthesis", "deposition", "growth", "fabrication", "processing"]
        for keyword in process_keywords:
            if keyword in search_lower:
                return {
                    "type": "process_by_type",
                    "process_type": keyword
                }
        
        # Check for similarity searches
        if "similar to" in search_lower or "like" in search_lower:
            return {"type": "similar_materials"}
        
        return {"type": "general"}
    
    def _normalize_property_name(self, property_name: str) -> str:
        """Normalize property name using mappings"""
        property_lower = property_name.lower().strip()
        return self.property_mappings.get(property_lower, property_lower.replace(" ", "_"))
    
    def _parse_operator(self, operator_text: str) -> str:
        """Parse comparison operator from text"""
        operator_map = {
            "greater": ">",
            "less": "<",
            "equal": "=",
            "above": ">",
            "below": "<",
            "between": "BETWEEN",
            "is": "=",
        }
        return operator_map.get(operator_text.lower(), "=")
    
    def _build_property_search(self, intent: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> str:
        """Build a property-based search query"""
        query = """
        MATCH (m:Material)-[:HAS_PROPERTY]->(p:Property {name: $property_name})
        WHERE p.value_type = 'numeric' AND toFloat(p.value) {operator} $value
        """
        
        if filters:
            if "structure_type" in filters:
                query += " AND m.structure_type = $structure_type"
            if "crystal_system" in filters:
                query += " AND m.crystal_system = $crystal_system"
        
        query += """
        RETURN m, p
        ORDER BY toFloat(p.value) DESC
        LIMIT 20
        """
        
        return query.format(operator=intent["operator"])
    
    def _build_formula_search(self, intent: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> str:
        """Build a formula-based search query"""
        query = """
        MATCH (m:Material)
        WHERE m.formula = $formula OR m.formula CONTAINS $formula
        """
        
        if filters:
            if "has_property" in filters:
                query += """
                AND EXISTS((m)-[:HAS_PROPERTY]->(:Property {name: $has_property}))
                """
        
        query += """
        OPTIONAL MATCH (m)-[:HAS_PROPERTY]->(p:Property)
        RETURN m, collect(p) as properties
        ORDER BY m.created_at DESC
        LIMIT 10
        """
        
        return query
    
    def _build_process_search(self, intent: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> str:
        """Build a process-based search query"""
        query = """
        MATCH (p:Process)
        WHERE p.process_type = $process_type
        """
        
        if filters:
            if "temperature_range" in filters:
                query += """
                AND p.temperature >= $min_temp AND p.temperature <= $max_temp
                """
        
        query += """
        OPTIONAL MATCH (m:Material)-[:SYNTHESIZED_BY]->(p)
        OPTIONAL MATCH (p)-[:USES_METHOD]->(method:Method)
        RETURN p, collect(DISTINCT m) as materials, collect(DISTINCT method) as methods
        ORDER BY p.created_at DESC
        LIMIT 20
        """
        
        return query
    
    def _build_similarity_search(self, intent: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> str:
        """Build a similarity search query"""
        return self.query_templates["recommend_materials"]
    
    def _build_general_search(self, search_text: str, filters: Optional[Dict[str, Any]]) -> str:
        """Build a general text search query"""
        query = """
        CALL db.index.fulltext.queryNodes('materials_fulltext', $search_text)
        YIELD node, score
        WHERE score > 0.5
        """
        
        if filters and "node_type" in filters:
            query += f" AND '{filters['node_type']}' IN labels(node)"
        
        query += """
        RETURN node, score
        ORDER BY score DESC
        LIMIT 20
        """
        
        return query
    
    def build_path_query(self, start_id: str, end_id: str, 
                        max_depth: int = 5, 
                        relationship_types: Optional[List[str]] = None) -> GraphQuery:
        """
        Build a path-finding query.
        
        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            max_depth: Maximum path depth
            relationship_types: Optional list of relationship types to traverse
            
        Returns:
            GraphQuery object
        """
        if relationship_types:
            rel_pattern = "|".join(relationship_types)
            rel_clause = f"[:{rel_pattern}*..{max_depth}]"
        else:
            rel_clause = f"[*..{max_depth}]"
        
        cypher = f"""
        MATCH path = shortestPath(
            (start {{id: $start_id}})-{rel_clause}-(end {{id: $end_id}})
        )
        RETURN path, length(path) as path_length
        """
        
        return GraphQuery(
            query_type=QueryType.PATH,
            cypher=cypher,
            parameters={"start_id": start_id, "end_id": end_id},
            description=f"Find shortest path from {start_id} to {end_id}"
        )
    
    def build_pattern_query(self, pattern: str, where_clause: Optional[str] = None,
                           return_clause: str = "*", limit: int = 100) -> GraphQuery:
        """
        Build a pattern matching query.
        
        Args:
            pattern: Cypher pattern to match
            where_clause: Optional WHERE conditions
            return_clause: What to return
            limit: Result limit
            
        Returns:
            GraphQuery object
        """
        cypher = f"MATCH {pattern}"
        
        if where_clause:
            cypher += f"\nWHERE {where_clause}"
        
        cypher += f"\nRETURN {return_clause}\nLIMIT {limit}"
        
        return GraphQuery(
            query_type=QueryType.PATTERN,
            cypher=cypher,
            parameters={"limit": limit},
            description=f"Pattern match: {pattern}"
        )
    
    def build_aggregation_query(self, node_type: str, property_name: str,
                               group_by: Optional[str] = None) -> GraphQuery:
        """
        Build an aggregation query.
        
        Args:
            node_type: Type of nodes to aggregate
            property_name: Property to aggregate
            group_by: Optional grouping field
            
        Returns:
            GraphQuery object
        """
        if group_by:
            cypher = f"""
            MATCH (n:{node_type})-[:HAS_PROPERTY]->(p:Property {{name: $property_name}})
            WHERE p.value_type = 'numeric'
            WITH n.{group_by} as group_key, p
            RETURN 
                group_key,
                avg(toFloat(p.value)) as avg_value,
                min(toFloat(p.value)) as min_value,
                max(toFloat(p.value)) as max_value,
                count(p) as count
            ORDER BY avg_value DESC
            """
        else:
            cypher = self.query_templates["aggregate_properties"].format(node_type=node_type)
        
        return GraphQuery(
            query_type=QueryType.AGGREGATE,
            cypher=cypher,
            parameters={"property_name": property_name},
            description=f"Aggregate {property_name} for {node_type}"
        )
    
    def build_recommendation_query(self, material_id: str, limit: int = 10,
                                  min_similarity: float = 0.5) -> GraphQuery:
        """
        Build a material recommendation query.
        
        Args:
            material_id: Source material ID
            limit: Number of recommendations
            min_similarity: Minimum similarity threshold
            
        Returns:
            GraphQuery object
        """
        cypher = """
        MATCH (target:Material {id: $material_id})
        MATCH (target)-[:HAS_PROPERTY]->(p1:Property)<-[:HAS_PROPERTY]-(similar:Material)
        WHERE target <> similar
        
        WITH target, similar, count(DISTINCT p1) as common_props
        MATCH (target)-[:HAS_PROPERTY]->(all_props:Property)
        WITH target, similar, common_props, count(DISTINCT all_props) as total_props
        
        WITH similar, common_props, total_props,
             toFloat(common_props) / toFloat(total_props) as similarity
        WHERE similarity >= $min_similarity
        
        OPTIONAL MATCH (similar)-[:HAS_PROPERTY]->(unique_prop:Property)
        WHERE NOT EXISTS((target)-[:HAS_PROPERTY]->(unique_prop))
        
        RETURN 
            similar,
            similarity,
            common_props,
            collect(DISTINCT unique_prop) as unique_properties
        ORDER BY similarity DESC
        LIMIT $limit
        """
        
        return GraphQuery(
            query_type=QueryType.RECOMMENDATION,
            cypher=cypher,
            parameters={
                "material_id": material_id,
                "limit": limit,
                "min_similarity": min_similarity
            },
            description=f"Recommend materials similar to {material_id}"
        )
    
    def build_subgraph_query(self, root_id: str, depth: int = 2,
                            node_types: Optional[List[str]] = None) -> GraphQuery:
        """
        Build a subgraph extraction query.
        
        Args:
            root_id: Root node ID
            depth: Depth of subgraph
            node_types: Optional filter for node types
            
        Returns:
            GraphQuery object
        """
        node_filter = ""
        if node_types:
            labels = " OR ".join([f"'{t}' IN labels(n)" for t in node_types])
            node_filter = f"WHERE {labels}"
        
        cypher = f"""
        MATCH path = (root {{id: $root_id}})-[*0..{depth}]-(n)
        {node_filter}
        WITH collect(DISTINCT path) as paths
        UNWIND paths as p
        WITH nodes(p) as nodes_in_path, relationships(p) as rels_in_path
        UNWIND nodes_in_path as n
        WITH collect(DISTINCT n) as all_nodes, rels_in_path
        UNWIND rels_in_path as r
        WITH all_nodes, collect(DISTINCT r) as all_rels
        RETURN all_nodes, all_rels
        """
        
        return GraphQuery(
            query_type=QueryType.PATTERN,
            cypher=cypher,
            parameters={"root_id": root_id},
            description=f"Extract subgraph around {root_id} with depth {depth}"
        )
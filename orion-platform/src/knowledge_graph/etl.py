"""
Knowledge Graph ETL Pipeline
===========================

Extract, Transform, Load pipeline for ingesting data into the knowledge graph.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from ..core.exceptions import DataIngestionError, ValidationError
from .schema import (
    MaterialNode, ProcessNode, PropertyNode, MethodNode,
    PublicationNode, Relationship, RelationshipType,
    StructureType, CrystalSystem, ProcessType
)

logger = logging.getLogger(__name__)


@dataclass
class ETLConfig:
    """ETL pipeline configuration"""
    batch_size: int = 1000
    parallel_workers: int = 4
    checkpoint_interval: int = 5000
    error_threshold: float = 0.05  # 5% error tolerance
    deduplication: bool = True


class DataParser:
    """Base class for data parsers"""
    
    def parse(self, data: Any) -> Dict[str, List[Any]]:
        """Parse data and return nodes and relationships"""
        raise NotImplementedError


class MaterialsProjectParser(DataParser):
    """Parser for Materials Project data"""
    
    def parse(self, data: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Parse Materials Project JSON data"""
        nodes = {
            "materials": [],
            "properties": [],
            "methods": []
        }
        relationships = []
        
        # Parse material
        material_id = f"mp-{data.get('material_id', str(uuid.uuid4()))}"
        material = MaterialNode(
            id=material_id,
            formula=data.get("pretty_formula", ""),
            structure_type=self._get_structure_type(data),
            crystal_system=self._get_crystal_system(data),
            space_group=data.get("spacegroup", {}).get("symbol"),
            source="Materials Project",
            metadata={
                "mp_id": data.get("material_id"),
                "icsd_ids": data.get("icsd_ids", [])
            }
        )
        nodes["materials"].append(material)
        
        # Parse properties
        property_mappings = {
            "band_gap": ("band_gap", "eV", "electronic"),
            "formation_energy_per_atom": ("formation_energy", "eV/atom", "thermodynamic"),
            "e_above_hull": ("energy_above_hull", "eV/atom", "thermodynamic"),
            "density": ("density", "g/cm³", "structural"),
            "volume": ("volume", "Ų", "structural"),
        }
        
        for mp_key, (prop_name, unit, prop_type) in property_mappings.items():
            if mp_key in data:
                prop_id = f"{material_id}-{prop_name}"
                property_node = PropertyNode(
                    id=prop_id,
                    name=prop_name,
                    value=float(data[mp_key]),
                    unit=unit,
                    property_type=prop_type,
                    source="Materials Project"
                )
                nodes["properties"].append(property_node)
                
                # Create relationship
                relationships.append(Relationship(
                    source_id=material_id,
                    target_id=prop_id,
                    relationship_type=RelationshipType.HAS_PROPERTY,
                    source="Materials Project"
                ))
        
        # Parse computational method
        if "calc_settings" in data:
            method_id = f"{material_id}-dft"
            method = MethodNode(
                id=method_id,
                name="DFT",
                method_type="computational",
                software="VASP",
                parameters=data.get("calc_settings", {}),
                source="Materials Project"
            )
            nodes["methods"].append(method)
            
            # Link properties to method
            for prop in nodes["properties"]:
                relationships.append(Relationship(
                    source_id=method_id,
                    target_id=prop.id,
                    relationship_type=RelationshipType.PRODUCES_PROPERTY,
                    source="Materials Project"
                ))
        
        return {
            "nodes": nodes,
            "relationships": relationships
        }
    
    def _get_structure_type(self, data: Dict[str, Any]) -> Optional[StructureType]:
        """Determine structure type from data"""
        # Simple heuristic based on dimensionality
        if data.get("is_layered"):
            return StructureType.TWO_D
        elif data.get("is_metal"):
            return StructureType.METAL
        else:
            return StructureType.THREE_D
    
    def _get_crystal_system(self, data: Dict[str, Any]) -> Optional[CrystalSystem]:
        """Extract crystal system from spacegroup data"""
        crystal_system_map = {
            "cubic": CrystalSystem.CUBIC,
            "tetragonal": CrystalSystem.TETRAGONAL,
            "orthorhombic": CrystalSystem.ORTHORHOMBIC,
            "hexagonal": CrystalSystem.HEXAGONAL,
            "trigonal": CrystalSystem.TRIGONAL,
            "monoclinic": CrystalSystem.MONOCLINIC,
            "triclinic": CrystalSystem.TRICLINIC,
        }
        
        system = data.get("spacegroup", {}).get("crystal_system", "").lower()
        return crystal_system_map.get(system)


class PublicationParser(DataParser):
    """Parser for scientific publications"""
    
    def parse(self, data: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Parse publication data"""
        nodes = {
            "publications": [],
            "materials": [],
            "properties": []
        }
        relationships = []
        
        # Parse publication
        pub_id = f"pub-{data.get('doi', str(uuid.uuid4())).replace('/', '-')}"
        publication = PublicationNode(
            id=pub_id,
            title=data.get("title", ""),
            authors=data.get("authors", []),
            year=data.get("year", datetime.now().year),
            journal=data.get("journal"),
            doi=data.get("doi"),
            abstract=data.get("abstract"),
            keywords=data.get("keywords", []),
            source=data.get("source", "Literature")
        )
        nodes["publications"].append(publication)
        
        # Extract materials mentioned
        for material_data in data.get("materials", []):
            material_id = f"{pub_id}-{material_data.get('formula', uuid.uuid4())}"
            material = MaterialNode(
                id=material_id,
                formula=material_data.get("formula", ""),
                name=material_data.get("name"),
                source=pub_id
            )
            nodes["materials"].append(material)
            
            # Create relationship
            relationships.append(Relationship(
                source_id=pub_id,
                target_id=material_id,
                relationship_type=RelationshipType.MENTIONS,
                source=pub_id
            ))
            
            # Extract properties if available
            for prop_data in material_data.get("properties", []):
                prop_id = f"{material_id}-{prop_data.get('name', uuid.uuid4())}"
                property_node = PropertyNode(
                    id=prop_id,
                    name=prop_data.get("name", ""),
                    value=prop_data.get("value"),
                    unit=prop_data.get("unit", ""),
                    property_type=prop_data.get("type", "unknown"),
                    source=pub_id
                )
                nodes["properties"].append(property_node)
                
                relationships.append(Relationship(
                    source_id=material_id,
                    target_id=prop_id,
                    relationship_type=RelationshipType.HAS_PROPERTY,
                    source=pub_id
                ))
        
        return {
            "nodes": nodes,
            "relationships": relationships
        }


class ExperimentalDataParser(DataParser):
    """Parser for experimental data"""
    
    def parse(self, data: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Parse experimental data"""
        nodes = {
            "materials": [],
            "processes": [],
            "properties": [],
            "methods": []
        }
        relationships = []
        
        experiment_id = data.get("experiment_id", str(uuid.uuid4()))
        
        # Parse material
        material_data = data.get("material", {})
        material_id = f"exp-{experiment_id}-{material_data.get('formula', uuid.uuid4())}"
        material = MaterialNode(
            id=material_id,
            formula=material_data.get("formula", ""),
            name=material_data.get("name"),
            source=f"Experiment-{experiment_id}"
        )
        nodes["materials"].append(material)
        
        # Parse synthesis process
        for process_data in data.get("processes", []):
            process_id = f"{material_id}-{process_data.get('type', 'process')}-{uuid.uuid4()}"
            process = ProcessNode(
                id=process_id,
                process_type=ProcessType(process_data.get("type", "synthesis")),
                name=process_data.get("name", ""),
                temperature=process_data.get("temperature"),
                pressure=process_data.get("pressure"),
                duration=process_data.get("duration"),
                atmosphere=process_data.get("atmosphere"),
                parameters=process_data.get("parameters", {}),
                source=f"Experiment-{experiment_id}"
            )
            nodes["processes"].append(process)
            
            relationships.append(Relationship(
                source_id=material_id,
                target_id=process_id,
                relationship_type=RelationshipType.SYNTHESIZED_BY,
                source=f"Experiment-{experiment_id}"
            ))
        
        # Parse characterization results
        for char_data in data.get("characterization", []):
            # Create method node
            method_id = f"{material_id}-{char_data.get('technique', 'char')}-{uuid.uuid4()}"
            method = MethodNode(
                id=method_id,
                name=char_data.get("technique", ""),
                method_type="experimental",
                parameters=char_data.get("parameters", {}),
                source=f"Experiment-{experiment_id}"
            )
            nodes["methods"].append(method)
            
            # Parse measured properties
            for prop_data in char_data.get("results", []):
                prop_id = f"{method_id}-{prop_data.get('property', uuid.uuid4())}"
                property_node = PropertyNode(
                    id=prop_id,
                    name=prop_data.get("property", ""),
                    value=prop_data.get("value"),
                    unit=prop_data.get("unit", ""),
                    property_type=prop_data.get("type", "experimental"),
                    uncertainty=prop_data.get("uncertainty"),
                    measurement_method=char_data.get("technique"),
                    conditions=prop_data.get("conditions", {}),
                    source=f"Experiment-{experiment_id}"
                )
                nodes["properties"].append(property_node)
                
                relationships.append(Relationship(
                    source_id=material_id,
                    target_id=prop_id,
                    relationship_type=RelationshipType.HAS_PROPERTY,
                    source=f"Experiment-{experiment_id}"
                ))
                
                relationships.append(Relationship(
                    source_id=method_id,
                    target_id=prop_id,
                    relationship_type=RelationshipType.PRODUCES_PROPERTY,
                    source=f"Experiment-{experiment_id}"
                ))
        
        return {
            "nodes": nodes,
            "relationships": relationships
        }


class ETLPipeline:
    """
    ETL pipeline for ingesting data into the knowledge graph.
    
    Features:
    - Multiple data source support
    - Parallel processing
    - Data validation
    - Deduplication
    - Error handling and recovery
    - Progress tracking
    """
    
    def __init__(self, config: ETLConfig, knowledge_graph_manager):
        """
        Initialize ETL pipeline.
        
        Args:
            config: ETL configuration
            knowledge_graph_manager: Knowledge graph manager instance
        """
        self.config = config
        self.kg_manager = knowledge_graph_manager
        
        # Initialize parsers
        self.parsers = {
            "materials_project": MaterialsProjectParser(),
            "publication": PublicationParser(),
            "experimental": ExperimentalDataParser(),
        }
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": 0,
            "duplicates": 0
        }
        
        # Deduplication cache
        self.seen_ids = set()
    
    async def ingest_file(self, file_path: Union[str, Path], 
                         data_type: str, 
                         parser_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest data from a file.
        
        Args:
            file_path: Path to data file
            data_type: Type of data (materials_project, publication, experimental)
            parser_kwargs: Additional arguments for parser
            
        Returns:
            Ingestion statistics
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DataIngestionError(f"File not found: {file_path}")
        
        logger.info(f"Starting ingestion of {file_path} as {data_type}")
        
        # Load data
        if file_path.suffix == ".json":
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.suffix == ".csv":
            data = pd.read_csv(file_path).to_dict('records')
        else:
            raise DataIngestionError(f"Unsupported file format: {file_path.suffix}")
        
        # Process data
        if isinstance(data, list):
            return await self.ingest_batch(data, data_type, parser_kwargs)
        else:
            return await self.ingest_single(data, data_type, parser_kwargs)
    
    async def ingest_batch(self, data_list: List[Dict[str, Any]], 
                          data_type: str,
                          parser_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ingest a batch of data records.
        
        Args:
            data_list: List of data records
            data_type: Type of data
            parser_kwargs: Additional arguments for parser
            
        Returns:
            Ingestion statistics
        """
        parser = self.parsers.get(data_type)
        if not parser:
            raise DataIngestionError(f"Unknown data type: {data_type}")
        
        total_records = len(data_list)
        logger.info(f"Processing {total_records} records")
        
        # Process in batches
        for i in range(0, total_records, self.config.batch_size):
            batch = data_list[i:i + self.config.batch_size]
            
            # Parse records in parallel
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                parse_futures = [
                    executor.submit(self._parse_record, record, parser, parser_kwargs)
                    for record in batch
                ]
                
                parsed_results = []
                for future in parse_futures:
                    try:
                        result = future.result()
                        if result:
                            parsed_results.append(result)
                    except Exception as e:
                        logger.error(f"Parse error: {e}")
                        self.stats["errors"] += 1
            
            # Deduplicate if enabled
            if self.config.deduplication:
                parsed_results = self._deduplicate(parsed_results)
            
            # Load into knowledge graph
            await self._load_batch(parsed_results)
            
            # Checkpoint
            if (i + len(batch)) % self.config.checkpoint_interval == 0:
                logger.info(f"Checkpoint: Processed {i + len(batch)}/{total_records} records")
                logger.info(f"Stats: {self.stats}")
        
        # Check error threshold
        error_rate = self.stats["errors"] / total_records if total_records > 0 else 0
        if error_rate > self.config.error_threshold:
            raise DataIngestionError(
                f"Error rate ({error_rate:.2%}) exceeded threshold ({self.config.error_threshold:.2%})"
            )
        
        return self.stats
    
    async def ingest_single(self, data: Dict[str, Any], 
                           data_type: str,
                           parser_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Ingest a single data record"""
        return await self.ingest_batch([data], data_type, parser_kwargs)
    
    def _parse_record(self, record: Dict[str, Any], 
                     parser: DataParser,
                     parser_kwargs: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Parse a single record"""
        try:
            self.stats["total_processed"] += 1
            
            # Apply parser kwargs if provided
            if parser_kwargs:
                record.update(parser_kwargs)
            
            return parser.parse(record)
            
        except Exception as e:
            logger.error(f"Failed to parse record: {e}")
            raise
    
    def _deduplicate(self, parsed_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate nodes based on ID"""
        deduplicated = []
        
        for result in parsed_results:
            # Check nodes
            for node_type, nodes in result.get("nodes", {}).items():
                unique_nodes = []
                for node in nodes:
                    if node.id not in self.seen_ids:
                        self.seen_ids.add(node.id)
                        unique_nodes.append(node)
                    else:
                        self.stats["duplicates"] += 1
                
                result["nodes"][node_type] = unique_nodes
            
            if any(result.get("nodes", {}).values()) or result.get("relationships"):
                deduplicated.append(result)
        
        return deduplicated
    
    async def _load_batch(self, parsed_results: List[Dict[str, Any]]):
        """Load parsed data into knowledge graph"""
        # Collect all nodes and relationships
        all_nodes = []
        all_relationships = []
        
        for result in parsed_results:
            for node_list in result.get("nodes", {}).values():
                all_nodes.extend(node_list)
            all_relationships.extend(result.get("relationships", []))
        
        # Batch create nodes
        if all_nodes:
            try:
                created_ids = await self.kg_manager.batch_create_nodes(all_nodes)
                self.stats["nodes_created"] += len(created_ids)
            except Exception as e:
                logger.error(f"Failed to create nodes: {e}")
                self.stats["errors"] += len(all_nodes)
        
        # Batch create relationships
        if all_relationships:
            try:
                created_count = await self.kg_manager.batch_create_relationships(all_relationships)
                self.stats["relationships_created"] += created_count
            except Exception as e:
                logger.error(f"Failed to create relationships: {e}")
                self.stats["errors"] += len(all_relationships)
    
    def reset_stats(self):
        """Reset ingestion statistics"""
        self.stats = {
            "total_processed": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": 0,
            "duplicates": 0
        }
        self.seen_ids.clear()
    
    async def validate_ingestion(self, sample_size: int = 100) -> Dict[str, Any]:
        """Validate ingested data by sampling and checking integrity"""
        validation_results = {
            "sample_size": sample_size,
            "valid_nodes": 0,
            "valid_relationships": 0,
            "orphaned_nodes": 0,
            "broken_relationships": 0
        }
        
        # Sample nodes and check
        # Implementation would sample nodes and verify relationships
        
        return validation_results
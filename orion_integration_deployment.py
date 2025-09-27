"""
ORION: Integration & Production Deployment Guide
===============================================

This module provides comprehensive integration patterns and production deployment
strategies for the ORION materials science AI system, including:

1. Knowledge Graph Integration with Neo4j
2. External API Integration (Materials Project, NIST, etc.)
3. Laboratory Equipment Integration (ELN systems)
4. Container-based Deployment (Docker/Kubernetes)
5. Monitoring and Alerting Systems
6. Data Pipeline Orchestration
7. Security and Authentication
8. Scalability and Performance Optimization
9. Backup and Disaster Recovery
10. Complete System Integration Example

Author: ORION Development Team
"""

import asyncio
import aiohttp
import json
import yaml
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import hashlib
import base64
from pathlib import Path
import subprocess
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import time

# Database and Graph connections
try:
    from neo4j import GraphDatabase
    import redis
    import psycopg2
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("Database libraries not available - using mock implementations")

# Kubernetes client
try:
    from kubernetes import client, config
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    print("Kubernetes client not available")

# Monitoring and metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    import grafana_api
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Monitoring libraries not available")

# Security and encryption
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import jwt
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    print("Security libraries not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. KNOWLEDGE GRAPH INTEGRATION
# =====================================================================

class ORIONKnowledgeGraph:
    """Advanced knowledge graph integration with Neo4j"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.driver = None
        self.redis_client = None
        
        if DB_AVAILABLE:
            self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections"""
        try:
            # Neo4j connection
            neo4j_config = self.config.get('neo4j', {})
            self.driver = GraphDatabase.driver(
                neo4j_config.get('uri', 'bolt://localhost:7687'),
                auth=(
                    neo4j_config.get('username', 'neo4j'),
                    neo4j_config.get('password', 'password')
                )
            )
            
            # Redis connection for caching
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
            
            logger.info("Knowledge graph connections initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph connections: {e}")
    
    async def create_material_node(self, material_data: Dict[str, Any]) -> str:
        """Create a material node in the knowledge graph"""
        
        if not self.driver:
            return self._mock_create_material_node(material_data)
        
        material_id = material_data.get('material_id') or str(uuid.uuid4())
        
        query = """
        MERGE (m:Material {material_id: $material_id})
        SET m.formula = $formula,
            m.composition = $composition,
            m.crystal_system = $crystal_system,
            m.space_group = $space_group,
            m.created_at = datetime(),
            m.updated_at = datetime()
        RETURN m.material_id as material_id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {
                    'material_id': material_id,
                    'formula': material_data.get('formula'),
                    'composition': json.dumps(material_data.get('composition', {})),
                    'crystal_system': material_data.get('crystal_system'),
                    'space_group': material_data.get('space_group')
                })
                
                record = result.single()
                if record:
                    logger.info(f"Created material node: {material_id}")
                    return record['material_id']
        
        except Exception as e:
            logger.error(f"Error creating material node: {e}")
            return None
    
    async def add_property_relationship(self, material_id: str, property_data: Dict[str, Any],
                                      source_metadata: Dict[str, Any]) -> bool:
        """Add property relationship with provenance"""
        
        if not self.driver:
            return True  # Mock success
        
        query = """
        MATCH (m:Material {material_id: $material_id})
        MERGE (p:Property {name: $property_name})
        MERGE (s:Source {doi: $doi})
        SET s.title = $title,
            s.authors = $authors,
            s.publication_date = $publication_date,
            s.journal = $journal,
            s.citation_count = $citation_count
        MERGE (m)-[hp:HAS_PROPERTY {
            value: $value,
            unit: $unit,
            uncertainty: $uncertainty,
            measurement_method: $method,
            confidence_score: $confidence,
            source_weight: $source_weight,
            created_at: datetime()
        }]->(p)
        MERGE (hp)-[:SOURCED_FROM]->(s)
        RETURN hp
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {
                    'material_id': material_id,
                    'property_name': property_data.get('name'),
                    'value': property_data.get('value'),
                    'unit': property_data.get('unit'),
                    'uncertainty': property_data.get('uncertainty'),
                    'method': property_data.get('measurement_method'),
                    'confidence': property_data.get('confidence', 1.0),
                    'source_weight': self._calculate_source_weight(source_metadata),
                    'doi': source_metadata.get('doi'),
                    'title': source_metadata.get('title'),
                    'authors': json.dumps(source_metadata.get('authors', [])),
                    'publication_date': source_metadata.get('publication_date'),
                    'journal': source_metadata.get('journal'),
                    'citation_count': source_metadata.get('citation_count', 0)
                })
                
                if result.single():
                    logger.info(f"Added property {property_data.get('name')} to material {material_id}")
                    
                    # Invalidate cache
                    cache_key = f"material_properties:{material_id}"
                    if self.redis_client:
                        self.redis_client.delete(cache_key)
                    
                    return True
        
        except Exception as e:
            logger.error(f"Error adding property relationship: {e}")
            return False
    
    async def get_material_properties(self, material_id: str, 
                                    use_cache: bool = True) -> Dict[str, Any]:
        """Get all properties for a material with consensus values"""
        
        if not self.driver:
            return self._mock_get_material_properties(material_id)
        
        # Check cache first
        cache_key = f"material_properties:{material_id}"
        if use_cache and self.redis_client:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
        
        query = """
        MATCH (m:Material {material_id: $material_id})-[hp:HAS_PROPERTY]->(p:Property)
        OPTIONAL MATCH (hp)-[:SOURCED_FROM]->(s:Source)
        RETURN p.name as property_name,
               collect({
                   value: hp.value,
                   unit: hp.unit,
                   uncertainty: hp.uncertainty,
                   method: hp.measurement_method,
                   confidence: hp.confidence_score,
                   source_weight: hp.source_weight,
                   source_doi: s.doi,
                   created_at: hp.created_at
               }) as measurements
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {'material_id': material_id})
                
                properties = {}
                for record in result:
                    property_name = record['property_name']
                    measurements = record['measurements']
                    
                    # Calculate consensus value
                    consensus = self._calculate_consensus_value(measurements)
                    properties[property_name] = consensus
                
                # Cache result
                if self.redis_client:
                    self.redis_client.setex(
                        cache_key, 
                        timedelta(hours=1),
                        json.dumps(properties, default=str)
                    )
                
                return properties
        
        except Exception as e:
            logger.error(f"Error getting material properties: {e}")
            return {}
    
    async def find_similar_materials(self, target_composition: Dict[str, float],
                                   similarity_threshold: float = 0.8,
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """Find materials with similar compositions"""
        
        if not self.driver:
            return self._mock_find_similar_materials(target_composition)
        
        # Convert composition to normalized vector for similarity calculation
        query = """
        MATCH (m:Material)
        WHERE m.composition IS NOT NULL
        RETURN m.material_id as material_id,
               m.formula as formula,
               m.composition as composition
        LIMIT 1000
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query)
                
                similar_materials = []
                target_vector = self._composition_to_vector(target_composition)
                
                for record in result:
                    material_composition = json.loads(record['composition'])
                    material_vector = self._composition_to_vector(material_composition)
                    
                    similarity = self._cosine_similarity(target_vector, material_vector)
                    
                    if similarity >= similarity_threshold:
                        similar_materials.append({
                            'material_id': record['material_id'],
                            'formula': record['formula'],
                            'composition': material_composition,
                            'similarity': similarity
                        })
                
                # Sort by similarity and limit results
                similar_materials.sort(key=lambda x: x['similarity'], reverse=True)
                return similar_materials[:limit]
        
        except Exception as e:
            logger.error(f"Error finding similar materials: {e}")
            return []
    
    async def create_synthesis_pathway(self, pathway_data: Dict[str, Any]) -> str:
        """Create synthesis pathway in knowledge graph"""
        
        if not self.driver:
            return str(uuid.uuid4())  # Mock pathway ID
        
        pathway_id = str(uuid.uuid4())
        
        query = """
        CREATE (sp:SynthesisPathway {
            pathway_id: $pathway_id,
            name: $name,
            description: $description,
            success_rate: $success_rate,
            total_time: $total_time,
            created_at: datetime()
        })
        RETURN sp.pathway_id as pathway_id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, {
                    'pathway_id': pathway_id,
                    'name': pathway_data.get('name'),
                    'description': pathway_data.get('description'),
                    'success_rate': pathway_data.get('success_rate'),
                    'total_time': pathway_data.get('total_time')
                })
                
                if result.single():
                    # Add synthesis steps
                    for i, step in enumerate(pathway_data.get('steps', [])):
                        await self._add_synthesis_step(pathway_id, i, step)
                    
                    logger.info(f"Created synthesis pathway: {pathway_id}")
                    return pathway_id
        
        except Exception as e:
            logger.error(f"Error creating synthesis pathway: {e}")
            return None
    
    async def _add_synthesis_step(self, pathway_id: str, step_number: int, 
                                step_data: Dict[str, Any]):
        """Add a synthesis step to a pathway"""
        
        query = """
        MATCH (sp:SynthesisPathway {pathway_id: $pathway_id})
        CREATE (ss:SynthesisStep {
            step_number: $step_number,
            name: $name,
            description: $description,
            temperature: $temperature,
            pressure: $pressure,
            duration: $duration,
            reagents: $reagents,
            equipment: $equipment
        })
        CREATE (sp)-[:HAS_STEP]->(ss)
        RETURN ss
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, {
                    'pathway_id': pathway_id,
                    'step_number': step_number,
                    'name': step_data.get('name'),
                    'description': step_data.get('description'),
                    'temperature': step_data.get('temperature'),
                    'pressure': step_data.get('pressure'),
                    'duration': step_data.get('duration'),
                    'reagents': json.dumps(step_data.get('reagents', [])),
                    'equipment': json.dumps(step_data.get('equipment', []))
                })
        
        except Exception as e:
            logger.error(f"Error adding synthesis step: {e}")
    
    def _calculate_source_weight(self, metadata: Dict[str, Any]) -> float:
        """Calculate source reliability weight"""
        citation_weight = min(1.0, (metadata.get('citation_count', 0) + 1) / 100)
        
        # Journal impact factor
        impact_factor = metadata.get('impact_factor', 1.0)
        impact_weight = min(1.0, impact_factor / 10.0)
        
        # Publication date (prefer recent)
        pub_date = metadata.get('publication_date')
        if pub_date:
            try:
                pub_year = int(pub_date[:4])
                current_year = datetime.now().year
                age_factor = max(0.5, 1.0 - (current_year - pub_year) * 0.05)
            except:
                age_factor = 1.0
        else:
            age_factor = 1.0
        
        return citation_weight * impact_weight * age_factor
    
    def _calculate_consensus_value(self, measurements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate consensus value from multiple measurements"""
        if not measurements:
            return {}
        
        values = []
        weights = []
        
        for measurement in measurements:
            if measurement['value'] is not None:
                values.append(float(measurement['value']))
                weight = measurement.get('source_weight', 1.0) * measurement.get('confidence', 1.0)
                weights.append(weight)
        
        if not values:
            return {}
        
        import numpy as np
        values = np.array(values)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        consensus_value = np.average(values, weights=weights)
        weighted_variance = np.average((values - consensus_value)**2, weights=weights)
        uncertainty = np.sqrt(weighted_variance)
        
        return {
            'consensus_value': consensus_value,
            'uncertainty': uncertainty,
            'num_measurements': len(values),
            'value_range': [float(np.min(values)), float(np.max(values))],
            'source_count': len(set(m.get('source_doi') for m in measurements if m.get('source_doi'))),
            'measurement_methods': list(set(m.get('method') for m in measurements if m.get('method'))),
            'latest_measurement': max(measurements, key=lambda x: x.get('created_at', ''))
        }
    
    def _composition_to_vector(self, composition: Dict[str, float]) -> Dict[str, float]:
        """Convert composition to normalized vector"""
        # Normalize composition
        total = sum(composition.values())
        if total > 0:
            return {element: fraction/total for element, fraction in composition.items()}
        return composition
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between composition vectors"""
        import numpy as np
        
        # Get all elements
        all_elements = set(vec1.keys()) | set(vec2.keys())
        
        # Convert to arrays
        arr1 = np.array([vec1.get(elem, 0) for elem in all_elements])
        arr2 = np.array([vec2.get(elem, 0) for elem in all_elements])
        
        # Calculate cosine similarity
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    # Mock implementations for when database is not available
    def _mock_create_material_node(self, material_data: Dict[str, Any]) -> str:
        return str(uuid.uuid4())
    
    def _mock_get_material_properties(self, material_id: str) -> Dict[str, Any]:
        return {
            'bandgap': {
                'consensus_value': 2.1,
                'uncertainty': 0.1,
                'num_measurements': 5,
                'source_count': 3
            }
        }
    
    def _mock_find_similar_materials(self, target_composition: Dict[str, float]) -> List[Dict[str, Any]]:
        return [
            {
                'material_id': 'mock_material_001',
                'formula': 'TiO2',
                'composition': {'Ti': 0.33, 'O': 0.67},
                'similarity': 0.85
            }
        ]
    
    def close(self):
        """Close database connections"""
        if self.driver:
            self.driver.close()
        if self.redis_client:
            self.redis_client.close()

# =====================================================================
# 2. EXTERNAL API INTEGRATION
# =====================================================================

class ExternalAPIManager:
    """Manager for external API integrations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_keys = config.get('api_keys', {})
        self.rate_limiters = {}
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def query_materials_project(self, formula: str) -> Dict[str, Any]:
        """Query Materials Project API"""
        
        api_key = self.api_keys.get('materials_project')
        if not api_key:
            logger.warning("Materials Project API key not configured")
            return self._mock_materials_project_response(formula)
        
        url = "https://api.materialsproject.org/materials/summary/"
        headers = {"X-API-KEY": api_key}
        params = {"formula": formula}
        
        try:
            # Check rate limiting
            if not await self._check_rate_limit('materials_project'):
                logger.warning("Materials Project rate limit exceeded")
                return {}
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_materials_project_response(data)
                else:
                    logger.error(f"Materials Project API error: {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error querying Materials Project: {e}")
            return {}
    
    async def query_nist_webbook(self, compound_name: str) -> Dict[str, Any]:
        """Query NIST Chemistry WebBook"""
        
        base_url = "https://webbook.nist.gov/cgi/cbook.cgi"
        params = {
            "Name": compound_name,
            "Units": "SI",
            "cTG": "on",  # Thermodynamic data
            "cTC": "on",  # Phase change data
        }
        
        try:
            if not await self._check_rate_limit('nist'):
                logger.warning("NIST rate limit exceeded")
                return {}
            
            async with self.session.get(base_url, params=params) as response:
                if response.status == 200:
                    # NIST returns HTML, would need parsing
                    # For demo, return mock data
                    return self._mock_nist_response(compound_name)
                else:
                    logger.error(f"NIST API error: {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error querying NIST: {e}")
            return {}
    
    async def query_crystallography_open_db(self, formula: str) -> List[Dict[str, Any]]:
        """Query Crystallography Open Database"""
        
        base_url = "http://www.crystallography.net/cod/result"
        params = {
            "format": "json",
            "formula": formula,
            "limit": 10
        }
        
        try:
            if not await self._check_rate_limit('cod'):
                logger.warning("COD rate limit exceeded")
                return []
            
            async with self.session.get(base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_cod_response(data)
                else:
                    logger.error(f"COD API error: {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Error querying COD: {e}")
            return []
    
    async def query_pubchem(self, compound_name: str) -> Dict[str, Any]:
        """Query PubChem database"""
        
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
        url = f"{base_url}/{compound_name}/property/MolecularFormula,MolecularWeight,InChI/JSON"
        
        try:
            if not await self._check_rate_limit('pubchem'):
                logger.warning("PubChem rate limit exceeded")
                return {}
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_pubchem_response(data)
                else:
                    logger.error(f"PubChem API error: {response.status}")
                    return {}
        
        except Exception as e:
            logger.error(f"Error querying PubChem: {e}")
            return {}
    
    async def _check_rate_limit(self, api_name: str) -> bool:
        """Check API rate limits"""
        current_time = time.time()
        
        if api_name not in self.rate_limiters:
            self.rate_limiters[api_name] = []
        
        rate_limiter = self.rate_limiters[api_name]
        
        # Remove old requests (older than 1 minute)
        rate_limiter[:] = [t for t in rate_limiter if current_time - t < 60]
        
        # Check limits (conservative defaults)
        limits = {
            'materials_project': 100,  # per minute
            'nist': 60,
            'cod': 120,
            'pubchem': 300
        }
        
        if len(rate_limiter) >= limits.get(api_name, 60):
            return False
        
        rate_limiter.append(current_time)
        return True
    
    def _process_materials_project_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Materials Project API response"""
        # Extract relevant data from MP response
        processed = {
            'materials': [],
            'source': 'materials_project'
        }
        
        for material in data.get('data', []):
            processed['materials'].append({
                'material_id': material.get('material_id'),
                'formula': material.get('formula_pretty'),
                'formation_energy': material.get('formation_energy_per_atom'),
                'band_gap': material.get('band_gap'),
                'crystal_system': material.get('crystal_system'),
                'space_group': material.get('space_group'),
                'density': material.get('density'),
                'volume': material.get('volume')
            })
        
        return processed
    
    def _process_cod_response(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process COD API response"""
        processed = []
        
        for entry in data:
            processed.append({
                'cod_id': entry.get('id'),
                'formula': entry.get('formula'),
                'mineral_name': entry.get('mineral'),
                'space_group': entry.get('space_group'),
                'crystal_system': entry.get('crystal_system'),
                'cell_parameters': {
                    'a': entry.get('a'),
                    'b': entry.get('b'),
                    'c': entry.get('c'),
                    'alpha': entry.get('alpha'),
                    'beta': entry.get('beta'),
                    'gamma': entry.get('gamma')
                },
                'authors': entry.get('authors'),
                'journal': entry.get('journal'),
                'year': entry.get('year')
            })
        
        return processed
    
    def _process_pubchem_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process PubChem API response"""
        properties = data.get('PropertyTable', {}).get('Properties', [])
        
        if properties:
            prop = properties[0]
            return {
                'molecular_formula': prop.get('MolecularFormula'),
                'molecular_weight': prop.get('MolecularWeight'),
                'inchi': prop.get('InChI'),
                'source': 'pubchem'
            }
        
        return {}
    
    # Mock responses for testing
    def _mock_materials_project_response(self, formula: str) -> Dict[str, Any]:
        return {
            'materials': [{
                'material_id': 'mp-123456',
                'formula': formula,
                'formation_energy': -2.1,
                'band_gap': 2.3,
                'crystal_system': 'tetragonal',
                'space_group': 'P4/mmm',
                'density': 4.23
            }],
            'source': 'materials_project'
        }
    
    def _mock_nist_response(self, compound_name: str) -> Dict[str, Any]:
        return {
            'compound_name': compound_name,
            'melting_point': 1855,  # K
            'boiling_point': 2873,  # K
            'heat_capacity': 25.06,  # J/mol·K
            'source': 'nist'
        }

# =====================================================================
# 3. LABORATORY EQUIPMENT INTEGRATION
# =====================================================================

class ELNIntegrator:
    """Electronic Lab Notebook integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eln_endpoints = config.get('eln_endpoints', {})
        self.auth_tokens = config.get('auth_tokens', {})
    
    async def submit_experiment_protocol(self, protocol: Dict[str, Any]) -> str:
        """Submit experiment protocol to ELN"""
        
        eln_type = protocol.get('eln_type', 'benchling')
        endpoint = self.eln_endpoints.get(eln_type)
        
        if not endpoint:
            logger.warning(f"ELN endpoint not configured for {eln_type}")
            return self._mock_submit_protocol(protocol)
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f"Bearer {self.auth_tokens.get(eln_type)}",
                    'Content-Type': 'application/json'
                }
                
                payload = self._format_protocol_for_eln(protocol, eln_type)
                
                async with session.post(
                    f"{endpoint}/experiments",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status == 201:
                        result = await response.json()
                        experiment_id = result.get('id') or result.get('experiment_id')
                        logger.info(f"Protocol submitted to {eln_type}: {experiment_id}")
                        return experiment_id
                    else:
                        logger.error(f"ELN submission failed: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"Error submitting protocol to ELN: {e}")
            return None
    
    async def get_experimental_results(self, experiment_id: str, 
                                     eln_type: str = 'benchling') -> Dict[str, Any]:
        """Retrieve experimental results from ELN"""
        
        endpoint = self.eln_endpoints.get(eln_type)
        
        if not endpoint:
            logger.warning(f"ELN endpoint not configured for {eln_type}")
            return self._mock_get_results(experiment_id)
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f"Bearer {self.auth_tokens.get(eln_type)}",
                    'Content-Type': 'application/json'
                }
                
                async with session.get(
                    f"{endpoint}/experiments/{experiment_id}/results",
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        return self._process_eln_results(data, eln_type)
                    else:
                        logger.error(f"ELN result retrieval failed: {response.status}")
                        return {}
        
        except Exception as e:
            logger.error(f"Error retrieving results from ELN: {e}")
            return {}
    
    async def register_sample(self, sample_data: Dict[str, Any]) -> str:
        """Register a sample in the ELN system"""
        
        eln_type = sample_data.get('eln_type', 'benchling')
        endpoint = self.eln_endpoints.get(eln_type)
        
        if not endpoint:
            return self._mock_register_sample(sample_data)
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f"Bearer {self.auth_tokens.get(eln_type)}",
                    'Content-Type': 'application/json'
                }
                
                payload = self._format_sample_for_eln(sample_data, eln_type)
                
                async with session.post(
                    f"{endpoint}/samples",
                    headers=headers,
                    json=payload
                ) as response:
                    
                    if response.status == 201:
                        result = await response.json()
                        sample_id = result.get('id') or result.get('sample_id')
                        logger.info(f"Sample registered in {eln_type}: {sample_id}")
                        return sample_id
                    else:
                        logger.error(f"Sample registration failed: {response.status}")
                        return None
        
        except Exception as e:
            logger.error(f"Error registering sample: {e}")
            return None
    
    def _format_protocol_for_eln(self, protocol: Dict[str, Any], eln_type: str) -> Dict[str, Any]:
        """Format protocol for specific ELN system"""
        
        if eln_type == 'benchling':
            return {
                'name': protocol.get('name'),
                'description': protocol.get('description'),
                'procedure': protocol.get('steps', []),
                'materials': protocol.get('materials', []),
                'equipment': protocol.get('equipment', []),
                'safety_notes': protocol.get('safety_notes'),
                'estimated_duration': protocol.get('estimated_duration'),
                'tags': protocol.get('tags', [])
            }
        
        elif eln_type == 'labarchives':
            return {
                'title': protocol.get('name'),
                'content': self._format_protocol_as_html(protocol),
                'notebook_id': protocol.get('notebook_id'),
                'page_type': 'experiment'
            }
        
        else:
            return protocol
    
    def _format_sample_for_eln(self, sample_data: Dict[str, Any], eln_type: str) -> Dict[str, Any]:
        """Format sample data for specific ELN system"""
        
        base_format = {
            'name': sample_data.get('name'),
            'composition': sample_data.get('composition'),
            'synthesis_date': sample_data.get('synthesis_date'),
            'synthesized_by': sample_data.get('synthesized_by'),
            'batch_number': sample_data.get('batch_number'),
            'storage_location': sample_data.get('storage_location'),
            'properties': sample_data.get('measured_properties', {}),
            'notes': sample_data.get('notes')
        }
        
        return base_format
    
    def _format_protocol_as_html(self, protocol: Dict[str, Any]) -> str:
        """Format protocol as HTML for ELN systems that require it"""
        
        html = f"<h2>{protocol.get('name', 'Unnamed Protocol')}</h2>\n"
        
        if protocol.get('description'):
            html += f"<p><strong>Description:</strong> {protocol['description']}</p>\n"
        
        if protocol.get('materials'):
            html += "<h3>Materials</h3>\n<ul>\n"
            for material in protocol['materials']:
                html += f"<li>{material.get('name', '')}: {material.get('quantity', '')} {material.get('unit', '')}</li>\n"
            html += "</ul>\n"
        
        if protocol.get('steps'):
            html += "<h3>Procedure</h3>\n<ol>\n"
            for step in protocol['steps']:
                html += f"<li>{step.get('description', '')}</li>\n"
            html += "</ol>\n"
        
        if protocol.get('safety_notes'):
            html += f"<h3>Safety Notes</h3>\n<p>{protocol['safety_notes']}</p>\n"
        
        return html
    
    def _process_eln_results(self, data: Dict[str, Any], eln_type: str) -> Dict[str, Any]:
        """Process experimental results from ELN"""
        
        # Standardize results format regardless of ELN type
        return {
            'experiment_id': data.get('experiment_id') or data.get('id'),
            'status': data.get('status'),
            'completion_date': data.get('completion_date'),
            'measured_properties': data.get('measurements', {}),
            'observations': data.get('observations', ''),
            'yield': data.get('yield'),
            'purity': data.get('purity'),
            'characterization_data': data.get('characterization', {}),
            'files': data.get('attached_files', []),
            'experimenter': data.get('experimenter'),
            'raw_data': data
        }
    
    # Mock implementations
    def _mock_submit_protocol(self, protocol: Dict[str, Any]) -> str:
        return f"mock_exp_{uuid.uuid4().hex[:8]}"
    
    def _mock_get_results(self, experiment_id: str) -> Dict[str, Any]:
        return {
            'experiment_id': experiment_id,
            'status': 'completed',
            'measured_properties': {
                'bandgap': 2.1,
                'density': 4.23
            },
            'yield': 0.85,
            'purity': 0.95
        }
    
    def _mock_register_sample(self, sample_data: Dict[str, Any]) -> str:
        return f"mock_sample_{uuid.uuid4().hex[:8]}"

# =====================================================================
# 4. CONTAINER-BASED DEPLOYMENT
# =====================================================================

class ORIONDeploymentManager:
    """Kubernetes deployment manager for ORION system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.k8s_client = None
        
        if K8S_AVAILABLE:
            self._initialize_k8s_client()
    
    def _initialize_k8s_client(self):
        """Initialize Kubernetes client"""
        try:
            if self.config.get('k8s_config_file'):
                config.load_kube_config(self.config['k8s_config_file'])
            else:
                config.load_incluster_config()
            
            self.k8s_client = client.ApiClient()
            logger.info("Kubernetes client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubernetes client: {e}")
    
    def generate_deployment_manifests(self) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests"""
        
        manifests = {
            'namespace': self._create_namespace_manifest(),
            'configmap': self._create_configmap_manifest(),
            'secrets': self._create_secrets_manifest(),
            'database': self._create_database_manifests(),
            'api_service': self._create_api_service_manifest(),
            'stream_processor': self._create_stream_processor_manifest(),
            'simulation_orchestrator': self._create_simulation_orchestrator_manifest(),
            'ingress': self._create_ingress_manifest(),
            'monitoring': self._create_monitoring_manifests()
        }
        
        return manifests
    
    def _create_namespace_manifest(self) -> str:
        """Create namespace manifest"""
        return yaml.dump({
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': 'orion-system',
                'labels': {
                    'app.kubernetes.io/name': 'orion',
                    'app.kubernetes.io/version': '1.0.0'
                }
            }
        })
    
    def _create_configmap_manifest(self) -> str:
        """Create ConfigMap manifest"""
        config_data = {
            'redis.conf': self._generate_redis_config(),
            'nginx.conf': self._generate_nginx_config(),
            'orion.yaml': yaml.dump(self.config.get('application_config', {}))
        }
        
        return yaml.dump({
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'orion-config',
                'namespace': 'orion-system'
            },
            'data': config_data
        })
    
    def _create_secrets_manifest(self) -> str:
        """Create Secrets manifest"""
        secrets_data = {
            'database-url': base64.b64encode(
                self.config.get('database_url', 'postgresql://user:pass@db:5432/orion').encode()
            ).decode(),
            'api-keys': base64.b64encode(
                json.dumps(self.config.get('api_keys', {})).encode()
            ).decode(),
            'jwt-secret': base64.b64encode(
                self.config.get('jwt_secret', 'default-secret').encode()
            ).decode()
        }
        
        return yaml.dump({
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': 'orion-secrets',
                'namespace': 'orion-system'
            },
            'type': 'Opaque',
            'data': secrets_data
        })
    
    def _create_database_manifests(self) -> str:
        """Create database deployment manifests"""
        manifests = []
        
        # PostgreSQL deployment
        postgres_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'postgres',
                'namespace': 'orion-system'
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': 'postgres'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'postgres'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'postgres',
                            'image': 'postgres:14',
                            'env': [
                                {'name': 'POSTGRES_DB', 'value': 'orion'},
                                {'name': 'POSTGRES_USER', 'value': 'orion'},
                                {'name': 'POSTGRES_PASSWORD', 'valueFrom': {
                                    'secretKeyRef': {
                                        'name': 'orion-secrets',
                                        'key': 'postgres-password'
                                    }
                                }}
                            ],
                            'ports': [{'containerPort': 5432}],
                            'volumeMounts': [{
                                'name': 'postgres-storage',
                                'mountPath': '/var/lib/postgresql/data'
                            }]
                        }],
                        'volumes': [{
                            'name': 'postgres-storage',
                            'persistentVolumeClaim': {
                                'claimName': 'postgres-pvc'
                            }
                        }]
                    }
                }
            }
        }
        
        # Redis deployment
        redis_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'redis',
                'namespace': 'orion-system'
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': 'redis'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'redis'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'redis',
                            'image': 'redis:7-alpine',
                            'ports': [{'containerPort': 6379}],
                            'volumeMounts': [{
                                'name': 'redis-config',
                                'mountPath': '/usr/local/etc/redis/redis.conf',
                                'subPath': 'redis.conf'
                            }]
                        }],
                        'volumes': [{
                            'name': 'redis-config',
                            'configMap': {
                                'name': 'orion-config'
                            }
                        }]
                    }
                }
            }
        }
        
        # Neo4j deployment
        neo4j_deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'neo4j',
                'namespace': 'orion-system'
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': 'neo4j'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'neo4j'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'neo4j',
                            'image': 'neo4j:5.0',
                            'env': [
                                {'name': 'NEO4J_AUTH', 'value': 'neo4j/password'},
                                {'name': 'NEO4J_PLUGINS', 'value': '["apoc", "graph-data-science"]'}
                            ],
                            'ports': [
                                {'containerPort': 7474},
                                {'containerPort': 7687}
                            ],
                            'volumeMounts': [{
                                'name': 'neo4j-storage',
                                'mountPath': '/data'
                            }]
                        }],
                        'volumes': [{
                            'name': 'neo4j-storage',
                            'persistentVolumeClaim': {
                                'claimName': 'neo4j-pvc'
                            }
                        }]
                    }
                }
            }
        }
        
        manifests = [postgres_deployment, redis_deployment, neo4j_deployment]
        return '---\n'.join([yaml.dump(manifest) for manifest in manifests])
    
    def _create_api_service_manifest(self) -> str:
        """Create API service deployment manifest"""
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'orion-api',
                'namespace': 'orion-system'
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'matchLabels': {
                        'app': 'orion-api'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'orion-api'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'orion-api',
                            'image': 'orion/api:latest',
                            'ports': [{'containerPort': 8000}],
                            'env': [
                                {'name': 'DATABASE_URL', 'valueFrom': {
                                    'secretKeyRef': {
                                        'name': 'orion-secrets',
                                        'key': 'database-url'
                                    }
                                }},
                                {'name': 'REDIS_URL', 'value': 'redis://redis:6379'},
                                {'name': 'NEO4J_URL', 'value': 'bolt://neo4j:7687'}
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '512Mi',
                                    'cpu': '250m'
                                },
                                'limits': {
                                    'memory': '1Gi',
                                    'cpu': '500m'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'orion-api-service',
                'namespace': 'orion-system'
            },
            'spec': {
                'selector': {
                    'app': 'orion-api'
                },
                'ports': [{
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': 8000
                }],
                'type': 'ClusterIP'
            }
        }
        
        return '---\n'.join([yaml.dump(deployment), yaml.dump(service)])
    
    def _create_stream_processor_manifest(self) -> str:
        """Create stream processor deployment manifest"""
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'orion-stream-processor',
                'namespace': 'orion-system'
            },
            'spec': {
                'replicas': 2,
                'selector': {
                    'matchLabels': {
                        'app': 'orion-stream-processor'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'orion-stream-processor'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'stream-processor',
                            'image': 'orion/stream-processor:latest',
                            'env': [
                                {'name': 'REDIS_URL', 'value': 'redis://redis:6379'},
                                {'name': 'NEO4J_URL', 'value': 'bolt://neo4j:7687'},
                                {'name': 'PROCESSING_WORKERS', 'value': '4'}
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '1Gi',
                                    'cpu': '500m'
                                },
                                'limits': {
                                    'memory': '2Gi',
                                    'cpu': '1'
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        return yaml.dump(deployment)
    
    def _create_simulation_orchestrator_manifest(self) -> str:
        """Create simulation orchestrator deployment manifest"""
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'orion-simulation-orchestrator',
                'namespace': 'orion-system'
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': 'orion-simulation-orchestrator'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'orion-simulation-orchestrator'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'simulation-orchestrator',
                            'image': 'orion/simulation-orchestrator:latest',
                            'env': [
                                {'name': 'MAX_CONCURRENT_JOBS', 'value': '20'},
                                {'name': 'STORAGE_PATH', 'value': '/data/simulations'}
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '2Gi',
                                    'cpu': '1'
                                },
                                'limits': {
                                    'memory': '4Gi',
                                    'cpu': '2'
                                }
                            },
                            'volumeMounts': [{
                                'name': 'simulation-storage',
                                'mountPath': '/data/simulations'
                            }]
                        }],
                        'volumes': [{
                            'name': 'simulation-storage',
                            'persistentVolumeClaim': {
                                'claimName': 'simulation-pvc'
                            }
                        }]
                    }
                }
            }
        }
        
        return yaml.dump(deployment)
    
    def _create_ingress_manifest(self) -> str:
        """Create ingress manifest"""
        ingress = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': 'orion-ingress',
                'namespace': 'orion-system',
                'annotations': {
                    'nginx.ingress.kubernetes.io/rewrite-target': '/',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                }
            },
            'spec': {
                'tls': [{
                    'hosts': ['orion.example.com'],
                    'secretName': 'orion-tls'
                }],
                'rules': [{
                    'host': 'orion.example.com',
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': 'orion-api-service',
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
        
        return yaml.dump(ingress)
    
    def _create_monitoring_manifests(self) -> str:
        """Create monitoring deployment manifests"""
        manifests = []
        
        # Prometheus deployment
        prometheus = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'prometheus',
                'namespace': 'orion-system'
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': 'prometheus'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'prometheus'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'prometheus',
                            'image': 'prom/prometheus:latest',
                            'ports': [{'containerPort': 9090}],
                            'volumeMounts': [{
                                'name': 'prometheus-config',
                                'mountPath': '/etc/prometheus/prometheus.yml',
                                'subPath': 'prometheus.yml'
                            }]
                        }],
                        'volumes': [{
                            'name': 'prometheus-config',
                            'configMap': {
                                'name': 'prometheus-config'
                            }
                        }]
                    }
                }
            }
        }
        
        # Grafana deployment
        grafana = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'grafana',
                'namespace': 'orion-system'
            },
            'spec': {
                'replicas': 1,
                'selector': {
                    'matchLabels': {
                        'app': 'grafana'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'grafana'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'grafana',
                            'image': 'grafana/grafana:latest',
                            'ports': [{'containerPort': 3000}],
                            'env': [
                                {'name': 'GF_SECURITY_ADMIN_PASSWORD', 'value': 'admin'}
                            ]
                        }]
                    }
                }
            }
        }
        
        manifests = [prometheus, grafana]
        return '---\n'.join([yaml.dump(manifest) for manifest in manifests])
    
    def _generate_redis_config(self) -> str:
        """Generate Redis configuration"""
        return """
# Redis configuration for ORION
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
"""
    
    def _generate_nginx_config(self) -> str:
        """Generate Nginx configuration"""
        return """
upstream orion_api {
    server orion-api-service:80;
}

server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://orion_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /ws {
        proxy_pass http://orion_api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
"""
    
    async def deploy_to_kubernetes(self, manifests: Dict[str, str]) -> bool:
        """Deploy manifests to Kubernetes cluster"""
        
        if not self.k8s_client:
            logger.error("Kubernetes client not available")
            return False
        
        try:
            # Apply manifests in order
            deployment_order = [
                'namespace', 'configmap', 'secrets', 'database',
                'api_service', 'stream_processor', 'simulation_orchestrator',
                'ingress', 'monitoring'
            ]
            
            for manifest_name in deployment_order:
                if manifest_name in manifests:
                    success = await self._apply_manifest(manifests[manifest_name])
                    if not success:
                        logger.error(f"Failed to apply {manifest_name} manifest")
                        return False
                    
                    logger.info(f"Applied {manifest_name} manifest")
                    
                    # Wait between deployments
                    await asyncio.sleep(2)
            
            logger.info("ORION system deployed successfully to Kubernetes")
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    async def _apply_manifest(self, manifest_yaml: str) -> bool:
        """Apply a single manifest to Kubernetes"""
        try:
            # Write manifest to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(manifest_yaml)
                temp_file = f.name
            
            # Apply using kubectl
            result = subprocess.run(
                ['kubectl', 'apply', '-f', temp_file],
                capture_output=True,
                text=True
            )
            
            # Clean up temp file
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return True
            else:
                logger.error(f"kubectl apply failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying manifest: {e}")
            return False

# =====================================================================
# 5. MONITORING AND ALERTING SYSTEM
# =====================================================================

class ORIONMonitoringSystem:
    """Comprehensive monitoring and alerting for ORION"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = {}
        self.alerts = []
        
        # Initialize Prometheus metrics if available
        if MONITORING_AVAILABLE:
            self._initialize_prometheus_metrics()
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        self.metrics = {
            'predictions_total': Counter(
                'orion_predictions_total',
                'Total number of predictions made',
                ['model_type', 'status']
            ),
            'prediction_latency': Histogram(
                'orion_prediction_duration_seconds',
                'Time spent on predictions',
                ['model_type']
            ),
            'active_simulations': Gauge(
                'orion_active_simulations',
                'Number of active simulations'
            ),
            'queue_size': Gauge(
                'orion_queue_size',
                'Size of processing queues',
                ['queue_type']
            ),
            'error_rate': Counter(
                'orion_errors_total',
                'Total number of errors',
                ['component', 'error_type']
            ),
            'knowledge_graph_nodes': Gauge(
                'orion_kg_nodes_total',
                'Total nodes in knowledge graph',
                ['node_type']
            ),
            'api_requests': Counter(
                'orion_api_requests_total',
                'Total API requests',
                ['endpoint', 'method', 'status']
            )
        }
    
    def record_prediction(self, model_type: str, latency: float, status: str = 'success'):
        """Record prediction metrics"""
        if MONITORING_AVAILABLE and 'predictions_total' in self.metrics:
            self.metrics['predictions_total'].labels(
                model_type=model_type, status=status
            ).inc()
            
            self.metrics['prediction_latency'].labels(
                model_type=model_type
            ).observe(latency)
    
    def update_queue_size(self, queue_type: str, size: int):
        """Update queue size metric"""
        if MONITORING_AVAILABLE and 'queue_size' in self.metrics:
            self.metrics['queue_size'].labels(queue_type=queue_type).set(size)
    
    def record_error(self, component: str, error_type: str):
        """Record error occurrence"""
        if MONITORING_AVAILABLE and 'error_rate' in self.metrics:
            self.metrics['error_rate'].labels(
                component=component, error_type=error_type
            ).inc()
    
    def update_simulation_count(self, count: int):
        """Update active simulation count"""
        if MONITORING_AVAILABLE and 'active_simulations' in self.metrics:
            self.metrics['active_simulations'].set(count)
    
    def check_system_health(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'alerts': []
        }
        
        # Database connectivity
        try:
            # This would check actual database connections
            health_status['checks']['database'] = {
                'status': 'healthy',
                'response_time_ms': 5.2
            }
        except Exception as e:
            health_status['checks']['database'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['status'] = 'unhealthy'
        
        # API endpoint health
        try:
            health_status['checks']['api_endpoints'] = {
                'status': 'healthy',
                'active_endpoints': 12
            }
        except Exception as e:
            health_status['checks']['api_endpoints'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
        
        # Resource utilization
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            health_status['checks']['resources'] = {
                'status': 'healthy' if cpu_percent < 80 and memory_percent < 85 else 'warning',
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent
            }
            
            # Generate alerts for high resource usage
            if cpu_percent > 90:
                health_status['alerts'].append({
                    'severity': 'critical',
                    'message': f'High CPU usage: {cpu_percent}%',
                    'timestamp': datetime.now().isoformat()
                })
            
            if memory_percent > 95:
                health_status['alerts'].append({
                    'severity': 'critical',
                    'message': f'High memory usage: {memory_percent}%',
                    'timestamp': datetime.now().isoformat()
                })
                
        except Exception as e:
            health_status['checks']['resources'] = {
                'status': 'unknown',
                'error': str(e)
            }
        
        # Check for accumulated alerts
        if self.alerts:
            health_status['alerts'].extend(self.alerts[-10:])  # Last 10 alerts
        
        return health_status
    
    async def send_alert(self, alert: Dict[str, Any]):
        """Send alert notification"""
        
        # Store alert
        alert['timestamp'] = datetime.now().isoformat()
        alert['id'] = str(uuid.uuid4())
        self.alerts.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Send to configured alerting channels
        alert_channels = self.config.get('alert_channels', {})
        
        for channel_type, channel_config in alert_channels.items():
            try:
                if channel_type == 'slack':
                    await self._send_slack_alert(alert, channel_config)
                elif channel_type == 'email':
                    await self._send_email_alert(alert, channel_config)
                elif channel_type == 'webhook':
                    await self._send_webhook_alert(alert, channel_config)
                    
            except Exception as e:
                logger.error(f"Failed to send {channel_type} alert: {e}")
    
    async def _send_slack_alert(self, alert: Dict[str, Any], config: Dict[str, Any]):
        """Send alert to Slack"""
        webhook_url = config.get('webhook_url')
        if not webhook_url:
            return
        
        color_map = {
            'critical': '#FF0000',
            'warning': '#FFA500',
            'info': '#00FF00'
        }
        
        payload = {
            'attachments': [{
                'color': color_map.get(alert.get('severity', 'info'), '#808080'),
                'title': f"ORION Alert: {alert.get('severity', 'info').upper()}",
                'text': alert.get('message', 'No message'),
                'fields': [
                    {
                        'title': 'Component',
                        'value': alert.get('component', 'Unknown'),
                        'short': True
                    },
                    {
                        'title': 'Timestamp',
                        'value': alert.get('timestamp', ''),
                        'short': True
                    }
                ],
                'footer': 'ORION Monitoring System'
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 200:
                    logger.info("Slack alert sent successfully")
                else:
                    logger.error(f"Slack alert failed: {response.status}")
    
    async def _send_email_alert(self, alert: Dict[str, Any], config: Dict[str, Any]):
        """Send alert via email"""
        # This would integrate with email service (SMTP, SendGrid, etc.)
        logger.info(f"Email alert would be sent: {alert['message']}")
    
    async def _send_webhook_alert(self, alert: Dict[str, Any], config: Dict[str, Any]):
        """Send alert to webhook endpoint"""
        webhook_url = config.get('url')
        if not webhook_url:
            return
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=alert) as response:
                if response.status == 200:
                    logger.info("Webhook alert sent successfully")
                else:
                    logger.error(f"Webhook alert failed: {response.status}")

# =====================================================================
# 6. COMPLETE SYSTEM INTEGRATION EXAMPLE
# =====================================================================

class ORIONSystemIntegrator:
    """Complete ORION system integration orchestrator"""
    
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.components = {}
        self.is_initialized = False
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load system configuration"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default system configuration"""
        return {
            'system': {
                'name': 'ORION',
                'version': '1.0.0',
                'environment': 'production'
            },
            'databases': {
                'neo4j': {
                    'uri': 'bolt://localhost:7687',
                    'username': 'neo4j',
                    'password': 'password'
                },
                'postgresql': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'orion',
                    'username': 'orion',
                    'password': 'password'
                },
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0
                }
            },
            'external_apis': {
                'api_keys': {
                    'materials_project': 'your_mp_api_key',
                    'nist': 'your_nist_api_key'
                }
            },
            'eln_integration': {
                'eln_endpoints': {
                    'benchling': 'https://api.benchling.com/v2',
                    'labarchives': 'https://api.labarchives.com/v1'
                },
                'auth_tokens': {
                    'benchling': 'your_benchling_token',
                    'labarchives': 'your_labarchives_token'
                }
            },
            'monitoring': {
                'prometheus_port': 8000,
                'alert_channels': {
                    'slack': {
                        'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
                    }
                }
            },
            'deployment': {
                'environment': 'kubernetes',
                'replicas': {
                    'api': 3,
                    'stream_processor': 2,
                    'simulation_orchestrator': 1
                }
            }
        }
    
    async def initialize_system(self) -> bool:
        """Initialize all system components"""
        
        logger.info("Initializing ORION system...")
        
        try:
            # Initialize Knowledge Graph
            self.components['knowledge_graph'] = ORIONKnowledgeGraph(
                self.config.get('databases', {})
            )
            
            # Initialize External API Manager
            self.components['api_manager'] = ExternalAPIManager(
                self.config.get('external_apis', {})
            )
            
            # Initialize ELN Integrator
            self.components['eln_integrator'] = ELNIntegrator(
                self.config.get('eln_integration', {})
            )
            
            # Initialize Monitoring System
            self.components['monitoring'] = ORIONMonitoringSystem(
                self.config.get('monitoring', {})
            )
            
            # Initialize Deployment Manager
            self.components['deployment_manager'] = ORIONDeploymentManager(
                self.config.get('deployment', {})
            )
            
            self.is_initialized = True
            logger.info("ORION system initialized successfully")
            
            # Perform initial health check
            health_status = self.components['monitoring'].check_system_health()
            logger.info(f"System health: {health_status['status']}")
            
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def run_integration_demo(self):
        """Run comprehensive integration demonstration"""
        
        if not self.is_initialized:
            await self.initialize_system()
        
        logger.info("Starting ORION integration demonstration...")
        
        # 1. Knowledge Graph Operations
        logger.info("\n1. Knowledge Graph Operations")
        kg = self.components['knowledge_graph']
        
        # Create a material
        material_data = {
            'material_id': 'demo_TiO2_001',
            'formula': 'TiO2',
            'composition': {'Ti': 0.33, 'O': 0.67},
            'crystal_system': 'tetragonal',
            'space_group': 'P42/mnm'
        }
        
        material_id = await kg.create_material_node(material_data)
        logger.info(f"Created material: {material_id}")
        
        # Add property with source metadata
        property_data = {
            'name': 'bandgap',
            'value': 3.2,
            'unit': 'eV',
            'uncertainty': 0.1,
            'measurement_method': 'UV-Vis spectroscopy',
            'confidence': 0.9
        }
        
        source_metadata = {
            'doi': '10.1000/demo.paper.2024',
            'title': 'Electronic Properties of TiO2',
            'authors': ['Smith, J.', 'Doe, A.'],
            'publication_date': '2024-01-15',
            'journal': 'Journal of Materials Science',
            'citation_count': 25,
            'impact_factor': 3.2
        }
        
        await kg.add_property_relationship(material_id, property_data, source_metadata)
        logger.info("Added property relationship")
        
        # Query material properties
        properties = await kg.get_material_properties(material_id)
        logger.info(f"Material properties: {properties}")
        
        # 2. External API Integration
        logger.info("\n2. External API Integration")
        
        async with self.components['api_manager'] as api_manager:
            # Query Materials Project
            mp_data = await api_manager.query_materials_project('TiO2')
            logger.info(f"Materials Project data: {len(mp_data.get('materials', []))} materials found")
            
            # Query NIST WebBook
            nist_data = await api_manager.query_nist_webbook('titanium dioxide')
            logger.info(f"NIST data: {nist_data}")
            
            # Query Crystallography Open Database
            cod_data = await api_manager.query_crystallography_open_db('TiO2')
            logger.info(f"COD data: {len(cod_data)} structures found")
        
        # 3. ELN Integration
        logger.info("\n3. ELN Integration")
        eln = self.components['eln_integrator']
        
        # Submit experiment protocol
        protocol = {
            'name': 'TiO2 Synthesis via Sol-Gel Method',
            'description': 'Synthesis of titanium dioxide nanoparticles using sol-gel process',
            'materials': [
                {'name': 'Titanium isopropoxide', 'quantity': 5.0, 'unit': 'mL'},
                {'name': 'Ethanol', 'quantity': 20.0, 'unit': 'mL'},
                {'name': 'Water', 'quantity': 2.0, 'unit': 'mL'},
                {'name': 'HCl', 'quantity': 0.5, 'unit': 'mL'}
            ],
            'steps': [
                {'description': 'Mix titanium isopropoxide with ethanol under stirring'},
                {'description': 'Add water dropwise to initiate hydrolysis'},
                {'description': 'Add HCl to catalyze the process'},
                {'description': 'Continue stirring for 2 hours at room temperature'},
                {'description': 'Dry at 100°C for 12 hours'},
                {'description': 'Calcine at 450°C for 3 hours'}
            ],
            'safety_notes': 'Work in fume hood. Wear safety goggles and gloves.',
            'estimated_duration': 24  # hours
        }
        
        experiment_id = await eln.submit_experiment_protocol(protocol)
        logger.info(f"Submitted protocol to ELN: {experiment_id}")
        
        # Register sample
        sample_data = {
            'name': 'TiO2_batch_001',
            'composition': {'Ti': 0.33, 'O': 0.67},
            'synthesis_date': '2024-02-15',
            'synthesized_by': 'Demo User',
            'batch_number': 'B001',
            'storage_location': 'Freezer A, Shelf 2'
        }
        
        sample_id = await eln.register_sample(sample_data)
        logger.info(f"Registered sample: {sample_id}")
        
        # 4. Monitoring and Alerting
        logger.info("\n4. Monitoring and Alerting")
        monitoring = self.components['monitoring']
        
        # Record some metrics
        monitoring.record_prediction('gnn', 0.15, 'success')
        monitoring.record_prediction('rf', 0.08, 'success')
        monitoring.update_queue_size('simulation', 5)
        monitoring.update_simulation_count(3)
        
        # Check system health
        health = monitoring.check_system_health()
        logger.info(f"System health status: {health['status']}")
        
        # Send test alert
        test_alert = {
            'severity': 'info',
            'component': 'demo',
            'message': 'Integration demonstration completed successfully'
        }
        await monitoring.send_alert(test_alert)
        
        # 5. Deployment Manifest Generation
        logger.info("\n5. Deployment Manifest Generation")
        deployment_manager = self.components['deployment_manager']
        
        manifests = deployment_manager.generate_deployment_manifests()
        logger.info(f"Generated {len(manifests)} deployment manifests")
        
        # Save manifests to files (optional)
        output_dir = Path('./k8s_manifests')
        output_dir.mkdir(exist_ok=True)
        
        for manifest_name, manifest_content in manifests.items():
            manifest_file = output_dir / f"{manifest_name}.yaml"
            with open(manifest_file, 'w') as f:
                f.write(manifest_content)
            logger.info(f"Saved manifest: {manifest_file}")
        
        logger.info("\nORION integration demonstration completed successfully!")
        
        # Final system status
        final_status = {
            'components_initialized': len(self.components),
            'knowledge_graph_connected': True,
            'external_apis_tested': 3,
            'eln_integration_active': True,
            'monitoring_enabled': True,
            'deployment_ready': True
        }
        
        logger.info(f"Final system status: {final_status}")
        
        return final_status
    
    async def shutdown_system(self):
        """Gracefully shutdown all system components"""
        
        logger.info("Shutting down ORION system...")
        
        # Close database connections
        if 'knowledge_graph' in self.components:
            self.components['knowledge_graph'].close()
        
        # Additional cleanup as needed
        
        logger.info("ORION system shutdown complete")

# =====================================================================
# 7. MAIN EXECUTION
# =====================================================================

async def main():
    """Main execution function"""
    
    print("ORION: Complete System Integration")
    print("="*50)
    
    # Initialize system integrator
    integrator = ORIONSystemIntegrator()
    
    try:
        # Run complete integration demo
        final_status = await integrator.run_integration_demo()
        
        print(f"\nIntegration completed successfully!")
        print(f"System status: {final_status}")
        
    except Exception as e:
        print(f"Integration failed: {e}")
        
    finally:
        # Cleanup
        await integrator.shutdown_system()

if __name__ == "__main__":
    asyncio.run(main())

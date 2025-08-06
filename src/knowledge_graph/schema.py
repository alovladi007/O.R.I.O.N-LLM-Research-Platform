"""
Knowledge Graph Schema Definitions
=================================

Defines the ontology and schema for the ORION knowledge graph.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class StructureType(str, Enum):
    """Material structure types"""
    TWO_D = "2D"
    THREE_D = "3D"
    ALLOY = "alloy"
    COMPOSITE = "composite"
    POLYMER = "polymer"
    CERAMIC = "ceramic"
    METAL = "metal"
    SEMICONDUCTOR = "semiconductor"


class CrystalSystem(str, Enum):
    """Crystal system types"""
    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"
    AMORPHOUS = "amorphous"


class ProcessType(str, Enum):
    """Process types"""
    SYNTHESIS = "synthesis"
    DEPOSITION = "deposition"
    LITHOGRAPHY = "lithography"
    ANNEALING = "annealing"
    ETCHING = "etching"
    DOPING = "doping"
    CHARACTERIZATION = "characterization"


class SynthesisMethod(str, Enum):
    """Synthesis methods"""
    SOL_GEL = "sol_gel"
    CVD = "cvd"
    ALD = "ald"
    PVD = "pvd"
    HYDROTHERMAL = "hydrothermal"
    SOLID_STATE = "solid_state"
    ELECTROCHEMICAL = "electrochemical"
    SOLUTION_PROCESSING = "solution_processing"
    MOLECULAR_BEAM_EPITAXY = "mbe"


@dataclass
class BaseNode:
    """Base class for all graph nodes"""
    id: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j"""
        data = {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence": self.confidence,
        }
        if self.source:
            data["source"] = self.source
        if self.metadata:
            data["metadata"] = json.dumps(self.metadata)
        return data


@dataclass
class MaterialNode(BaseNode):
    """Material entity in the knowledge graph"""
    formula: str
    name: Optional[str] = None
    structure_type: Optional[StructureType] = None
    crystal_system: Optional[CrystalSystem] = None
    space_group: Optional[str] = None
    lattice_parameters: Optional[Dict[str, float]] = None
    composition: Optional[Dict[str, float]] = None
    molecular_weight: Optional[float] = None
    density: Optional[float] = None
    synonyms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "formula": self.formula,
            "label": "Material"
        })
        
        if self.name:
            data["name"] = self.name
        if self.structure_type:
            data["structure_type"] = self.structure_type.value
        if self.crystal_system:
            data["crystal_system"] = self.crystal_system.value
        if self.space_group:
            data["space_group"] = self.space_group
        if self.lattice_parameters:
            data["lattice_parameters"] = json.dumps(self.lattice_parameters)
        if self.composition:
            data["composition"] = json.dumps(self.composition)
        if self.molecular_weight:
            data["molecular_weight"] = self.molecular_weight
        if self.density:
            data["density"] = self.density
        if self.synonyms:
            data["synonyms"] = self.synonyms
            
        return data


@dataclass
class ProcessNode(BaseNode):
    """Process entity in the knowledge graph"""
    process_type: ProcessType
    name: str
    temperature: Optional[float] = None  # Kelvin
    pressure: Optional[float] = None  # Pa
    duration: Optional[float] = None  # seconds
    atmosphere: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "process_type": self.process_type.value,
            "name": self.name,
            "label": "Process"
        })
        
        if self.temperature:
            data["temperature"] = self.temperature
        if self.pressure:
            data["pressure"] = self.pressure
        if self.duration:
            data["duration"] = self.duration
        if self.atmosphere:
            data["atmosphere"] = self.atmosphere
        if self.parameters:
            data["parameters"] = json.dumps(self.parameters)
            
        return data


@dataclass
class PropertyNode(BaseNode):
    """Property entity in the knowledge graph"""
    name: str
    value: Union[float, str, List[float]]
    unit: str
    property_type: str  # electrical, optical, mechanical, thermal, magnetic
    conditions: Optional[Dict[str, Any]] = None
    uncertainty: Optional[float] = None
    measurement_method: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "name": self.name,
            "unit": self.unit,
            "property_type": self.property_type,
            "label": "Property"
        })
        
        # Handle different value types
        if isinstance(self.value, list):
            data["value"] = json.dumps(self.value)
            data["value_type"] = "array"
        elif isinstance(self.value, (int, float)):
            data["value"] = float(self.value)
            data["value_type"] = "numeric"
        else:
            data["value"] = str(self.value)
            data["value_type"] = "string"
        
        if self.conditions:
            data["conditions"] = json.dumps(self.conditions)
        if self.uncertainty:
            data["uncertainty"] = self.uncertainty
        if self.measurement_method:
            data["measurement_method"] = self.measurement_method
            
        return data


@dataclass
class MethodNode(BaseNode):
    """Method/Technique entity in the knowledge graph"""
    name: str
    method_type: str  # computational, experimental
    software: Optional[str] = None
    version: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    accuracy: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "name": self.name,
            "method_type": self.method_type,
            "label": "Method"
        })
        
        if self.software:
            data["software"] = self.software
        if self.version:
            data["version"] = self.version
        if self.parameters:
            data["parameters"] = json.dumps(self.parameters)
        if self.accuracy:
            data["accuracy"] = self.accuracy
            
        return data


@dataclass
class EquipmentNode(BaseNode):
    """Equipment entity in the knowledge graph"""
    name: str
    model: str
    manufacturer: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    specifications: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "name": self.name,
            "model": self.model,
            "label": "Equipment"
        })
        
        if self.manufacturer:
            data["manufacturer"] = self.manufacturer
        if self.capabilities:
            data["capabilities"] = self.capabilities
        if self.specifications:
            data["specifications"] = json.dumps(self.specifications)
        if self.location:
            data["location"] = self.location
            
        return data


@dataclass
class PublicationNode(BaseNode):
    """Publication entity in the knowledge graph"""
    title: str
    authors: List[str]
    year: int
    journal: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data.update({
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "label": "Publication"
        })
        
        if self.journal:
            data["journal"] = self.journal
        if self.doi:
            data["doi"] = self.doi
        if self.abstract:
            data["abstract"] = self.abstract
        if self.keywords:
            data["keywords"] = self.keywords
            
        return data


# Relationship types
class RelationshipType(str, Enum):
    """Types of relationships in the knowledge graph"""
    # Material relationships
    HAS_PROPERTY = "HAS_PROPERTY"
    SYNTHESIZED_BY = "SYNTHESIZED_BY"
    DERIVED_FROM = "DERIVED_FROM"
    SIMILAR_TO = "SIMILAR_TO"
    
    # Process relationships
    USES_METHOD = "USES_METHOD"
    CONDUCTED_ON = "CONDUCTED_ON"
    REQUIRES_EQUIPMENT = "REQUIRES_EQUIPMENT"
    FOLLOWED_BY = "FOLLOWED_BY"
    
    # Property relationships
    MEASURED_IN_PROCESS = "MEASURED_IN_PROCESS"
    CALCULATED_BY = "CALCULATED_BY"
    CORRELATED_WITH = "CORRELATED_WITH"
    
    # Method relationships
    PRODUCES_PROPERTY = "PRODUCES_PROPERTY"
    VALIDATES = "VALIDATES"
    
    # Publication relationships
    MENTIONS = "MENTIONS"
    CITES = "CITES"
    AUTHORED_BY = "AUTHORED_BY"


@dataclass
class Relationship:
    """Relationship between nodes in the knowledge graph"""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "confidence": self.confidence,
        }
        if self.source:
            data["source"] = self.source
        if self.properties:
            data.update(self.properties)
        return data


# Schema validation
class SchemaValidator:
    """Validates nodes and relationships against the schema"""
    
    @staticmethod
    def validate_material(material: MaterialNode) -> bool:
        """Validate material node"""
        if not material.formula:
            raise ValueError("Material must have a formula")
        if material.structure_type and material.structure_type not in StructureType:
            raise ValueError(f"Invalid structure type: {material.structure_type}")
        if material.crystal_system and material.crystal_system not in CrystalSystem:
            raise ValueError(f"Invalid crystal system: {material.crystal_system}")
        return True
    
    @staticmethod
    def validate_process(process: ProcessNode) -> bool:
        """Validate process node"""
        if not process.name:
            raise ValueError("Process must have a name")
        if process.process_type not in ProcessType:
            raise ValueError(f"Invalid process type: {process.process_type}")
        if process.temperature and process.temperature < 0:
            raise ValueError("Temperature must be positive (Kelvin)")
        if process.pressure and process.pressure < 0:
            raise ValueError("Pressure must be positive")
        return True
    
    @staticmethod
    def validate_property(property_node: PropertyNode) -> bool:
        """Validate property node"""
        if not property_node.name:
            raise ValueError("Property must have a name")
        if not property_node.unit:
            raise ValueError("Property must have a unit")
        if property_node.uncertainty and property_node.uncertainty < 0:
            raise ValueError("Uncertainty must be non-negative")
        return True
    
    @staticmethod
    def validate_relationship(relationship: Relationship) -> bool:
        """Validate relationship"""
        if not relationship.source_id or not relationship.target_id:
            raise ValueError("Relationship must have source and target IDs")
        if relationship.relationship_type not in RelationshipType:
            raise ValueError(f"Invalid relationship type: {relationship.relationship_type}")
        if relationship.confidence < 0 or relationship.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        return True
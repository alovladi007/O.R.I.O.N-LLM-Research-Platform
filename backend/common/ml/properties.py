"""
ML-based property prediction for materials and structures.

This module provides stub implementations for material property prediction
using machine learning. The predictions are deterministic (based on structure
ID hashing) but realistic, allowing the API infrastructure to be developed
and tested before real ML models are integrated.

Future Integration:
-------------------
This module is designed to be a drop-in replacement point for real ML models:

1. CGCNN (Crystal Graph Convolutional Neural Networks)
   - Graph-based approach for crystal structures
   - Good for formation energy and band gap prediction
   - Integration: Replace predict_properties_for_structure with CGCNN inference

2. MEGNet (MatErials Graph Network)
   - Universal graph neural network for materials
   - Predicts multiple properties simultaneously
   - Integration: Load pre-trained MEGNet model, call predict()

3. M3GNET (Materials 3-body Graph Network)
   - Advanced architecture with 3-body interactions
   - High accuracy for energies and forces
   - Integration: Use m3gnet package for inference

4. ALIGNN (Atomistic Line Graph Neural Network)
   - State-of-the-art for many material properties
   - Uses line graphs for better geometric representation
   - Integration: Load ALIGNN checkpoint, run forward pass

Model Registry Design:
----------------------
The module uses a model registry pattern to support multiple models:
- Each model has a name, version, and availability status
- Models can be enabled/disabled via configuration
- Future: Load models dynamically from checkpoint files
- Future: Support model ensembles for uncertainty quantification
"""

import hashlib
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelName(str, Enum):
    """Available ML model names."""
    STUB = "STUB"              # Stub implementation (current)
    CGCNN = "CGCNN"            # Crystal Graph CNN
    MEGNET = "MEGNET"          # MatErials Graph Network
    M3GNET = "M3GNET"          # Materials 3-body Graph Network
    ALIGNN = "ALIGNN"          # Atomistic Line Graph Neural Network
    ENSEMBLE = "ENSEMBLE"      # Ensemble of multiple models


@dataclass
class ModelInfo:
    """Information about an available ML model."""
    name: str
    version: str
    available: bool
    description: str
    supported_properties: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "name": self.name,
            "version": self.version,
            "available": self.available,
            "description": self.description,
            "supported_properties": self.supported_properties,
        }


# Model registry
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    ModelName.STUB: ModelInfo(
        name=ModelName.STUB,
        version="1.0.0",
        available=True,
        description="Stub implementation with deterministic predictions based on structure hashing",
        supported_properties=["bandgap", "formation_energy", "stability_score"],
    ),
    ModelName.CGCNN: ModelInfo(
        name=ModelName.CGCNN,
        version="1.0.0",
        available=False,  # Not yet integrated
        description="Crystal Graph Convolutional Neural Networks",
        supported_properties=["bandgap", "formation_energy"],
    ),
    ModelName.MEGNET: ModelInfo(
        name=ModelName.MEGNET,
        version="2.0.0",
        available=False,  # Not yet integrated
        description="MatErials Graph Network - Universal property predictor",
        supported_properties=["bandgap", "formation_energy", "elastic_moduli"],
    ),
    ModelName.M3GNET: ModelInfo(
        name=ModelName.M3GNET,
        version="1.0.0",
        available=False,  # Not yet integrated
        description="Materials 3-body Graph Network - High accuracy energies and forces",
        supported_properties=["formation_energy", "forces", "stresses"],
    ),
    ModelName.ALIGNN: ModelInfo(
        name=ModelName.ALIGNN,
        version="2.0.0",
        available=False,  # Not yet integrated
        description="Atomistic Line Graph Neural Network - State-of-the-art predictions",
        supported_properties=["bandgap", "formation_energy", "elastic_moduli", "dielectric"],
    ),
}


def get_available_models() -> List[ModelInfo]:
    """
    Get list of available ML models.

    Returns:
        List of ModelInfo objects describing available models
    """
    return list(MODEL_REGISTRY.values())


def _deterministic_hash(structure_id: str, property_name: str) -> float:
    """
    Generate deterministic hash value for a structure and property.

    Uses SHA256 hashing to create reproducible "random" values based on
    structure ID. This ensures predictions are consistent across API calls.

    Args:
        structure_id: UUID of the structure
        property_name: Name of the property being predicted

    Returns:
        Float between 0 and 1
    """
    hash_input = f"{structure_id}:{property_name}"
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
    # Convert first 8 hex digits to float between 0 and 1
    int_value = int(hash_value[:8], 16)
    return int_value / (16**8)


def _predict_stub(
    structure_id: str,
    formula: Optional[str],
    num_atoms: Optional[int],
    dimensionality: Optional[int],
    composition: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """
    Stub implementation for property prediction.

    Generates realistic, deterministic predictions based on structure properties
    and deterministic hashing. The predictions follow realistic ranges:

    - Bandgap: 0-5 eV (higher for 2D materials, lower for metals)
    - Formation energy: -8 to -2 eV/atom (more negative = more stable)
    - Stability score: 0-1 (based on formation energy and heuristics)

    Args:
        structure_id: UUID of the structure as string
        formula: Chemical formula (e.g., "MoS2")
        num_atoms: Number of atoms in the structure
        dimensionality: 0=molecule, 1=1D, 2=2D, 3=bulk
        composition: Dictionary of element symbols to fractions

    Returns:
        Dictionary with predicted properties and confidence scores
    """
    logger.info(f"Generating stub predictions for structure {structure_id}")

    # Generate deterministic "random" values for this structure
    h_bandgap = _deterministic_hash(structure_id, "bandgap")
    h_energy = _deterministic_hash(structure_id, "formation_energy")
    h_stability = _deterministic_hash(structure_id, "stability")

    # Predict bandgap (0-5 eV)
    # 2D materials tend to have larger bandgaps
    base_bandgap = h_bandgap * 4.0  # 0-4 eV base
    if dimensionality == 2:
        bandgap = base_bandgap + 0.5  # Shift up for 2D
    elif dimensionality == 0:
        bandgap = base_bandgap + 1.0  # Molecules often have large gaps
    else:
        bandgap = base_bandgap

    # Clip to reasonable range
    bandgap = max(0.0, min(5.0, bandgap))

    # Predict formation energy (-8 to -2 eV/atom)
    # More negative = more stable
    formation_energy = -8.0 + (h_energy * 6.0)  # Range: -8 to -2

    # Stability score (0-1) based on formation energy
    # More negative formation energy = higher stability
    stability_base = (-formation_energy - 2.0) / 6.0  # Normalize to 0-1
    stability_score = stability_base * 0.7 + h_stability * 0.3  # Add some variation
    stability_score = max(0.0, min(1.0, stability_score))

    # Confidence scores (stub values, would come from model uncertainty in real implementation)
    # Higher confidence for more common dimensionalities
    confidence_base = 0.85 if dimensionality in [2, 3] else 0.75
    confidence_bandgap = confidence_base + _deterministic_hash(structure_id, "conf_bg") * 0.1
    confidence_energy = confidence_base + _deterministic_hash(structure_id, "conf_en") * 0.1
    confidence_stability = confidence_base + _deterministic_hash(structure_id, "conf_st") * 0.1

    return {
        "bandgap": round(bandgap, 3),
        "formation_energy": round(formation_energy, 3),
        "stability_score": round(stability_score, 3),
        "model_version": "1.0.0",
        "model_name": ModelName.STUB,
        "confidence": {
            "bandgap": round(confidence_bandgap, 3),
            "formation_energy": round(confidence_energy, 3),
            "stability_score": round(confidence_stability, 3),
        }
    }


def predict_properties_for_structure(
    structure: Any,
    model_name: str = ModelName.STUB,
) -> Dict[str, Any]:
    """
    Predict material properties using ML (stub for now).

    This is the main entry point for ML property prediction. Currently uses
    a stub implementation but is designed to support multiple ML models
    in the future.

    Args:
        structure: Structure model object (from src.api.models.structure)
        model_name: Name of the ML model to use (default: STUB)

    Returns:
        Dictionary with predicted properties:
        {
            "bandgap": float,  # eV
            "formation_energy": float,  # eV/atom
            "stability_score": float,  # 0-1
            "model_version": str,
            "model_name": str,
            "confidence": {
                "bandgap": float,  # 0-1
                "formation_energy": float,  # 0-1
                "stability_score": float,  # 0-1
            }
        }

    Raises:
        ValueError: If requested model is not available

    Example:
        >>> structure = db.get(Structure, structure_id)
        >>> predictions = predict_properties_for_structure(structure)
        >>> print(f"Predicted bandgap: {predictions['bandgap']} eV")
        Predicted bandgap: 2.341 eV
    """
    # Validate model availability
    if model_name not in MODEL_REGISTRY:
        available_models = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {available_models}"
        )

    model_info = MODEL_REGISTRY[model_name]
    if not model_info.available:
        raise ValueError(
            f"Model {model_name} is not currently available. "
            f"Only stub implementation is available in this version."
        )

    # Extract structure properties
    structure_id = str(structure.id)
    formula = structure.formula
    num_atoms = structure.num_atoms
    dimensionality = structure.dimensionality

    # Extract composition if available
    composition = None
    if structure.atoms and len(structure.atoms) > 0:
        # Build composition from atoms
        composition = {}
        for atom in structure.atoms:
            species = atom.get("species")
            if species:
                composition[species] = composition.get(species, 0) + 1
        # Normalize to fractions
        total = sum(composition.values())
        if total > 0:
            composition = {k: v/total for k, v in composition.items()}

    logger.info(
        f"Predicting properties for structure {structure_id} "
        f"(formula={formula}, atoms={num_atoms}, dim={dimensionality}) "
        f"using model {model_name}"
    )

    # Route to appropriate model
    if model_name == ModelName.STUB:
        predictions = _predict_stub(
            structure_id=structure_id,
            formula=formula,
            num_atoms=num_atoms,
            dimensionality=dimensionality,
            composition=composition,
        )
    elif model_name == ModelName.CGCNN:
        # Future: Load CGCNN model and run inference
        # predictions = _predict_cgcnn(structure)
        raise NotImplementedError(f"Model {model_name} not yet implemented")
    elif model_name == ModelName.MEGNET:
        # Future: Load MEGNet model and run inference
        # predictions = _predict_megnet(structure)
        raise NotImplementedError(f"Model {model_name} not yet implemented")
    elif model_name == ModelName.M3GNET:
        # Future: Load M3GNET model and run inference
        # predictions = _predict_m3gnet(structure)
        raise NotImplementedError(f"Model {model_name} not yet implemented")
    elif model_name == ModelName.ALIGNN:
        # Future: Load ALIGNN model and run inference
        # predictions = _predict_alignn(structure)
        raise NotImplementedError(f"Model {model_name} not yet implemented")
    else:
        raise ValueError(f"Model {model_name} is not supported")

    logger.info(
        f"Predictions generated: "
        f"bandgap={predictions['bandgap']} eV, "
        f"formation_energy={predictions['formation_energy']} eV/atom, "
        f"stability={predictions['stability_score']}"
    )

    return predictions


# Future integration examples:
"""
def _predict_cgcnn(structure: Any) -> Dict[str, Any]:
    '''Predict properties using CGCNN model.'''
    from cgcnn.model import CrystalGraphConvNet

    # Load pre-trained model
    model = CrystalGraphConvNet.load_checkpoint('models/cgcnn_best.pth')

    # Convert structure to CGCNN format
    cgcnn_input = convert_to_cgcnn_format(structure)

    # Run inference
    with torch.no_grad():
        predictions = model(cgcnn_input)

    return {
        'bandgap': predictions['band_gap'].item(),
        'formation_energy': predictions['formation_energy'].item(),
        'model_version': model.version,
        'model_name': 'CGCNN',
        'confidence': extract_uncertainty(predictions),
    }

def _predict_alignn(structure: Any) -> Dict[str, Any]:
    '''Predict properties using ALIGNN model.'''
    from alignn.models import ALIGNN
    from alignn.data import StructureDataset

    # Load pre-trained model
    model = ALIGNN.load_pretrained('models/alignn_bandgap.pth')

    # Convert structure to ALIGNN graph format
    graph = structure_to_alignn_graph(structure)

    # Run inference
    predictions = model.predict(graph)

    return {
        'bandgap': predictions['bandgap'],
        'formation_energy': predictions.get('formation_energy'),
        'model_version': model.version,
        'model_name': 'ALIGNN',
        'confidence': predictions.get('uncertainty', {}),
    }
"""

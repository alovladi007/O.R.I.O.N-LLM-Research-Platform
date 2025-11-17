"""
Pydantic schemas for ML property prediction API.

These schemas define the request and response formats for the ML prediction
endpoints, including:
- Property prediction requests
- Property prediction responses
- Model information
- Comparison results
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid


class PropertyPredictionRequest(BaseModel):
    """
    Request schema for ML property prediction.

    Used to request predictions for a specific structure, with options
    to force recomputation or use cached results.
    """
    structure_id: uuid.UUID = Field(
        ...,
        description="ID of the structure to predict properties for"
    )
    model_name: Optional[str] = Field(
        default="STUB",
        description="Name of ML model to use (STUB, CGCNN, ALIGNN, etc.)"
    )
    force_recompute: bool = Field(
        default=False,
        description="Force new prediction even if cached result exists"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "model_name": "STUB",
                "force_recompute": False
            }
        }


class PropertyPredictionResponse(BaseModel):
    """
    Response schema for ML property prediction.

    Returns the predicted properties along with confidence scores and
    metadata about the prediction.
    """
    id: uuid.UUID = Field(..., description="ID of the prediction record")
    structure_id: uuid.UUID = Field(..., description="ID of the structure")
    model_name: str = Field(..., description="Name of the ML model used")
    model_version: str = Field(..., description="Version of the ML model")
    properties: Dict[str, Any] = Field(
        ...,
        description="Predicted properties (bandgap, formation_energy, etc.)"
    )
    confidence_scores: Dict[str, float] = Field(
        ...,
        description="Confidence scores for each predicted property (0-1)"
    )
    cached: bool = Field(
        ...,
        description="Whether this result was retrieved from cache"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata about the prediction"
    )
    created_at: datetime = Field(..., description="When the prediction was made")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "223e4567-e89b-12d3-a456-426614174000",
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "model_name": "STUB",
                "model_version": "1.0.0",
                "properties": {
                    "bandgap": 2.341,
                    "formation_energy": -4.521,
                    "stability_score": 0.823
                },
                "confidence_scores": {
                    "bandgap": 0.89,
                    "formation_energy": 0.91,
                    "stability_score": 0.87
                },
                "cached": False,
                "metadata": {
                    "inference_time_ms": 123.45
                },
                "created_at": "2025-11-16T12:00:00Z"
            }
        }


class ModelInfoResponse(BaseModel):
    """
    Response schema for ML model information.

    Describes an available ML model, its capabilities, and availability status.
    """
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    available: bool = Field(..., description="Whether model is currently available")
    description: str = Field(..., description="Model description")
    supported_properties: List[str] = Field(
        ...,
        description="List of properties this model can predict"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "STUB",
                "version": "1.0.0",
                "available": True,
                "description": "Stub implementation with deterministic predictions",
                "supported_properties": ["bandgap", "formation_energy", "stability_score"]
            }
        }


class ModelsListResponse(BaseModel):
    """
    Response schema for listing available ML models.

    Returns a list of all registered ML models with their information.
    """
    models: List[ModelInfoResponse] = Field(
        ...,
        description="List of available ML models"
    )
    count: int = Field(..., description="Total number of models")

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "name": "STUB",
                        "version": "1.0.0",
                        "available": True,
                        "description": "Stub implementation",
                        "supported_properties": ["bandgap", "formation_energy"]
                    }
                ],
                "count": 1
            }
        }


class PropertyComparisonRequest(BaseModel):
    """
    Request schema for comparing ML predictions with simulation results.

    Used to compare predicted properties against actual simulation results
    to evaluate model accuracy.
    """
    predicted_properties_id: uuid.UUID = Field(
        ...,
        description="ID of the ML prediction to compare"
    )
    simulation_result_id: uuid.UUID = Field(
        ...,
        description="ID of the simulation result to compare against"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_properties_id": "223e4567-e89b-12d3-a456-426614174000",
                "simulation_result_id": "323e4567-e89b-12d3-a456-426614174000"
            }
        }


class PropertyComparisonResult(BaseModel):
    """
    Single property comparison result.

    Contains predicted value, simulated value, and error metrics.
    """
    predicted: Optional[float] = Field(None, description="ML predicted value")
    simulated: Optional[float] = Field(None, description="Simulation result value")
    error: Optional[float] = Field(None, description="Absolute error")
    percent_error: Optional[float] = Field(None, description="Percentage error")
    relative_error: Optional[float] = Field(None, description="Relative error")

    class Config:
        json_schema_extra = {
            "example": {
                "predicted": 2.341,
                "simulated": 2.450,
                "error": -0.109,
                "percent_error": -4.45,
                "relative_error": 0.045
            }
        }


class PropertyComparisonResponse(BaseModel):
    """
    Response schema for property comparison.

    Returns detailed comparison between ML predictions and simulation results
    for each property.
    """
    predicted_properties_id: uuid.UUID
    simulation_result_id: uuid.UUID
    structure_id: uuid.UUID
    model_name: str
    model_version: str
    comparisons: Dict[str, PropertyComparisonResult] = Field(
        ...,
        description="Comparison results for each property"
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics (mean error, RMSE, etc.)"
    )
    created_at: datetime = Field(
        ...,
        description="When the comparison was performed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_properties_id": "223e4567-e89b-12d3-a456-426614174000",
                "simulation_result_id": "323e4567-e89b-12d3-a456-426614174000",
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "model_name": "STUB",
                "model_version": "1.0.0",
                "comparisons": {
                    "bandgap": {
                        "predicted": 2.341,
                        "simulated": 2.450,
                        "error": -0.109,
                        "percent_error": -4.45,
                        "relative_error": 0.045
                    }
                },
                "summary": {
                    "mean_absolute_error": 0.109,
                    "mean_percent_error": 4.45,
                    "properties_compared": 1
                },
                "created_at": "2025-11-16T12:00:00Z"
            }
        }


class PredictionHistoryResponse(BaseModel):
    """
    Response schema for prediction history.

    Returns all predictions for a structure, grouped by model and version.
    """
    structure_id: uuid.UUID
    predictions: List[PropertyPredictionResponse] = Field(
        ...,
        description="List of all predictions for this structure"
    )
    count: int = Field(..., description="Total number of predictions")
    models_used: List[str] = Field(
        ...,
        description="List of unique models that made predictions"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "predictions": [
                    {
                        "id": "223e4567-e89b-12d3-a456-426614174000",
                        "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                        "model_name": "STUB",
                        "model_version": "1.0.0",
                        "properties": {"bandgap": 2.341},
                        "confidence_scores": {"bandgap": 0.89},
                        "cached": True,
                        "created_at": "2025-11-16T12:00:00Z"
                    }
                ],
                "count": 1,
                "models_used": ["STUB"]
            }
        }


class BatchPredictionRequest(BaseModel):
    """
    Request schema for batch property prediction.

    Allows predicting properties for multiple structures at once.
    """
    structure_ids: List[uuid.UUID] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of structure IDs to predict properties for"
    )
    model_name: Optional[str] = Field(
        default="STUB",
        description="Name of ML model to use"
    )
    force_recompute: bool = Field(
        default=False,
        description="Force new predictions even if cached results exist"
    )

    @validator('structure_ids')
    def validate_structure_ids(cls, v):
        """Ensure structure_ids list is not empty and not too large."""
        if len(v) == 0:
            raise ValueError("structure_ids must contain at least one structure ID")
        if len(v) > 100:
            raise ValueError("structure_ids cannot contain more than 100 IDs")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "structure_ids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "223e4567-e89b-12d3-a456-426614174000"
                ],
                "model_name": "STUB",
                "force_recompute": False
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch property prediction.

    Returns predictions for all requested structures.
    """
    predictions: List[PropertyPredictionResponse] = Field(
        ...,
        description="List of predictions for each structure"
    )
    total: int = Field(..., description="Total number of predictions")
    cached: int = Field(..., description="Number of cached results used")
    new: int = Field(..., description="Number of new predictions computed")
    errors: Optional[Dict[str, str]] = Field(
        default=None,
        description="Errors for structures that failed (structure_id -> error message)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [],
                "total": 2,
                "cached": 1,
                "new": 1,
                "errors": None
            }
        }


# ============================================================================
# Session 14: Feature Extraction Schemas
# ============================================================================

class FeatureComputeRequest(BaseModel):
    """Request schema for computing ML features for a structure."""
    cutoff_radius: float = Field(
        default=5.0,
        ge=0.1,
        le=20.0,
        description="Cutoff radius for neighbor search (Angstroms)"
    )
    force_recompute: bool = Field(
        default=False,
        description="Force recomputation even if cached features exist"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "cutoff_radius": 5.0,
                "force_recompute": False
            }
        }


class FeatureComputeResponse(BaseModel):
    """Response schema for feature computation."""
    id: uuid.UUID = Field(..., description="ID of the StructureFeatures record")
    structure_id: uuid.UUID = Field(..., description="ID of the structure")
    graph_repr: Dict[str, Any] = Field(..., description="Graph representation")
    scalar_features: Dict[str, Any] = Field(..., description="Scalar features")
    feature_version: str = Field(..., description="Version of feature extraction code")
    cached: bool = Field(..., description="Whether features were cached")
    created_at: datetime = Field(..., description="When features were computed")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "323e4567-e89b-12d3-a456-426614174000",
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "graph_repr": {
                    "num_atoms": 64,
                    "num_edges": 256
                },
                "scalar_features": {
                    "avg_electronegativity": 2.55,
                    "density": 2.33
                },
                "feature_version": "1.0.0",
                "cached": False,
                "created_at": "2025-11-16T12:00:00Z"
            }
        }


class BatchFeatureComputeRequest(BaseModel):
    """Request schema for batch feature computation."""
    structure_ids: List[uuid.UUID] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of structure IDs to compute features for"
    )
    cutoff_radius: float = Field(
        default=5.0,
        ge=0.1,
        le=20.0,
        description="Cutoff radius for neighbor search (Angstroms)"
    )
    force_recompute: bool = Field(
        default=False,
        description="Force recomputation even if cached features exist"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "structure_ids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "223e4567-e89b-12d3-a456-426614174000"
                ],
                "cutoff_radius": 5.0,
                "force_recompute": False
            }
        }


class BatchFeatureComputeResponse(BaseModel):
    """Response schema for batch feature computation."""
    features: List[FeatureComputeResponse] = Field(
        ...,
        description="List of computed features for each structure"
    )
    total: int = Field(..., description="Total number of structures processed")
    cached: int = Field(..., description="Number of cached results used")
    new: int = Field(..., description="Number of newly computed features")
    errors: Optional[Dict[str, str]] = Field(
        default=None,
        description="Errors for structures that failed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": [],
                "total": 2,
                "cached": 1,
                "new": 1,
                "errors": None
            }
        }


# ============================================================================
# Session 15: GNN-Specific Inference Schemas
# ============================================================================

class GNNPredictionRequest(BaseModel):
    """Request schema for GNN-based property prediction."""
    structure_id: uuid.UUID = Field(
        ...,
        description="ID of the structure to predict properties for"
    )
    gnn_model_name: str = Field(
        default="cgcnn_bandgap_v1",
        description="Name of GNN model (cgcnn_bandgap_v1, cgcnn_formation_energy_v1)"
    )
    use_cached_features: bool = Field(
        default=True,
        description="Use cached features if available"
    )
    cutoff_radius: float = Field(
        default=5.0,
        ge=0.1,
        le=20.0,
        description="Cutoff radius for graph construction (if features need to be computed)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "gnn_model_name": "cgcnn_bandgap_v1",
                "use_cached_features": True,
                "cutoff_radius": 5.0
            }
        }


class GNNPredictionResponse(BaseModel):
    """Response schema for GNN property prediction."""
    structure_id: uuid.UUID = Field(..., description="ID of the structure")
    gnn_model_name: str = Field(..., description="Name of GNN model used")
    target_property: str = Field(..., description="Target property predicted")
    prediction: float = Field(..., description="Predicted value")
    uncertainty: Optional[float] = Field(None, description="Prediction uncertainty")
    features_cached: bool = Field(..., description="Whether features were cached")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (inference_time_ms, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "gnn_model_name": "cgcnn_bandgap_v1",
                "target_property": "bandgap",
                "prediction": 2.45,
                "uncertainty": 0.12,
                "features_cached": True,
                "metadata": {
                    "inference_time_ms": 45.3
                }
            }
        }


# ============================================================================
# Session 16: Model Training & Registry Schemas
# ============================================================================

class TrainingRequest(BaseModel):
    """Request schema for starting a model training job."""
    target_property: str = Field(
        ...,
        description="Target property to train for (bandgap, formation_energy, etc.)"
    )
    model_type: str = Field(
        default="CGCNN",
        description="Model architecture type (CGCNN, ALIGNN, RandomForest)"
    )
    model_name: str = Field(
        ...,
        description="Name for the trained model (e.g., 'cgcnn_bandgap_v2')"
    )
    training_config: Dict[str, Any] = Field(
        ...,
        description="Training hyperparameters (epochs, batch_size, learning_rate, etc.)"
    )
    min_samples: int = Field(
        default=100,
        ge=10,
        description="Minimum number of samples required for training"
    )
    train_fraction: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="Fraction of data to use for training (vs validation)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "target_property": "bandgap",
                "model_type": "CGCNN",
                "model_name": "cgcnn_bandgap_v2",
                "training_config": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "hidden_dim": 128,
                    "num_layers": 3
                },
                "min_samples": 100,
                "train_fraction": 0.8
            }
        }


class TrainingResponse(BaseModel):
    """Response schema for training job submission."""
    job_id: str = Field(..., description="ID of the training job (for tracking)")
    status: str = Field(..., description="Job status (PENDING, RUNNING, COMPLETED, FAILED)")
    message: str = Field(..., description="Status message")
    model_name: str = Field(..., description="Name of model being trained")
    estimated_time_minutes: Optional[int] = Field(
        None,
        description="Estimated training time in minutes"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "train_123e4567",
                "status": "PENDING",
                "message": "Training job submitted successfully",
                "model_name": "cgcnn_bandgap_v2",
                "estimated_time_minutes": 30
            }
        }


class ModelRegistryResponse(BaseModel):
    """Response schema for model registry entry."""
    id: uuid.UUID = Field(..., description="ID of the model registry entry")
    name: str = Field(..., description="Unique model name")
    version: str = Field(..., description="Model version")
    target: str = Field(..., description="Target property")
    description: Optional[str] = Field(None, description="Model description")
    model_type: str = Field(..., description="Model architecture type")
    checkpoint_path: str = Field(..., description="Path to model checkpoint")
    training_config: Dict[str, Any] = Field(..., description="Training hyperparameters")
    metrics: Dict[str, Any] = Field(..., description="Training metrics (MSE, MAE, R²)")
    dataset_info: Optional[Dict[str, Any]] = Field(None, description="Dataset information")
    is_active: bool = Field(..., description="Whether model is active")
    is_system_provided: bool = Field(..., description="System vs user-trained")
    created_at: datetime = Field(..., description="When model was created")
    updated_at: datetime = Field(..., description="When model was last updated")

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "423e4567-e89b-12d3-a456-426614174000",
                "name": "cgcnn_bandgap_v2",
                "version": "2.0.0",
                "target": "bandgap",
                "description": "CGCNN model trained on 1000 structures",
                "model_type": "CGCNN",
                "checkpoint_path": "/models/cgcnn_bandgap_v2.pth",
                "training_config": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001
                },
                "metrics": {
                    "train_mse": 0.023,
                    "val_mse": 0.034,
                    "val_mae": 0.145,
                    "val_r2": 0.92
                },
                "dataset_info": {
                    "num_train": 800,
                    "num_val": 200
                },
                "is_active": True,
                "is_system_provided": False,
                "created_at": "2025-11-16T12:00:00Z",
                "updated_at": "2025-11-16T12:00:00Z"
            }
        }


# ============================================================================
# Session 20: Active Learning Schemas
# ============================================================================

class UncertaintyPredictionRequest(BaseModel):
    """
    Request schema for predictions with uncertainty estimation.

    Used for active learning to get both predictions and uncertainty estimates.
    """
    structure_id: uuid.UUID = Field(
        ...,
        description="ID of the structure to predict properties for"
    )
    gnn_model_name: str = Field(
        default="cgcnn_bandgap_v1",
        description="Name of GNN model to use"
    )
    uncertainty_method: str = Field(
        default="mc_dropout",
        description="Uncertainty estimation method (mc_dropout, ensemble, none)"
    )
    n_samples: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Number of forward passes for uncertainty estimation"
    )
    use_cached_features: bool = Field(
        default=True,
        description="Use cached features if available"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "gnn_model_name": "cgcnn_bandgap_v1",
                "uncertainty_method": "mc_dropout",
                "n_samples": 20,
                "use_cached_features": True
            }
        }


class UncertaintyPredictionResponse(BaseModel):
    """
    Response schema for predictions with uncertainty.

    Returns prediction, uncertainty estimate, and all sampled predictions.
    """
    structure_id: uuid.UUID = Field(..., description="ID of the structure")
    gnn_model_name: str = Field(..., description="Name of GNN model used")
    target_property: str = Field(..., description="Target property predicted")
    prediction: float = Field(..., description="Mean predicted value")
    uncertainty: float = Field(..., description="Prediction uncertainty (std)")
    predictions_sample: List[float] = Field(
        ...,
        description="All sampled predictions (for MC dropout/ensemble)"
    )
    method: str = Field(..., description="Uncertainty estimation method used")
    n_samples: int = Field(..., description="Number of samples used")
    features_cached: bool = Field(..., description="Whether features were cached")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "gnn_model_name": "cgcnn_bandgap_v1",
                "target_property": "bandgap",
                "prediction": 2.45,
                "uncertainty": 0.12,
                "predictions_sample": [2.42, 2.47, 2.43, 2.46, 2.44],
                "method": "mc_dropout",
                "n_samples": 20,
                "features_cached": True,
                "metadata": {
                    "inference_time_ms": 234.5
                }
            }
        }


class CandidateSelectionRequest(BaseModel):
    """
    Request schema for active learning candidate selection.

    Given a list of candidates with predictions, select which ones
    should be evaluated with expensive simulations.
    """
    candidate_structure_ids: List[uuid.UUID] = Field(
        ...,
        min_items=1,
        description="List of candidate structure IDs to evaluate"
    )
    target_property: str = Field(
        ...,
        description="Target property to optimize (bandgap, formation_energy, etc.)"
    )
    selection_strategy: str = Field(
        default="uncertainty",
        description="Selection strategy (uncertainty, greedy_uncertainty, expected_improvement)"
    )
    max_simulations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of candidates to select for simulation"
    )
    simulation_budget_fraction: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of candidates to simulate (0-1)"
    )
    uncertainty_threshold: float = Field(
        default=0.1,
        ge=0.0,
        description="Uncertainty threshold for automatic selection"
    )
    high_value_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Value threshold for high-priority candidates"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "candidate_structure_ids": [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "223e4567-e89b-12d3-a456-426614174000",
                    "323e4567-e89b-12d3-a456-426614174000"
                ],
                "target_property": "bandgap",
                "selection_strategy": "greedy_uncertainty",
                "max_simulations": 10,
                "simulation_budget_fraction": 0.2,
                "uncertainty_threshold": 0.1,
                "high_value_threshold": 0.8
            }
        }


class SelectedCandidate(BaseModel):
    """Schema for a candidate selected for simulation."""
    structure_id: uuid.UUID = Field(..., description="Structure ID")
    predicted_value: float = Field(..., description="ML predicted value")
    uncertainty: float = Field(..., description="Prediction uncertainty")
    acquisition_score: float = Field(..., description="Selection score")
    selection_reason: str = Field(
        ...,
        description="Reason for selection (high_uncertainty, high_value, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                "predicted_value": 2.45,
                "uncertainty": 0.15,
                "acquisition_score": 0.3675,
                "selection_reason": "high_uncertainty"
            }
        }


class CandidateSelectionResponse(BaseModel):
    """
    Response schema for active learning candidate selection.

    Returns selected candidates and classification of all candidates
    by confidence level.
    """
    selected_for_simulation: List[SelectedCandidate] = Field(
        ...,
        description="Candidates selected for expensive simulation"
    )
    trusted_predictions: List[uuid.UUID] = Field(
        ...,
        description="High-confidence candidates (trust ML prediction)"
    )
    total_candidates: int = Field(..., description="Total candidates evaluated")
    num_selected: int = Field(..., description="Number selected for simulation")
    num_trusted: int = Field(..., description="Number trusted (no simulation)")
    selection_strategy: str = Field(..., description="Strategy used")
    budget_info: Dict[str, Any] = Field(
        ...,
        description="Budget information (total, used, remaining)"
    )
    statistics: Dict[str, Any] = Field(
        ...,
        description="Selection statistics (mean uncertainty, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "selected_for_simulation": [
                    {
                        "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                        "predicted_value": 2.45,
                        "uncertainty": 0.15,
                        "acquisition_score": 0.3675,
                        "selection_reason": "high_uncertainty"
                    }
                ],
                "trusted_predictions": [
                    "223e4567-e89b-12d3-a456-426614174000"
                ],
                "total_candidates": 10,
                "num_selected": 3,
                "num_trusted": 7,
                "selection_strategy": "greedy_uncertainty",
                "budget_info": {
                    "total_budget": 10,
                    "used": 3,
                    "remaining": 7
                },
                "statistics": {
                    "mean_uncertainty": 0.08,
                    "max_uncertainty": 0.15,
                    "mean_predicted_value": 2.3
                }
            }
        }

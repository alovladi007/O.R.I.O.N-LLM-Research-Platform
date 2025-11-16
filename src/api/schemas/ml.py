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

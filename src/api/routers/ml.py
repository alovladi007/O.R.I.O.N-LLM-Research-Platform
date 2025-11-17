"""
ML property prediction router for NANO-OS API.

Provides endpoints for:
- Property prediction using ML models
- Listing available ML models
- Retrieving prediction history
- Batch predictions
- Comparing ML predictions with simulation results

The router implements a caching strategy to avoid redundant predictions:
- Checks for existing predictions before computing new ones
- Allows force_recompute to override cache
- Returns metadata indicating whether result is cached
"""

from fastapi import APIRouter, Depends, Query, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.orm import selectinload
from typing import List, Optional
from datetime import datetime
import logging
import uuid

from ..database import get_db
from ..models import User, Structure, PredictedProperties, StructureFeatures, MLModelRegistry
from ..models.provenance import EntityType, EventType
from ..schemas.ml import (
    PropertyPredictionRequest,
    PropertyPredictionResponse,
    ModelInfoResponse,
    ModelsListResponse,
    PredictionHistoryResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    # Session 14: Feature extraction
    FeatureComputeRequest,
    FeatureComputeResponse,
    BatchFeatureComputeRequest,
    BatchFeatureComputeResponse,
    # Session 15: GNN inference
    GNNPredictionRequest,
    GNNPredictionResponse,
    # Session 16: Training & registry
    TrainingRequest,
    TrainingResponse,
    ModelRegistryResponse,
)
from ..auth.security import get_current_active_user
from ..exceptions import NotFoundError, ValidationError
from backend.common.ml import (
    predict_properties_for_structure,
    get_available_models,
)
from backend.common.ml.features import extract_structure_features
from backend.common.ml.models.cgcnn_like import get_model, list_models
from backend.common.provenance import (
    record_provenance,
    get_system_info,
    get_code_version,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ml",
    tags=["machine-learning"],
    dependencies=[Depends(get_current_active_user)],
    responses={
        401: {"description": "Not authenticated"},
        404: {"description": "Resource not found"}
    }
)


@router.post(
    "/properties",
    response_model=PropertyPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict material properties",
    description="""
    Predict material properties using machine learning.

    This endpoint uses ML models to predict material properties such as:
    - Band gap (eV)
    - Formation energy (eV/atom)
    - Stability score (0-1)

    Caching Strategy:
    -----------------
    By default, if a prediction already exists for the given structure and model,
    the cached result is returned. Use `force_recompute=true` to compute a fresh
    prediction.

    Available Models:
    ----------------
    - STUB: Deterministic stub implementation (always available)
    - CGCNN, MEGNET, M3GNET, ALIGNN: Coming in future releases

    The response includes confidence scores for each predicted property.
    """,
    responses={
        200: {
            "description": "Properties predicted successfully",
            "content": {
                "application/json": {
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
                        "created_at": "2025-11-16T12:00:00Z"
                    }
                }
            }
        },
        404: {"description": "Structure not found"},
    }
)
async def predict_properties(
    request: PropertyPredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> PropertyPredictionResponse:
    """
    Predict material properties using ML.

    Steps:
    1. Check if structure exists
    2. Check if prediction already exists (unless force_recompute)
    3. If exists and not forcing, return cached result
    4. Otherwise, compute new prediction and save to database
    5. Return prediction with cache status
    """
    logger.info(
        f"Property prediction requested for structure {request.structure_id} "
        f"using model {request.model_name}, force_recompute={request.force_recompute}"
    )

    # Verify structure exists
    structure = await db.get(Structure, request.structure_id)
    if not structure:
        raise NotFoundError("Structure", request.structure_id)

    # Check for existing prediction (unless force_recompute)
    cached_prediction = None
    if not request.force_recompute:
        query = (
            select(PredictedProperties)
            .where(
                and_(
                    PredictedProperties.structure_id == request.structure_id,
                    PredictedProperties.model_name == request.model_name
                )
            )
            .order_by(desc(PredictedProperties.created_at))
            .limit(1)
        )
        result = await db.execute(query)
        cached_prediction = result.scalar_one_or_none()

    # If cached prediction exists and not forcing recompute, return it
    if cached_prediction:
        logger.info(f"Returning cached prediction {cached_prediction.id}")
        return PropertyPredictionResponse(
            id=cached_prediction.id,
            structure_id=cached_prediction.structure_id,
            model_name=cached_prediction.model_name,
            model_version=cached_prediction.model_version,
            properties=cached_prediction.properties,
            confidence_scores=cached_prediction.confidence_scores,
            cached=True,
            metadata=cached_prediction.metadata,
            created_at=cached_prediction.created_at,
        )

    # Compute new prediction
    logger.info(f"Computing new prediction for structure {request.structure_id}")
    try:
        prediction_result = predict_properties_for_structure(
            structure=structure,
            model_name=request.model_name
        )
    except ValueError as e:
        logger.error(f"Prediction failed: {e}")
        raise ValidationError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

    # Save prediction to database
    new_prediction = PredictedProperties(
        structure_id=request.structure_id,
        model_name=prediction_result["model_name"],
        model_version=prediction_result["model_version"],
        properties={
            "bandgap": prediction_result["bandgap"],
            "formation_energy": prediction_result["formation_energy"],
            "stability_score": prediction_result["stability_score"],
        },
        confidence_scores=prediction_result["confidence"],
        metadata={
            "formula": structure.formula,
            "num_atoms": structure.num_atoms,
            "dimensionality": structure.dimensionality,
        }
    )

    db.add(new_prediction)
    await db.commit()
    await db.refresh(new_prediction)

    logger.info(f"Prediction saved with ID {new_prediction.id}")

    # Record PREDICTED provenance event
    await record_provenance(
        db,
        EntityType.PREDICTION,
        new_prediction.id,
        EventType.PREDICTED,
        details={
            "model_name": new_prediction.model_name,
            "model_version": new_prediction.model_version,
            "structure_id": str(request.structure_id),
            "properties": new_prediction.properties,
            "confidence_scores": new_prediction.confidence_scores,
            "code_version": get_code_version(),
            "host_info": get_system_info(),
            "formula": structure.formula,
            "num_atoms": structure.num_atoms,
        }
    )

    return PropertyPredictionResponse(
        id=new_prediction.id,
        structure_id=new_prediction.structure_id,
        model_name=new_prediction.model_name,
        model_version=new_prediction.model_version,
        properties=new_prediction.properties,
        confidence_scores=new_prediction.confidence_scores,
        cached=False,
        metadata=new_prediction.metadata,
        created_at=new_prediction.created_at,
    )


@router.get(
    "/properties/{structure_id}",
    response_model=PropertyPredictionResponse,
    summary="Get latest prediction for structure",
    description="""
    Get the most recent ML property prediction for a structure.

    Returns the latest prediction made for the specified structure,
    regardless of which model was used. To get predictions from a
    specific model, use the `model_name` query parameter.

    If no predictions exist for the structure, returns 404.
    """,
    responses={
        200: {"description": "Latest prediction retrieved"},
        404: {"description": "No predictions found for structure"}
    }
)
async def get_latest_prediction(
    structure_id: uuid.UUID,
    model_name: Optional[str] = Query(
        None,
        description="Filter by specific model name"
    ),
    db: AsyncSession = Depends(get_db)
) -> PropertyPredictionResponse:
    """
    Get latest prediction for a structure.
    """
    logger.debug(
        f"Fetching latest prediction for structure {structure_id}, "
        f"model={model_name or 'any'}"
    )

    # Build query
    query = select(PredictedProperties).where(
        PredictedProperties.structure_id == structure_id
    )

    if model_name:
        query = query.where(PredictedProperties.model_name == model_name)

    query = query.order_by(desc(PredictedProperties.created_at)).limit(1)

    # Execute
    result = await db.execute(query)
    prediction = result.scalar_one_or_none()

    if not prediction:
        model_msg = f" from model {model_name}" if model_name else ""
        raise NotFoundError(
            "PredictedProperties",
            f"No predictions found for structure {structure_id}{model_msg}"
        )

    return PropertyPredictionResponse(
        id=prediction.id,
        structure_id=prediction.structure_id,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
        properties=prediction.properties,
        confidence_scores=prediction.confidence_scores,
        cached=True,  # Always true for GET requests
        metadata=prediction.metadata,
        created_at=prediction.created_at,
    )


@router.get(
    "/properties/{structure_id}/history",
    response_model=PredictionHistoryResponse,
    summary="Get prediction history for structure",
    description="""
    Get all property predictions for a structure.

    Returns all predictions ever made for the specified structure,
    including predictions from different models and versions.

    This is useful for:
    - Comparing predictions from different models
    - Tracking how predictions change over time
    - Analyzing model performance
    - Selecting the best prediction for downstream use

    Results are sorted by creation date (newest first).
    """,
    responses={
        200: {
            "description": "Prediction history retrieved",
            "content": {
                "application/json": {
                    "example": {
                        "structure_id": "123e4567-e89b-12d3-a456-426614174000",
                        "predictions": [],
                        "count": 2,
                        "models_used": ["STUB", "CGCNN"]
                    }
                }
            }
        }
    }
)
async def get_prediction_history(
    structure_id: uuid.UUID,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    db: AsyncSession = Depends(get_db)
) -> PredictionHistoryResponse:
    """
    Get all predictions for a structure.
    """
    logger.debug(f"Fetching prediction history for structure {structure_id}")

    # Query all predictions for structure
    query = (
        select(PredictedProperties)
        .where(PredictedProperties.structure_id == structure_id)
        .order_by(desc(PredictedProperties.created_at))
        .limit(limit)
    )

    result = await db.execute(query)
    predictions = result.scalars().all()

    # Build response
    prediction_responses = [
        PropertyPredictionResponse(
            id=p.id,
            structure_id=p.structure_id,
            model_name=p.model_name,
            model_version=p.model_version,
            properties=p.properties,
            confidence_scores=p.confidence_scores,
            cached=True,
            metadata=p.metadata,
            created_at=p.created_at,
        )
        for p in predictions
    ]

    # Get unique models used
    models_used = list(set(p.model_name for p in predictions))

    return PredictionHistoryResponse(
        structure_id=structure_id,
        predictions=prediction_responses,
        count=len(predictions),
        models_used=models_used,
    )


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List available ML models",
    description="""
    Get list of all registered ML models.

    Returns information about each model including:
    - Name and version
    - Availability status
    - Description
    - List of properties the model can predict

    Models marked as `available: true` can be used for predictions.
    Models marked as `available: false` are registered but not yet
    integrated (coming in future releases).
    """,
    responses={
        200: {
            "description": "List of ML models",
            "content": {
                "application/json": {
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
            }
        }
    }
)
async def list_models() -> ModelsListResponse:
    """
    Get list of available ML models.
    """
    logger.debug("Listing available ML models")

    models = get_available_models()

    model_responses = [
        ModelInfoResponse(
            name=model.name,
            version=model.version,
            available=model.available,
            description=model.description,
            supported_properties=model.supported_properties,
        )
        for model in models
    ]

    return ModelsListResponse(
        models=model_responses,
        count=len(model_responses),
    )


@router.post(
    "/properties/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch predict properties",
    description="""
    Predict properties for multiple structures in a single request.

    This endpoint is more efficient than making individual predictions
    when you need predictions for many structures.

    Features:
    - Processes up to 100 structures per request
    - Uses caching where possible (unless force_recompute=true)
    - Returns errors for individual structures without failing the entire batch
    - Provides summary statistics (total, cached, new)

    The response includes all successful predictions and a separate
    errors dictionary for any structures that failed.
    """,
    responses={
        200: {
            "description": "Batch predictions completed",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [],
                        "total": 2,
                        "cached": 1,
                        "new": 1,
                        "errors": None
                    }
                }
            }
        }
    }
)
async def batch_predict_properties(
    request: BatchPredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> BatchPredictionResponse:
    """
    Predict properties for multiple structures.
    """
    logger.info(
        f"Batch prediction requested for {len(request.structure_ids)} structures "
        f"using model {request.model_name}"
    )

    predictions: List[PropertyPredictionResponse] = []
    errors: dict = {}
    cached_count = 0
    new_count = 0

    for structure_id in request.structure_ids:
        try:
            # Create individual prediction request
            single_request = PropertyPredictionRequest(
                structure_id=structure_id,
                model_name=request.model_name,
                force_recompute=request.force_recompute,
            )

            # Get prediction (will use cache if available)
            prediction = await predict_properties(
                request=single_request,
                db=db,
                current_user=current_user,
            )

            predictions.append(prediction)

            if prediction.cached:
                cached_count += 1
            else:
                new_count += 1

        except NotFoundError as e:
            logger.warning(f"Structure {structure_id} not found: {e}")
            errors[str(structure_id)] = f"Structure not found"

        except Exception as e:
            logger.error(f"Error predicting for structure {structure_id}: {e}")
            errors[str(structure_id)] = str(e)

    logger.info(
        f"Batch prediction complete: {len(predictions)} successful, "
        f"{len(errors)} errors, {cached_count} cached, {new_count} new"
    )

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        cached=cached_count,
        new=new_count,
        errors=errors if errors else None,
    )


@router.delete(
    "/properties/{prediction_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete prediction",
    description="""
    Delete a specific property prediction.

    This permanently removes the prediction from the database.
    Use with caution - deleted predictions cannot be recovered.

    Typical use cases:
    - Removing incorrect predictions
    - Cleaning up old predictions from deprecated models
    - Forcing recomputation (delete then predict again)
    """,
    responses={
        204: {"description": "Prediction deleted successfully"},
        404: {"description": "Prediction not found"}
    }
)
async def delete_prediction(
    prediction_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> None:
    """
    Delete a prediction.
    """
    logger.info(f"Deleting prediction {prediction_id}")

    # Check permission - only admins can delete predictions
    if not current_user.is_admin:
        from ..exceptions import AuthorizationError
        raise AuthorizationError("Only admins can delete predictions")

    # Get prediction
    prediction = await db.get(PredictedProperties, prediction_id)
    if not prediction:
        raise NotFoundError("PredictedProperties", prediction_id)

    # Delete
    await db.delete(prediction)
    await db.commit()

    logger.info(f"Prediction {prediction_id} deleted")

    return None


# ============================================================================
# Session 14: Feature Extraction Endpoints
# ============================================================================

@router.post(
    "/features/structure/{structure_id}",
    response_model=FeatureComputeResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute ML features for a structure",
    description="""
    Compute graph and scalar features for a structure.

    This endpoint extracts ML-ready features from crystal structures:
    - **Graph features**: Neighbor lists, bond distances, atom properties
    - **Scalar features**: Composition, average properties, density

    Features are cached in the database and reused for subsequent requests
    unless `force_recompute=true`.

    The extracted features can be used for:
    - GNN model inference
    - Dataset building for model training
    - Structure analysis and visualization
    """
)
async def compute_structure_features(
    structure_id: uuid.UUID,
    request: FeatureComputeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> FeatureComputeResponse:
    """
    Compute and cache ML features for a structure.
    """
    logger.info(
        f"Feature computation requested for structure {structure_id}, "
        f"cutoff={request.cutoff_radius}, force_recompute={request.force_recompute}"
    )

    # Verify structure exists
    structure = await db.get(Structure, structure_id)
    if not structure:
        raise NotFoundError("Structure", structure_id)

    # Check for cached features
    cached_features = None
    if not request.force_recompute:
        query = select(StructureFeatures).where(
            StructureFeatures.structure_id == structure_id
        )
        result = await db.execute(query)
        cached_features = result.scalar_one_or_none()

    # Return cached if available
    if cached_features:
        logger.info(f"Returning cached features {cached_features.id}")
        return FeatureComputeResponse(
            id=cached_features.id,
            structure_id=cached_features.structure_id,
            graph_repr=cached_features.graph_repr,
            scalar_features=cached_features.scalar_features,
            feature_version=cached_features.feature_version,
            cached=True,
            created_at=cached_features.created_at,
        )

    # Compute new features
    logger.info(f"Computing new features for structure {structure_id}")
    try:
        graph_repr, scalar_features = extract_structure_features(
            structure,
            cutoff_radius=request.cutoff_radius
        )
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Feature extraction failed: {str(e)}"
        )

    # Save to database
    new_features = StructureFeatures(
        structure_id=structure_id,
        graph_repr=graph_repr,
        scalar_features=scalar_features,
        extraction_params={
            "cutoff_radius": request.cutoff_radius
        },
        feature_version="1.0.0"
    )

    db.add(new_features)
    await db.commit()
    await db.refresh(new_features)

    logger.info(f"Features saved with ID {new_features.id}")

    return FeatureComputeResponse(
        id=new_features.id,
        structure_id=new_features.structure_id,
        graph_repr=new_features.graph_repr,
        scalar_features=new_features.scalar_features,
        feature_version=new_features.feature_version,
        cached=False,
        created_at=new_features.created_at,
    )


@router.post(
    "/features/batch",
    response_model=BatchFeatureComputeResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch compute ML features",
    description="""
    Compute features for multiple structures in a single request.

    More efficient than individual requests when processing many structures.
    Features are cached and reused unless `force_recompute=true`.

    Returns successful features and errors separately.
    """
)
async def batch_compute_features(
    request: BatchFeatureComputeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> BatchFeatureComputeResponse:
    """
    Compute features for multiple structures.
    """
    logger.info(
        f"Batch feature computation for {len(request.structure_ids)} structures"
    )

    features_list: List[FeatureComputeResponse] = []
    errors: dict = {}
    cached_count = 0
    new_count = 0

    for structure_id in request.structure_ids:
        try:
            single_request = FeatureComputeRequest(
                cutoff_radius=request.cutoff_radius,
                force_recompute=request.force_recompute,
            )

            feature_response = await compute_structure_features(
                structure_id=structure_id,
                request=single_request,
                db=db,
                current_user=current_user,
            )

            features_list.append(feature_response)

            if feature_response.cached:
                cached_count += 1
            else:
                new_count += 1

        except NotFoundError as e:
            logger.warning(f"Structure {structure_id} not found: {e}")
            errors[str(structure_id)] = "Structure not found"

        except Exception as e:
            logger.error(f"Error computing features for {structure_id}: {e}")
            errors[str(structure_id)] = str(e)

    logger.info(
        f"Batch feature computation complete: {len(features_list)} successful, "
        f"{len(errors)} errors, {cached_count} cached, {new_count} new"
    )

    return BatchFeatureComputeResponse(
        features=features_list,
        total=len(features_list),
        cached=cached_count,
        new=new_count,
        errors=errors if errors else None,
    )


# ============================================================================
# Session 15: GNN Inference Endpoint
# ============================================================================

@router.post(
    "/gnn/properties",
    response_model=GNNPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict properties using GNN models",
    description="""
    Predict material properties using Graph Neural Network models.

    This endpoint uses the CGCNN-style GNN models from Session 15 to predict
    properties directly from crystal structure graphs.

    Available models:
    - `cgcnn_bandgap_v1`: Predicts bandgap (eV)
    - `cgcnn_formation_energy_v1`: Predicts formation energy (eV/atom)

    The endpoint:
    1. Extracts/retrieves features for the structure (cached if available)
    2. Loads the specified GNN model from registry
    3. Runs inference on the graph representation
    4. Returns prediction with uncertainty estimate
    """
)
async def predict_with_gnn(
    request: GNNPredictionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> GNNPredictionResponse:
    """
    Predict properties using GNN models.
    """
    logger.info(
        f"GNN prediction requested for structure {request.structure_id} "
        f"using model {request.gnn_model_name}"
    )

    # Verify structure exists
    structure = await db.get(Structure, request.structure_id)
    if not structure:
        raise NotFoundError("Structure", request.structure_id)

    # Get or compute features
    features_cached = False
    if request.use_cached_features:
        query = select(StructureFeatures).where(
            StructureFeatures.structure_id == request.structure_id
        )
        result = await db.execute(query)
        structure_features = result.scalar_one_or_none()

        if structure_features:
            graph_repr = structure_features.graph_repr
            features_cached = True
        else:
            # Features not cached, compute them
            graph_repr, _ = extract_structure_features(
                structure,
                cutoff_radius=request.cutoff_radius
            )
    else:
        # Force recompute features
        graph_repr, _ = extract_structure_features(
            structure,
            cutoff_radius=request.cutoff_radius
        )

    # Load GNN model
    try:
        gnn_model = get_model(request.gnn_model_name)
        if gnn_model is None:
            available_models = list_models()
            raise ValidationError(
                f"GNN model '{request.gnn_model_name}' not found. "
                f"Available models: {', '.join(available_models)}"
            )
    except Exception as e:
        logger.error(f"Failed to load GNN model: {e}")
        raise ValidationError(f"Failed to load GNN model: {str(e)}")

    # Run inference
    try:
        import time
        start_time = time.time()

        prediction_result = gnn_model.predict(graph_repr)

        inference_time_ms = (time.time() - start_time) * 1000

    except Exception as e:
        logger.error(f"GNN inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"GNN inference failed: {str(e)}"
        )

    logger.info(
        f"GNN prediction complete: {prediction_result['prediction']:.3f} "
        f"(uncertainty: {prediction_result.get('uncertainty', 0.0):.3f})"
    )

    return GNNPredictionResponse(
        structure_id=request.structure_id,
        gnn_model_name=request.gnn_model_name,
        target_property=gnn_model.target_property,
        prediction=prediction_result["prediction"],
        uncertainty=prediction_result.get("uncertainty"),
        features_cached=features_cached,
        metadata={
            "inference_time_ms": inference_time_ms,
            "model_info": prediction_result.get("model_info", {})
        }
    )


# ============================================================================
# Session 16: Model Registry & Training Endpoints
# ============================================================================

@router.get(
    "/models/{model_id}",
    response_model=ModelRegistryResponse,
    summary="Get model registry details",
    description="""
    Get detailed information about a specific trained model from the registry.

    Returns:
    - Model metadata (name, version, type)
    - Training configuration and hyperparameters
    - Performance metrics (MSE, MAE, R²)
    - Dataset information
    - Checkpoint path
    - Active status
    """
)
async def get_model_registry_entry(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> ModelRegistryResponse:
    """
    Get model registry entry by ID.
    """
    logger.debug(f"Fetching model registry entry {model_id}")

    model_entry = await db.get(MLModelRegistry, model_id)
    if not model_entry:
        raise NotFoundError("MLModelRegistry", model_id)

    return ModelRegistryResponse(
        id=model_entry.id,
        name=model_entry.name,
        version=model_entry.version,
        target=model_entry.target,
        description=model_entry.description,
        model_type=model_entry.model_type,
        checkpoint_path=model_entry.checkpoint_path,
        training_config=model_entry.training_config,
        metrics=model_entry.metrics,
        dataset_info=model_entry.dataset_info,
        is_active=model_entry.is_active,
        is_system_provided=model_entry.is_system_provided,
        created_at=model_entry.created_at,
        updated_at=model_entry.updated_at,
    )


@router.post(
    "/train",
    response_model=TrainingResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start model training job",
    description="""
    Submit a model training job for background processing.

    This endpoint queues a training job that will:
    1. Build a dataset from StructureFeatures + SimulationResults
    2. Split into train/validation sets
    3. Train the specified model architecture
    4. Save checkpoint and register in MLModelRegistry

    **Status**: Currently returns a stub response. Full training pipeline
    implementation is pending (would be implemented in worker tasks).

    The training job runs asynchronously in the background using Celery.
    Use the returned `job_id` to track progress.
    """
)
async def start_training_job(
    request: TrainingRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> TrainingResponse:
    """
    Submit a model training job.

    NOTE: This is currently a stub implementation. Full training pipeline
    would be implemented as a Celery worker task.
    """
    logger.info(
        f"Training job requested: {request.model_name} for {request.target_property}"
    )

    # Check if model name already exists
    query = select(MLModelRegistry).where(
        MLModelRegistry.name == request.model_name
    )
    result = await db.execute(query)
    existing_model = result.scalar_one_or_none()

    if existing_model and not current_user.is_admin:
        from ..exceptions import AuthorizationError
        raise AuthorizationError(
            f"Model name '{request.model_name}' already exists. "
            "Choose a different name or contact an admin."
        )

    # TODO: Implement actual training job submission
    # This would typically:
    # 1. Submit Celery task: train_cgcnn_model.delay(...)
    # 2. Return job ID for tracking
    # 3. Worker task handles training, checkpointing, and registry update

    # For now, return stub response
    job_id = f"train_{uuid.uuid4().hex[:12]}"

    logger.warning(
        "Training job submission is currently stubbed. "
        "Full implementation requires worker task integration."
    )

    return TrainingResponse(
        job_id=job_id,
        status="PENDING",
        message=(
            "Training job submitted successfully. "
            "NOTE: Full training pipeline not yet implemented - this is a stub response."
        ),
        model_name=request.model_name,
        estimated_time_minutes=None,
    )

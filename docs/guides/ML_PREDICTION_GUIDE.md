# NANO-OS ML Property Prediction Guide

## Overview

The NANO-OS ML property prediction system provides machine learning-based predictions for material properties. This system is designed to quickly estimate properties like band gap, formation energy, and stability without running expensive quantum mechanical simulations.

**Current Status**: Session 6 implementation complete with stub ML models. Real ML models (CGCNN, MEGNet, M3GNET, ALIGNN) will be integrated in future sessions.

## Table of Contents

1. [Architecture](#architecture)
2. [Current Implementation (Stub Models)](#current-implementation-stub-models)
3. [API Endpoints](#api-endpoints)
4. [Database Schema](#database-schema)
5. [Caching Strategy](#caching-strategy)
6. [Future ML Model Integration](#future-ml-model-integration)
7. [Usage Examples](#usage-examples)
8. [Comparison with Simulations](#comparison-with-simulations)

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
│                                                               │
│  ┌───────────────┐         ┌──────────────────┐            │
│  │  ML Router    │────────▶│ PredictedProperties│           │
│  │ /api/v1/ml/*  │         │     (Database)     │           │
│  └───────┬───────┘         └──────────────────┘            │
│          │                                                   │
│          ▼                                                   │
│  ┌────────────────────────────────────────────┐            │
│  │  backend.common.ml.properties              │            │
│  │                                             │            │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │            │
│  │  │   STUB   │  │  CGCNN   │  │ ALIGNN   │ │  (Future)  │
│  │  │  Model   │  │  Model   │  │  Model   │ │            │
│  │  └──────────┘  └──────────┘  └──────────┘ │            │
│  └────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Module Structure

```
backend/common/ml/
├── __init__.py           # Module exports
├── properties.py         # ML prediction logic
└── comparison.py         # ML vs simulation comparison

src/api/
├── models/
│   └── predicted_properties.py    # Database model
├── schemas/
│   └── ml.py                      # Pydantic schemas
└── routers/
    └── ml.py                      # API endpoints

alembic/versions/
└── 002_add_predicted_properties.py    # Database migration
```

---

## Current Implementation (Stub Models)

### How the Stub Works

The current implementation uses a **deterministic stub** that generates realistic predictions without actual ML models:

1. **Deterministic Hashing**: Uses SHA256 hash of structure ID to generate reproducible "random" values
2. **Realistic Ranges**: Predictions follow physically meaningful ranges:
   - **Band gap**: 0-5 eV (higher for 2D materials, lower for bulk)
   - **Formation energy**: -8 to -2 eV/atom (more negative = more stable)
   - **Stability score**: 0-1 (based on formation energy and heuristics)
3. **Confidence Scores**: Generates confidence values (0.75-0.95) based on structure properties

### Why Use a Stub?

The stub implementation allows us to:
- ✅ Develop and test the complete API infrastructure
- ✅ Implement caching and database storage
- ✅ Test comparison utilities
- ✅ Provide consistent, reproducible results for testing
- ✅ Demonstrate the API before real models are integrated

### Stub Prediction Algorithm

```python
def _predict_stub(structure_id, formula, num_atoms, dimensionality, composition):
    # Generate deterministic hash-based values
    h_bandgap = hash(structure_id + "bandgap")
    h_energy = hash(structure_id + "energy")

    # Calculate bandgap (0-5 eV)
    bandgap = h_bandgap * 4.0
    if dimensionality == 2:
        bandgap += 0.5  # 2D materials have larger gaps

    # Calculate formation energy (-8 to -2 eV/atom)
    formation_energy = -8.0 + (h_energy * 6.0)

    # Calculate stability score (0-1)
    stability_score = (-formation_energy - 2.0) / 6.0

    return {
        "bandgap": bandgap,
        "formation_energy": formation_energy,
        "stability_score": stability_score,
        "confidence": {...}
    }
```

---

## API Endpoints

### 1. POST /api/v1/ml/properties

Predict material properties using ML.

**Request:**
```json
{
  "structure_id": "123e4567-e89b-12d3-a456-426614174000",
  "model_name": "STUB",
  "force_recompute": false
}
```

**Response:**
```json
{
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
  "cached": false,
  "metadata": {
    "formula": "MoS2",
    "num_atoms": 3,
    "dimensionality": 2
  },
  "created_at": "2025-11-16T12:00:00Z"
}
```

**Caching Behavior:**
- If prediction exists for structure + model, returns cached result
- Set `force_recompute: true` to compute fresh prediction
- Response includes `cached` field indicating cache status

---

### 2. GET /api/v1/ml/properties/{structure_id}

Get the latest prediction for a structure.

**Query Parameters:**
- `model_name` (optional): Filter by specific model

**Response:**
Same as POST /properties, with `cached: true`

---

### 3. GET /api/v1/ml/properties/{structure_id}/history

Get all predictions for a structure.

**Response:**
```json
{
  "structure_id": "123e4567-e89b-12d3-a456-426614174000",
  "predictions": [
    {
      "id": "223e4567-e89b-12d3-a456-426614174000",
      "model_name": "STUB",
      "model_version": "1.0.0",
      "properties": {...},
      "created_at": "2025-11-16T12:00:00Z"
    }
  ],
  "count": 1,
  "models_used": ["STUB"]
}
```

**Use Cases:**
- Compare predictions from different models
- Track how predictions change over time
- Select best prediction for downstream use

---

### 4. GET /api/v1/ml/models

List available ML models.

**Response:**
```json
{
  "models": [
    {
      "name": "STUB",
      "version": "1.0.0",
      "available": true,
      "description": "Stub implementation with deterministic predictions",
      "supported_properties": [
        "bandgap",
        "formation_energy",
        "stability_score"
      ]
    },
    {
      "name": "CGCNN",
      "version": "1.0.0",
      "available": false,
      "description": "Crystal Graph Convolutional Neural Networks",
      "supported_properties": [
        "bandgap",
        "formation_energy"
      ]
    }
  ],
  "count": 2
}
```

**Notes:**
- `available: true` = model can be used now
- `available: false` = model registered but not yet integrated

---

### 5. POST /api/v1/ml/properties/batch

Batch predict properties for multiple structures.

**Request:**
```json
{
  "structure_ids": [
    "123e4567-e89b-12d3-a456-426614174000",
    "223e4567-e89b-12d3-a456-426614174000"
  ],
  "model_name": "STUB",
  "force_recompute": false
}
```

**Response:**
```json
{
  "predictions": [...],
  "total": 2,
  "cached": 1,
  "new": 1,
  "errors": null
}
```

**Features:**
- Process up to 100 structures per request
- Partial failure support (some structures can fail without failing entire batch)
- Returns summary statistics

---

### 6. DELETE /api/v1/ml/properties/{prediction_id}

Delete a specific prediction (admin only).

**Response:** 204 No Content

---

## Database Schema

### predicted_properties Table

```sql
CREATE TABLE predicted_properties (
    id UUID PRIMARY KEY,
    structure_id UUID REFERENCES structures(id) ON DELETE CASCADE,
    model_name VARCHAR(50) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    properties JSON NOT NULL,              -- {bandgap: 2.5, ...}
    confidence_scores JSON NOT NULL,       -- {bandgap: 0.95, ...}
    metadata JSON,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Indexes
    INDEX idx_structure_id (structure_id),
    INDEX idx_model_name (model_name),
    INDEX idx_created_at (created_at),
    INDEX idx_structure_model (structure_id, model_name, created_at),
    INDEX idx_model_version (model_name, model_version, created_at)
);
```

### Relationships

```
Material (1) ──── (N) Structure (1) ──── (N) PredictedProperties
                              │
                              └──── (N) SimulationJob (1) ──── (1) SimulationResult
```

---

## Caching Strategy

### How Caching Works

1. **Cache Key**: `(structure_id, model_name)`
2. **Cache Lookup**: Before computing new prediction, check if one exists
3. **Cache Hit**: Return existing prediction with `cached: true`
4. **Cache Miss**: Compute new prediction, save to DB, return with `cached: false`
5. **Force Recompute**: Set `force_recompute: true` to bypass cache

### Cache Invalidation

Predictions are **never automatically invalidated**. Reasons:
- ML models are versioned (model_version field)
- Old predictions remain for comparison
- Users can explicitly force recomputation

To invalidate cache:
```bash
# Option 1: Force recompute
POST /api/v1/ml/properties
{
  "structure_id": "...",
  "force_recompute": true
}

# Option 2: Delete old prediction (admin only)
DELETE /api/v1/ml/properties/{prediction_id}
```

### Performance Benefits

- **Average case**: O(1) database lookup (indexed query)
- **First prediction**: O(1) ML inference + O(1) database insert
- **Subsequent predictions**: O(1) database lookup (no inference)

**Benchmark (estimated):**
- Cached prediction: ~10-50ms
- New prediction (stub): ~50-100ms
- New prediction (real ML): ~500-2000ms (depending on model)

---

## Future ML Model Integration

### Supported Models (Coming Soon)

#### 1. CGCNN (Crystal Graph Convolutional Neural Networks)

**What it does:**
- Graph neural network for crystal structures
- Good for band gap and formation energy prediction

**Integration plan:**
```python
# backend/common/ml/properties.py

def _predict_cgcnn(structure: Structure) -> Dict[str, Any]:
    from cgcnn.model import CrystalGraphConvNet

    # Load pre-trained model
    model = CrystalGraphConvNet.load_checkpoint('models/cgcnn_best.pth')

    # Convert structure to CGCNN graph format
    graph = structure_to_cgcnn_graph(structure)

    # Run inference
    with torch.no_grad():
        predictions = model(graph)

    return {
        'bandgap': predictions['band_gap'].item(),
        'formation_energy': predictions['formation_energy'].item(),
        'model_version': model.version,
        'model_name': 'CGCNN',
        'confidence': extract_uncertainty(predictions),
    }
```

#### 2. MEGNet (MatErials Graph Network)

**What it does:**
- Universal graph neural network
- Predicts multiple properties simultaneously
- Uses 3D structure information

**Properties supported:**
- Band gap
- Formation energy
- Elastic moduli
- Bulk modulus
- Shear modulus

#### 3. M3GNET (Materials 3-body Graph Network)

**What it does:**
- Advanced architecture with 3-body interactions
- High accuracy for energies and forces
- Suitable for geometry optimization

**Properties supported:**
- Formation energy
- Atomic forces
- Stress tensors

#### 4. ALIGNN (Atomistic Line Graph Neural Network)

**What it does:**
- State-of-the-art accuracy
- Uses line graphs for better geometric representation
- Pre-trained on large datasets

**Properties supported:**
- Band gap
- Formation energy
- Elastic moduli
- Dielectric constants
- Refractive index
- And more...

### Model Registry Pattern

All models are registered in `MODEL_REGISTRY`:

```python
MODEL_REGISTRY = {
    "STUB": ModelInfo(
        name="STUB",
        version="1.0.0",
        available=True,
        ...
    ),
    "CGCNN": ModelInfo(
        name="CGCNN",
        version="1.0.0",
        available=False,  # Not yet integrated
        ...
    ),
    ...
}
```

To add a new model:
1. Add entry to `MODEL_REGISTRY`
2. Implement `_predict_{model_name}()` function
3. Add routing in `predict_properties_for_structure()`
4. Set `available=True` when ready

---

## Usage Examples

### Example 1: Basic Prediction

```python
import httpx

# Predict properties for a structure
response = httpx.post(
    "http://localhost:8000/api/v1/ml/properties",
    json={
        "structure_id": "123e4567-e89b-12d3-a456-426614174000",
        "model_name": "STUB"
    },
    headers={"Authorization": f"Bearer {token}"}
)

prediction = response.json()
print(f"Predicted band gap: {prediction['properties']['bandgap']} eV")
print(f"Confidence: {prediction['confidence_scores']['bandgap']}")
```

### Example 2: Check Cache Before Predicting

```python
# First, check if prediction exists
response = httpx.get(
    f"http://localhost:8000/api/v1/ml/properties/{structure_id}",
    params={"model_name": "STUB"},
    headers={"Authorization": f"Bearer {token}"}
)

if response.status_code == 200:
    # Use cached prediction
    prediction = response.json()
    print("Using cached prediction")
else:
    # Compute new prediction
    response = httpx.post(
        "http://localhost:8000/api/v1/ml/properties",
        json={"structure_id": structure_id, "model_name": "STUB"},
        headers={"Authorization": f"Bearer {token}"}
    )
    prediction = response.json()
    print("Computed new prediction")
```

### Example 3: Batch Prediction

```python
# Predict for multiple structures
structure_ids = [
    "123e4567-e89b-12d3-a456-426614174000",
    "223e4567-e89b-12d3-a456-426614174000",
    "323e4567-e89b-12d3-a456-426614174000",
]

response = httpx.post(
    "http://localhost:8000/api/v1/ml/properties/batch",
    json={
        "structure_ids": structure_ids,
        "model_name": "STUB"
    },
    headers={"Authorization": f"Bearer {token}"}
)

result = response.json()
print(f"Total: {result['total']}")
print(f"Cached: {result['cached']}")
print(f"New: {result['new']}")

for pred in result['predictions']:
    print(f"Structure {pred['structure_id']}: "
          f"Bandgap = {pred['properties']['bandgap']} eV")
```

### Example 4: Compare Models

```python
# Get predictions from different models (future)
models = ["STUB", "CGCNN", "ALIGNN"]
predictions = {}

for model in models:
    response = httpx.post(
        "http://localhost:8000/api/v1/ml/properties",
        json={
            "structure_id": structure_id,
            "model_name": model
        },
        headers={"Authorization": f"Bearer {token}"}
    )
    if response.status_code == 200:
        predictions[model] = response.json()

# Compare band gap predictions
for model, pred in predictions.items():
    bandgap = pred['properties']['bandgap']
    confidence = pred['confidence_scores']['bandgap']
    print(f"{model}: {bandgap:.3f} eV (confidence: {confidence:.3f})")
```

---

## Comparison with Simulations

### Using the Comparison Utility

```python
from backend.common.ml import compare_ml_vs_simulation
from src.api.models import PredictedProperties, SimulationResult

# Get prediction and simulation result
predicted = db.get(PredictedProperties, prediction_id)
simulated = db.get(SimulationResult, simulation_id)

# Compare
comparison = compare_ml_vs_simulation(predicted, simulated)

# Print results
for prop, comp in comparison['comparisons'].items():
    print(f"\n{prop}:")
    print(f"  Predicted: {comp['predicted']}")
    print(f"  Simulated: {comp['simulated']}")
    print(f"  Error: {comp['error']}")
    print(f"  Percent Error: {comp['percent_error']}%")

# Summary statistics
summary = comparison['summary']
print(f"\nMean Absolute Error: {summary['mean_absolute_error']}")
print(f"RMSE: {summary['root_mean_squared_error']}")
```

### Evaluate Prediction Quality

```python
from backend.common.ml.comparison import evaluate_prediction_quality

# Evaluate quality
quality = evaluate_prediction_quality(comparison)

print(f"Overall Quality: {quality['overall_quality']}")  # excellent/good/fair/poor

for prop, qual in quality['property_quality'].items():
    print(f"{prop}: {qual['quality']} (error: {qual['error']})")

# Print recommendations
for rec in quality['recommendations']:
    print(f"- {rec}")
```

### Quality Thresholds

| Property          | Excellent | Good   | Fair   | Poor    |
|-------------------|-----------|--------|--------|---------|
| Band gap          | < 0.1 eV  | < 0.3  | < 0.5  | >= 0.5  |
| Formation energy  | < 0.1 eV/atom | < 0.3 | < 0.5 | >= 0.5 |
| Stability score   | < 0.05    | < 0.1  | < 0.2  | >= 0.2  |

---

## Best Practices

### 1. Use Caching Wisely

✅ **Do:**
- Use cached predictions for production workflows
- Only force recompute when model version changes

❌ **Don't:**
- Force recompute on every request (wastes resources)
- Ignore the `cached` field in responses

### 2. Handle Prediction Uncertainty

✅ **Do:**
- Check confidence scores before using predictions
- Use simulation for low-confidence predictions
- Compare multiple models when available

❌ **Don't:**
- Blindly trust all predictions
- Ignore confidence scores

### 3. Model Selection

✅ **Do:**
- Use STUB for testing and development
- Use specialized models for specific properties (when available)
- Consider ensemble predictions for critical decisions

❌ **Don't:**
- Use STUB for production decisions
- Expect all models to support all properties

### 4. Performance Optimization

✅ **Do:**
- Use batch predictions for multiple structures
- Enable caching for repeated queries
- Monitor prediction latency

❌ **Don't:**
- Make individual requests in a loop (use batch endpoint)
- Disable caching without good reason

---

## Troubleshooting

### Issue: Predictions seem random

**Cause:** You're using the STUB model

**Solution:**
- STUB predictions are deterministic but hash-based
- Wait for real ML models (CGCNN, ALIGNN, etc.)
- Use simulations for critical properties

### Issue: Low confidence scores

**Cause:** Structure may be unusual or outside training data

**Solution:**
- Run a simulation for verification
- Check if structure is realistic
- Try multiple models (when available)

### Issue: Cached predictions are outdated

**Cause:** Model version was updated but cache wasn't invalidated

**Solution:**
```python
# Force recompute
response = httpx.post(
    "/api/v1/ml/properties",
    json={
        "structure_id": structure_id,
        "force_recompute": True
    }
)
```

### Issue: Batch predictions are slow

**Cause:** Too many structures or all require new predictions

**Solution:**
- Reduce batch size (max 100 per request)
- Pre-compute predictions for common structures
- Use caching effectively

---

## Migration Path

### Session 6 (Current): Stub Implementation
- ✅ Database schema
- ✅ API endpoints
- ✅ Caching strategy
- ✅ Comparison utilities
- ✅ Deterministic stub predictions

### Session 7 (Future): Real ML Models
- 🔄 Integrate CGCNN for band gap prediction
- 🔄 Add model checkpoints and loading
- 🔄 Implement uncertainty quantification
- 🔄 Benchmark against DFT results

### Session 8 (Future): Advanced Features
- 🔄 Ensemble predictions
- 🔄 Active learning integration
- 🔄 Model fine-tuning capabilities
- 🔄 Explainable AI features

---

## References

### Research Papers

1. **CGCNN**: Xie & Grossman (2018), "Crystal Graph Convolutional Neural Networks"
2. **MEGNet**: Chen et al. (2019), "Graph Networks as a Universal Machine Learning Framework"
3. **M3GNET**: Chen et al. (2022), "A Universal Graph Deep Learning Interatomic Potential"
4. **ALIGNN**: Choudhary & DeCost (2021), "Atomistic Line Graph Neural Network"

### Useful Links

- CGCNN: https://github.com/txie-93/cgcnn
- MEGNet: https://github.com/materialsvirtuallab/megnet
- M3GNET: https://github.com/materialsvirtuallab/m3gnet
- ALIGNN: https://github.com/usnistgov/alignn

---

## Support

For questions or issues:
1. Check this guide first
2. Review the API documentation at `/docs`
3. Check the code examples above
4. Contact the NANO-OS development team

---

**Document Version**: 1.0
**Last Updated**: 2025-11-16
**Session**: 6

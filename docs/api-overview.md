# NANO-OS API Overview

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: `https://api.nano-os.your-domain.com`

All API endpoints are prefixed with `/api`.

## Authentication

Most endpoints require JWT authentication (Sessions 10-12):

```bash
# Login
POST /api/auth/login
{
  "username": "user@example.com",
  "password": "password"
}

# Returns
{
  "access_token": "eyJ...",
  "token_type": "bearer"
}

# Use in subsequent requests
Authorization: Bearer eyJ...
```

## Core Endpoints

### Health Check

```http
GET /api/health
```

Returns system health status including database connectivity.

**Response**:
```json
{
  "status": "healthy",
  "database": "connected",
  "version": "1.0.0"
}
```

---

## Materials

### List Materials

```http
GET /api/materials
```

**Query Parameters**:
- `skip` (int): Pagination offset (default: 0)
- `limit` (int): Number of results (default: 50, max: 100)
- `search` (str): Search in name/formula

**Response**:
```json
[
  {
    "id": "uuid",
    "name": "Molybdenum Disulfide",
    "formula": "MoS2",
    "description": "2D semiconductor",
    "tags": ["2D", "TMD", "semiconductor"],
    "created_at": "2025-01-15T10:30:00Z",
    "structure_count": 3
  }
]
```

### Get Material

```http
GET /api/materials/{id}
```

**Response**: Single material with related structures.

### Create Material

```http
POST /api/materials
```

**Request**:
```json
{
  "name": "Graphene",
  "formula": "C",
  "description": "Single-layer carbon",
  "tags": ["2D", "carbon", "conductor"]
}
```

**Response**: Created material object.

### Update Material

```http
PUT /api/materials/{id}
```

**Request**: Same as create (partial updates supported).

### Delete Material

```http
DELETE /api/materials/{id}
```

---

## Structures

### List Structures

```http
GET /api/structures
```

**Query Parameters**:
- `material_id` (UUID): Filter by material
- `dimensionality` (int): Filter by 0D/1D/2D/3D
- `skip`, `limit`: Pagination

**Response**:
```json
[
  {
    "id": "uuid",
    "material_id": "uuid",
    "name": "MoS2 monolayer",
    "formula": "Mo2S4",
    "num_atoms": 6,
    "dimensionality": 2,
    "format": "CIF",
    "source": "USER_UPLOADED",
    "created_at": "2025-01-15T10:30:00Z"
  }
]
```

### Get Structure

```http
GET /api/structures/{id}
```

**Response**:
```json
{
  "id": "uuid",
  "material_id": "uuid",
  "name": "MoS2 monolayer",
  "formula": "Mo2S4",
  "num_atoms": 6,
  "dimensionality": 2,
  "format": "CIF",
  "source": "USER_UPLOADED",
  "lattice_vectors": [[3.16, 0, 0], [-1.58, 2.74, 0], [0, 0, 20.0]],
  "atoms": [
    {"species": "Mo", "position": [0, 0, 0], "fractional": [0, 0, 0]},
    {"species": "S", "position": [1.58, 0.91, 1.58], "fractional": [0.333, 0.333, 0.079]}
  ],
  "raw_text": "# CIF file content...",
  "created_at": "2025-01-15T10:30:00Z"
}
```

### Upload Structure

```http
POST /api/structures
```

**Request**:
```json
{
  "material_id": "uuid",
  "name": "Optimized MoS2",
  "file_content": "# CIF or POSCAR content as string",
  "format": "CIF"
}
```

**Response**: Created structure with parsed atomic data.

### Parse Structure File

```http
POST /api/structures/parse
```

**Request**:
```json
{
  "file_content": "# CIF or POSCAR content",
  "format": "CIF"
}
```

**Response**:
```json
{
  "formula": "Mo2S4",
  "num_atoms": 6,
  "lattice_vectors": [...],
  "atoms": [...],
  "dimensionality": 2
}
```

Useful for client-side preview before creating structure.

### Get Structure Details for Viewer

```http
GET /api/structures/{id}/detail
```

**Response**: Optimized format for 3D visualization (3Dmol.js).

---

## Workflow Templates

### List Templates

```http
GET /api/workflows/templates
```

**Response**:
```json
[
  {
    "id": "uuid",
    "name": "DFT_RELAX_QE",
    "engine": "qe",
    "description": "Quantum Espresso geometry relaxation",
    "default_params": {
      "ecutwfc": 50,
      "ecutrho": 400,
      "conv_thr": 1e-8
    },
    "is_system": true
  }
]
```

### Get Template

```http
GET /api/workflows/templates/{id}
```

---

## Simulation Jobs

### List Jobs

```http
GET /api/jobs
```

**Query Parameters**:
- `status` (str): Filter by PENDING/QUEUED/RUNNING/COMPLETED/FAILED
- `structure_id` (UUID): Filter by structure
- `skip`, `limit`: Pagination

**Response**:
```json
[
  {
    "id": "uuid",
    "structure_id": "uuid",
    "workflow_template_id": "uuid",
    "name": "DFT relaxation of MoS2",
    "status": "COMPLETED",
    "priority": 5,
    "submitted_at": "2025-01-15T10:00:00Z",
    "started_at": "2025-01-15T10:01:00Z",
    "completed_at": "2025-01-15T10:15:00Z",
    "result": {
      "success": true,
      "final_energy": -123.456,
      "convergence_reached": true
    }
  }
]
```

### Get Job

```http
GET /api/jobs/{id}
```

**Response**: Full job details including parameters and results.

### Create Job

```http
POST /api/jobs
```

**Request**:
```json
{
  "structure_id": "uuid",
  "workflow_template_id": "uuid",
  "name": "My DFT calculation",
  "parameters": {
    "ecutwfc": 60,
    "kpoints": [6, 6, 1]
  },
  "priority": 5
}
```

**Response**: Created job (status=PENDING, queued for execution).

### Get Job Status

```http
GET /api/jobs/{id}/status
```

**Response**:
```json
{
  "job_id": "uuid",
  "status": "RUNNING",
  "progress": 0.45,
  "message": "SCF iteration 15/30"
}
```

Suitable for polling (future: WebSocket).

### Cancel Job

```http
POST /api/jobs/{id}/cancel
```

### Get Job Result

```http
GET /api/jobs/{id}/result
```

**Response**:
```json
{
  "id": "uuid",
  "job_id": "uuid",
  "engine_name": "qe",
  "success": true,
  "summary": {
    "final_energy": -123.456,
    "forces": [[0.001, 0.002, 0.000], ...],
    "stress": [0.1, 0.1, 0.1, 0, 0, 0],
    "convergence_reached": true,
    "scf_iterations": 25
  },
  "artifacts_path": "/data/jobs/uuid/output",
  "created_at": "2025-01-15T10:15:00Z"
}
```

---

## ML & GNN (Sessions 14-16)

### Predict Properties (Stub or GNN)

```http
POST /api/ml/properties
```

**Request**:
```json
{
  "structure_id": "uuid",
  "model_name": "STUB",  // or "cgcnn_bandgap_v1"
  "use_gnn": false
}
```

**Response**:
```json
{
  "bandgap": 2.341,
  "formation_energy": -4.567,
  "stability_score": 0.834,
  "model_name": "STUB",
  "model_version": "1.0.0",
  "confidence": {
    "bandgap": 0.89,
    "formation_energy": 0.91,
    "stability_score": 0.85
  }
}
```

### Compute Structure Features

```http
POST /api/ml/features/structure/{id}
```

Triggers background task to compute GNN features for a structure.

**Response**:
```json
{
  "status": "queued",
  "task_id": "celery-task-id"
}
```

### Batch Compute Features

```http
POST /api/ml/features/batch
```

**Request**:
```json
{
  "structure_ids": ["uuid1", "uuid2", "uuid3"]
}
```

### GNN Inference

```http
POST /api/ml/gnn/properties
```

**Request**:
```json
{
  "structure_id": "uuid",
  "model_name": "cgcnn_bandgap_v1"
}
```

**Response**: Similar to `/ml/properties` but uses GNN model.

### List Available Models

```http
GET /api/ml/models
```

**Response**:
```json
[
  {
    "id": "uuid",
    "name": "cgcnn_bandgap_v1",
    "version": "1.0.0",
    "target": "bandgap",
    "metrics": {
      "mse": 0.123,
      "mae": 0.089,
      "r2": 0.956
    },
    "is_system_provided": true,
    "created_at": "2025-01-10T00:00:00Z"
  }
]
```

### Get Model

```http
GET /api/ml/models/{id}
```

### Train New Model

```http
POST /api/ml/train
```

**Request**:
```json
{
  "name": "my_bandgap_model",
  "target": "bandgap",
  "config": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "hidden_dim": 128
  }
}
```

**Response**:
```json
{
  "status": "training_queued",
  "model_id": "uuid",
  "task_id": "celery-task-id"
}
```

Training runs in background. Poll `/api/ml/models/{id}` for status.

---

## Design Search

### Launch Design Search

```http
POST /api/design/search
```

**Request**:
```json
{
  "base_structure_id": "uuid",
  "objective": "maximize_bandgap",
  "constraints": {
    "min_stability": 0.7,
    "max_atoms": 20
  },
  "num_candidates": 10,
  "algorithm": "genetic"
}
```

**Response**:
```json
{
  "search_id": "uuid",
  "candidates": [
    {
      "structure": {...},
      "predicted_bandgap": 3.45,
      "predicted_stability": 0.89,
      "rank": 1
    },
    ...
  ]
}
```

### Get Search Results

```http
GET /api/design/search/{id}
```

---

## Design Campaigns

### List Campaigns

```http
GET /api/campaigns
```

**Response**:
```json
[
  {
    "id": "uuid",
    "name": "High bandgap 2D materials",
    "description": "Search for 2D TMDs with bandgap > 2 eV",
    "objective": "maximize_bandgap",
    "status": "ACTIVE",
    "iterations_completed": 3,
    "max_iterations": 10,
    "best_candidates": [
      {
        "structure_id": "uuid",
        "bandgap": 3.2,
        "iteration": 2
      }
    ],
    "created_at": "2025-01-10T00:00:00Z"
  }
]
```

### Create Campaign

```http
POST /api/campaigns
```

**Request**:
```json
{
  "name": "Wide bandgap search",
  "description": "Find materials with bandgap > 3 eV",
  "search_space": {
    "elements": ["Mo", "W", "S", "Se"],
    "dimensionality": 2
  },
  "objective": "maximize_bandgap",
  "config": {
    "algorithm": "bayesian_optimization",
    "batch_size": 20,
    "use_ml": true,
    "dft_refinement": false
  },
  "max_iterations": 10
}
```

**Response**: Created campaign (status=ACTIVE).

### Get Campaign

```http
GET /api/campaigns/{id}
```

**Response**: Full campaign details with iteration history.

### Pause/Resume Campaign

```http
POST /api/campaigns/{id}/pause
POST /api/campaigns/{id}/resume
```

### Get Campaign Statistics

```http
GET /api/campaigns/{id}/stats
```

**Response**:
```json
{
  "campaign_id": "uuid",
  "total_candidates_evaluated": 153,
  "best_bandgap": 3.45,
  "iterations_completed": 8,
  "convergence_trend": [2.1, 2.5, 2.8, 3.0, 3.2, 3.3, 3.4, 3.45],
  "estimated_time_remaining": "2 hours"
}
```

---

## Provenance & Multi-Scale (Future)

### Get Provenance Trail

```http
GET /api/provenance/structure/{id}
```

**Response**: Full history of how a structure was generated (parent structures, simulations, design steps).

### Multi-Scale Workflow

```http
POST /api/multiscale/workflow
```

**Request**:
```json
{
  "name": "DFT to MD workflow",
  "steps": [
    {
      "type": "dft_relaxation",
      "engine": "qe",
      "structure_id": "uuid"
    },
    {
      "type": "md_equilibration",
      "engine": "lammps",
      "use_relaxed_structure": true,
      "parameters": {
        "temperature": 300,
        "timesteps": 100000
      }
    }
  ]
}
```

---

## Error Responses

All endpoints return standard HTTP status codes:

- `200 OK`: Success
- `201 Created`: Resource created
- `400 Bad Request`: Invalid input (with details in `detail` field)
- `401 Unauthorized`: Missing or invalid auth token
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error (logged for debugging)

**Error Format**:
```json
{
  "detail": "Validation error: field 'formula' is required"
}
```

---

## Rate Limiting

- Default: 100 requests/minute per user
- ML inference: 10 requests/minute (computationally expensive)
- Training: 1 request/5 minutes

Headers returned:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 73
X-RateLimit-Reset: 1642251600
```

---

## Pagination

List endpoints support pagination:

**Request**:
```http
GET /api/materials?skip=50&limit=50
```

**Response Headers**:
```
X-Total-Count: 237
Link: </api/materials?skip=100&limit=50>; rel="next"
```

---

## API Versioning

Current version: `v1` (implicit in `/api` prefix)

Future versions will use: `/api/v2/...`

---

## OpenAPI Documentation

Interactive API docs:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

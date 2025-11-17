# NANO-OS Architecture

## Overview

NANO-OS (Nanomaterials Operating System) is a comprehensive platform for computational materials research, combining multi-scale simulations, machine learning, and automated design workflows.

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                    │
│  ┌──────────┬──────────┬──────────┬──────────┬─────────────┐│
│  │ Materials│Structures│Simulation│  ML/GNN  │ Design      ││
│  │ Browser  │ Viewer   │ Dashboard│Predictions│ Campaigns   ││
│  └──────────┴──────────┴──────────┴──────────┴─────────────┘│
└────────────────────────┬─────────────────────────────────────┘
                         │ REST API / WebSocket
┌────────────────────────┴─────────────────────────────────────┐
│                    Backend API (FastAPI)                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Routes: /materials, /structures, /jobs, /ml, /design │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Common Modules                                        │  │
│  │  - engines/    : Simulation engine abstractions       │  │
│  │  - ml/         : ML models & training                  │  │
│  │  - design/     : Design search algorithms             │  │
│  │  - campaigns/  : Multi-iteration design loops         │  │
│  │  - structures/ : Structure parsing & manipulation     │  │
│  │  - provenance/ : Reproducibility tracking             │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────┐
│                   Database Layer (PostgreSQL)                 │
│  ┌──────────┬──────────┬──────────┬──────────┬────────────┐ │
│  │Materials │Structures│Simulation│   ML     │  Design    │ │
│  │          │          │Jobs/     │ Models & │ Campaigns  │ │
│  │          │          │Results   │ Features │            │ │
│  └──────────┴──────────┴──────────┴──────────┴────────────┘ │
└──────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────┐
│                   Worker Service (Celery)                     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Background Tasks:                                     │  │
│  │  - run_simulation_job()                                │  │
│  │  - compute_structure_features()                        │  │
│  │  - run_training_job()                                  │  │
│  │  - run_design_iteration()                              │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────┐
│                   Simulation Engines                          │
│  ┌──────────┬──────────┬──────────┬──────────┬────────────┐ │
│  │  Mock    │ Quantum  │ Molecular│Continuum │ Mesoscale  │ │
│  │ (Testing)│Espresso  │ Dynamics │   FEM    │   KMC      │ │
│  │          │  (DFT)   │(LAMMPS)  │          │            │ │
│  └──────────┴──────────┴──────────┴──────────┴────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Core Services

### 1. Frontend Service

**Technology**: Next.js 14 (React), TypeScript, TailwindCSS

**Pages**:
- `/` - Dashboard with platform overview
- `/materials` - Materials browser and creator
- `/structures/:id` - Structure viewer (3D visualization via 3Dmol.js)
- `/jobs` - Simulation job dashboard
- `/ml/predictions` - ML property predictions
- `/ml/models` - ML model management and training
- `/design/search` - Material design search
- `/campaigns/:id` - Design campaign monitoring

**State Management**: React Query for server state, Context API for UI state

### 2. Backend API Service

**Technology**: FastAPI, Python 3.10+, SQLAlchemy (async ORM)

**Key Routes**:
- `GET/POST /api/materials` - Material CRUD operations
- `GET/POST /api/structures` - Structure upload and retrieval
- `POST /api/structures/parse` - Parse CIF/POSCAR files
- `GET/POST /api/jobs` - Simulation job management
- `GET /api/jobs/{id}/status` - Real-time job status
- `POST /api/ml/properties` - ML property predictions
- `POST /api/ml/features/structure/{id}` - Compute GNN features
- `POST /api/ml/train` - Train new ML models
- `GET /api/ml/models` - List available models
- `POST /api/design/search` - Launch design search
- `GET/POST /api/campaigns` - Design campaign management

**Authentication**: JWT-based authentication (Sessions 10-12)

**Multi-tenancy**: All records linked to `owner_id` for user isolation

### 3. Worker Service

**Technology**: Celery with Redis broker

**Background Tasks**:

1. **`run_simulation_job(job_id: UUID)`**
   - Fetch job details from DB
   - Route to appropriate engine (Mock, QE, LAMMPS, etc.)
   - Execute simulation
   - Parse results and update DB
   - Update job status to COMPLETED/FAILED

2. **`compute_structure_features(structure_id: UUID)`**
   - Load structure from DB
   - Compute graph representation (neighbor lists, bond distances)
   - Compute scalar features (composition, electronegativity, etc.)
   - Store in `StructureFeatures` table

3. **`run_training_job(config: TrainingConfig)`**
   - Build dataset from StructureFeatures + SimulationResults
   - Train GNN model (PyTorch)
   - Save model checkpoint
   - Register in MLModelRegistry
   - Return metrics

4. **`run_design_iteration(campaign_id: UUID, iteration: int)`**
   - Fetch campaign config
   - Generate candidate structures
   - Run ML predictions or simulations
   - Update best candidates
   - Decide if campaign should continue

### 4. Database

**Technology**: PostgreSQL 14+ with pgvector extension

**Core Data Models**:

#### Materials
```python
class Material(Base):
    id: UUID
    owner_id: UUID  # Multi-tenancy
    name: str
    formula: str
    description: str
    tags: List[str]  # JSON array
    metadata: Dict   # JSON for extensibility
    created_at: datetime
    updated_at: datetime
```

#### Structures
```python
class Structure(Base):
    id: UUID
    owner_id: UUID
    material_id: UUID (FK)
    name: str
    format: StructureFormat (CIF, POSCAR, XYZ, INTERNAL)
    source: StructureSource (USER_UPLOADED, GENERATED, OPTIMIZED, etc.)
    raw_text: str  # Original file content
    lattice_vectors: List[List[float]]  # 3x3 matrix
    atoms: List[Dict]  # [{species, position, ...}]
    num_atoms: int
    dimensionality: int  # 0=molecule, 1=1D, 2=2D, 3=bulk
    formula: str
    created_at: datetime
```

#### Simulation Jobs & Results
```python
class SimulationJob(Base):
    id: UUID
    owner_id: UUID
    structure_id: UUID (FK)
    workflow_template_id: UUID (FK)
    status: JobStatus (PENDING, QUEUED, RUNNING, COMPLETED, FAILED)
    priority: int
    parameters: Dict  # JSON for engine-specific params
    celery_task_id: str
    submitted_at: datetime
    started_at: datetime
    completed_at: datetime

class SimulationResult(Base):
    id: UUID
    job_id: UUID (FK)
    engine_name: str
    success: bool
    summary: Dict  # JSON with energies, forces, etc.
    artifacts_path: str  # Path to output files
    error_message: str
    created_at: datetime
```

#### Workflow Templates
```python
class WorkflowTemplate(Base):
    id: UUID
    name: str
    engine: str (mock, qe, lammps, continuum, mesoscale)
    default_params: Dict  # JSON with engine-specific defaults
    description: str
    is_system: bool  # System-provided vs user-created
```

#### ML Models & Features (Sessions 14-16)
```python
class StructureFeatures(Base):
    id: UUID
    structure_id: UUID (FK, unique)
    graph_repr: Dict  # JSON: neighbor lists, bond distances
    scalar_features: Dict  # JSON: composition, descriptors
    created_at: datetime

class MLModelRegistry(Base):
    id: UUID
    name: str
    version: str
    target: str  # "bandgap", "formation_energy", etc.
    path: str  # Checkpoint file path
    training_config: Dict  # JSON with hyperparameters
    metrics: Dict  # JSON: MSE, MAE, R²
    is_system_provided: bool
    created_at: datetime
```

#### Design Campaigns
```python
class DesignCampaign(Base):
    id: UUID
    owner_id: UUID
    name: str
    description: str
    search_space: Dict  # JSON defining candidate space
    objective: str  # "maximize bandgap", etc.
    config: Dict  # JSON with algorithm parameters
    status: str (ACTIVE, PAUSED, COMPLETED)
    best_candidates: List[Dict]  # Top N candidates
    iterations_completed: int
    max_iterations: int
    created_at: datetime
    updated_at: datetime
```

## Data Flow Examples

### 1. Structure Upload → ML Prediction

```
User uploads CIF → POST /api/structures
  ↓
Backend parses CIF → creates Structure in DB
  ↓
User requests prediction → POST /api/ml/properties
  ↓
Background task: compute_structure_features(structure_id)
  ↓
Features stored in StructureFeatures
  ↓
Load GNN model from MLModelRegistry
  ↓
Run inference → return predictions
```

### 2. DFT Simulation Flow

```
User selects structure + workflow → POST /api/jobs
  ↓
Create SimulationJob (status=PENDING)
  ↓
Enqueue Celery task: run_simulation_job(job_id)
  ↓
Worker picks up task
  ↓
Update status → RUNNING
  ↓
Engine executes (e.g., Quantum Espresso PWscf)
  ↓
Parse output files
  ↓
Create SimulationResult
  ↓
Update job status → COMPLETED
  ↓
Frontend polls /api/jobs/{id}/status → displays results
```

### 3. Design Campaign Loop

```
User creates campaign → POST /api/campaigns
  ↓
For each iteration:
  ↓
  run_design_iteration(campaign_id, iteration)
    ↓
    Generate N candidate structures (design/search.py)
    ↓
    For each candidate:
      - Compute ML predictions OR run simulations
    ↓
    Rank candidates by objective
    ↓
    Update campaign.best_candidates
    ↓
    If max_iterations reached → status=COMPLETED
```

## Multi-Scale Integration (Future AGI Loops)

NANO-OS is architected to support multi-scale workflows where different engines operate at different length/time scales:

### Scale Hierarchy

1. **Quantum Scale** (Å, fs) - DFT via Quantum Espresso
2. **Atomistic Scale** (nm, ps) - MD via LAMMPS
3. **Mesoscale** (μm, ms) - KMC, Phase Field
4. **Continuum Scale** (mm+, s+) - FEM, CFD

### AGI Loop Architecture

The design campaign system is designed to be **AGI-compatible**:

```python
# AGI Agent Interface
class AGIDesignAgent:
    async def propose_candidates(self, campaign_id: UUID) -> List[Structure]:
        """AGI proposes new structures based on campaign history."""
        pass

    async def analyze_results(self, results: List[SimulationResult]) -> Insights:
        """AGI analyzes simulation outputs and generates insights."""
        pass

    async def update_strategy(self, campaign: DesignCampaign) -> StrategyUpdate:
        """AGI updates search strategy based on outcomes."""
        pass
```

**Integration Points**:
- AGI can read all data via REST API
- AGI can submit structures and jobs programmatically
- AGI can trigger multi-scale workflows (e.g., DFT refinement of ML predictions)
- Provenance system tracks AGI decisions for reproducibility

## Extensibility

### Adding a New Engine

1. Create `backend/common/engines/my_engine.py`
2. Inherit from `SimulationEngine` base class
3. Implement `prepare_input()`, `execute()`, `parse_output()`
4. Register in `backend/common/engines/registry.py`
5. Add workflow templates in DB seed script

### Adding a New ML Model

1. Implement model in `backend/common/ml/models/my_model.py`
2. Follow PyTorch `nn.Module` interface
3. Implement `load_from_checkpoint()` and `predict()` methods
4. Register in `MLModelRegistry` via DB or API
5. Model automatically available in `/api/ml/properties`

### Adding a New Design Algorithm

1. Implement in `backend/common/design/my_algorithm.py`
2. Follow `DesignAlgorithm` interface
3. Implement `generate_candidates()` method
4. Integrate with campaign loop in `backend/common/campaigns/loop.py`

## Security

- **Authentication**: JWT tokens (Sessions 10-12)
- **Authorization**: Row-level security via `owner_id` checks
- **Input Validation**: Pydantic schemas for all API inputs
- **SQL Injection**: Prevented by SQLAlchemy ORM
- **File Upload**: Validation of structure file formats
- **Rate Limiting**: Configurable per endpoint

## Monitoring & Observability

- **Health Checks**: `/api/health` endpoint
- **Logging**: Structured JSON logs (stdout for Docker)
- **Metrics**: Database pool stats, job queue length
- **Future**: Prometheus metrics, Grafana dashboards

## Deployment

### Development
```bash
docker-compose up
```

### Production
- Use `docker-compose.prod.yml` with:
  - Gunicorn for backend (multi-worker)
  - Nginx reverse proxy for frontend
  - Redis for Celery broker and caching
  - Persistent volumes for DB and models

### Kubernetes (Session 16+)
- Helm charts in `/k8s` directory
- Separate deployments for API, worker, frontend
- Horizontal pod autoscaling for workers

## Performance Considerations

- **Async I/O**: FastAPI + async SQLAlchemy for high concurrency
- **Connection Pooling**: Configured in `database.py`
- **Background Jobs**: Long-running simulations offloaded to Celery
- **ML Inference**: Cached in Redis (future), batched when possible
- **Database Indexes**: On `owner_id`, `status`, foreign keys

## Future Enhancements

1. **Real-time Updates**: WebSocket for job status streaming
2. **Collaborative Workflows**: Share campaigns between users
3. **Model Marketplace**: Share trained ML models
4. **HPC Integration**: Submit jobs to SLURM clusters
5. **Federated Learning**: Train models across institutions
6. **Knowledge Graph**: Neo4j for materials relationships
7. **AGI Integration**: LangChain/LlamaIndex for agentic design

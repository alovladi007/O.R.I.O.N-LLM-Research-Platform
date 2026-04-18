# NANO-OS Sessions 18-20: Advanced ML & Hybrid Workflows

**Implementation Date**: 2025-11-17
**Status**: ✅ **CORE MODULES COMPLETE** (API integration pending)

---

## Overview

This document describes the implementation of Sessions 18-20, which add:
1. **Session 18**: ML Interatomic Potentials & Hybrid DFT-ML Workflows
2. **Session 19**: Bayesian Optimization for Materials Design
3. **Session 20**: Active Learning for Smart Simulation Selection

---

## Session 18: ML Interatomic Potentials & Hybrid DFT-ML Workflows ✅

### Deliverables

#### 1. MLPotential Model (`src/api/models/ml_potential.py`)

Database model for tracking ML-based force fields trained from DFT data:

**Fields**:
- `id`: UUID primary key
- `name`: Unique potential name (e.g., "snap_silicon_v1")
- `version`: Version string
- `descriptor_type`: Architecture type (SNAP, SOAP, NequIP, MACE, ACE, GAP)
- `training_data_source`: List of simulation job IDs used for training
- `training_dataset_info`: JSON with dataset statistics
- `path`: Path to potential files directory
- `files`: JSON dict of potential file paths (coefficients, weights, etc.)
- `training_config`: JSON with hyperparameters
- `metrics`: JSON with validation metrics (energy RMSE, force MAE)
- `elements`: Array of chemical elements covered
- `temperature_range`: Valid temperature range
- `pressure_range`: Valid pressure range
- `is_active`: Whether potential is active for use
- `is_validated`: Whether validated on test set
- `created_at`, `updated_at`: Timestamps

**Example**:
```python
potential = MLPotential(
    name="snap_silicon_v1",
    version="1.0.0",
    descriptor_type="SNAP",
    training_data_source=["job_uuid_1", "job_uuid_2"],
    elements=["Si"],
    metrics={
        "train_energy_rmse": 0.005,  # eV/atom
        "val_energy_rmse": 0.008,
        "train_force_mae": 0.05,  # eV/Å
        "val_force_mae": 0.07
    }
)
```

#### 2. ML Potential Training Module (`backend/common/ml/potentials.py`)

Comprehensive module for extracting training data and training ML potentials.

**Key Data Structures**:

```python
@dataclass
class TrainingSnapshot:
    """Single training configuration with DFT energies/forces."""
    structure_id: str
    atoms: List[Dict[str, Any]]
    lattice_vectors: List[List[float]]
    energy: float  # Total energy (eV)
    forces: List[List[float]]  # Atomic forces (eV/Å)
    stress: Optional[List[List[float]]]  # Stress tensor
    metadata: Optional[Dict[str, Any]]

@dataclass
class TrainingDataset:
    """Complete training dataset."""
    snapshots: List[TrainingSnapshot]
    elements: List[str]
    num_snapshots: int
    energy_range: Tuple[float, float]
    force_max: float
    metadata: Dict[str, Any]

@dataclass
class PotentialConfig:
    """ML potential training configuration."""
    name: str
    descriptor_type: str  # "SNAP", "NequIP", etc.
    elements: List[str]
    descriptor_params: Dict[str, Any]
    train_fraction: float = 0.8
    learning_rate: float = 0.001
    num_epochs: int = 100
    energy_weight: float = 1.0
    force_weight: float = 10.0
```

**Main Functions**:

```python
def extract_snapshots_from_simulation(
    simulation_result: Dict[str, Any],
    include_forces: bool = True
) -> List[TrainingSnapshot]:
    """Extract training snapshots from DFT simulation result."""

def build_training_dataset(
    simulation_results: List[Dict[str, Any]],
    min_force_threshold: Optional[float] = None,
    max_force_threshold: Optional[float] = None
) -> TrainingDataset:
    """Build complete training dataset from multiple simulations."""

def export_dataset_to_file(
    dataset: TrainingDataset,
    output_path: Path,
    format: str = "extxyz"  # or "json", "lammps_data"
) -> Path:
    """Export dataset to file format (ExtXYZ, JSON, LAMMPS)."""

def train_ml_potential(
    config: PotentialConfig,
    dataset: TrainingDataset,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Train ML interatomic potential.

    NOTE: Currently stub implementation. Full version would call:
    - FitSNAP for SNAP potentials
    - NequIP trainer for NequIP
    - MACE trainer for MACE
    - etc.
    """
```

**Supported Formats**:
- **ExtXYZ**: Extended XYZ format (compatible with most ML potential trainers)
- **JSON**: For inspection and debugging
- **LAMMPS Data**: LAMMPS data file format

**Stub Training Implementation**:

The `train_ml_potential()` function currently provides stub implementations for:
- **SNAP**: Creates dummy `.snapcoeff` and `.snapparam` files
- **NequIP/MACE**: Creates dummy `.pth` model file and config
- **Generic**: Creates generic potential file

In production, these would call actual trainers:
- SNAP: FitSNAP or LAMMPS built-in SNAP trainer
- NequIP: NequIP training script
- MACE: MACE training script
- ACE: ACE1pack or ACE.jl
- GAP: QUIP GAP trainer

#### 3. Extended LAMMPS Engine (`backend/common/engines/lammps.py`)

Extended `LAMMPSEngine` to support ML potentials in MD simulations.

**New Parameters**:
```python
parameters = {
    "workflow_type": "MD_NVT_LAMMPS",
    "temperature": 300.0,
    # ... other MD parameters ...

    # Session 18: ML potential support
    "ml_potential_config": {
        "descriptor_type": "SNAP",
        "elements": ["Si"],
        "files": {
            "coefficients": "/path/to/snap.coeffs",
            "parameters": "/path/to/snap.param"
        },
        "params": {...}
    }
}
```

**Extended Methods**:

1. `prepare_input()`: Now accepts `ml_potential_config` parameter
2. `_write_potential_section()`: Checks for ML potential config first
3. **NEW**: `_write_ml_potential()`: Writes ML potential commands

**Supported ML Potential Types**:

```python
def _write_ml_potential(self, f, ml_config: Dict[str, Any]):
    """Write ML potential section for various descriptor types."""
```

Supports:
- **SNAP** (Spectral Neighbor Analysis Potential)
  ```
  pair_style snap
  pair_coeff * * snap.param snap.coeffs Si
  ```

- **NequIP/MACE** (Neural network equivariant potentials)
  ```
  pair_style mliap model nequip model.pth
  pair_coeff * * Si
  ```

- **ACE** (Atomic Cluster Expansion)
  ```
  pair_style pace
  pair_coeff * * ace.yace Si
  ```

- **GAP** (Gaussian Approximation Potential via QUIP)
  ```
  pair_style quip
  pair_coeff * * gap.xml "Potential xml_label=GAP" Si
  ```

**Notes**:
- ML potentials require LAMMPS to be compiled with appropriate packages
- SNAP: Built into LAMMPS ML-SNAP package
- NequIP/MACE: Requires ML-IAP package or plugin
- GAP: Requires LAMMPS-QUIP interface
- ACE: Requires PACE package

#### 4. Hybrid DFT → ML-MD Workflow (Planned)

**Architecture**:

```
┌─────────────┐
│   DFT Jobs  │  Run small DFT dataset
│  (QE/VASP)  │  (MD snapshots or static)
└──────┬──────┘
       │ Energies + Forces
       ▼
┌─────────────────┐
│ Extract         │
│ Training Data   │  backend/common/ml/potentials.py
└──────┬──────────┘
       │ TrainingDataset
       ▼
┌─────────────────┐
│ Train ML        │
│ Potential       │  train_ml_potential()
└──────┬──────────┘
       │ MLPotential (in DB)
       ▼
┌─────────────────┐
│ LAMMPS MD       │  Large-scale MD with
│ with ML Pot     │  trained potential
└─────────────────┘
```

**Implementation Status**: ⚠️ **Partially Complete**
- ✅ MLPotential model
- ✅ Training data extraction
- ✅ Stub training implementation
- ✅ LAMMPS ML potential support
- ⚠️ Hybrid workflow orchestration (pending)
- ⚠️ Worker task for training (pending)
- ⚠️ API endpoints (pending)

---

## Session 19: Bayesian Optimization for Materials Design ✅

### Goals

Implement Bayesian Optimization (BO) for intelligent materials discovery within the existing DesignCampaign framework.

### Deliverables

#### 1. BO Module (`backend/common/ml/bo.py`)

**Gaussian Process-based BO**:

```python
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class Candidate:
    """Candidate structure for optimization."""
    structure_id: Optional[str]
    parameters: Dict[str, float]  # Continuous design parameters
    predicted_score: Optional[float]
    predicted_uncertainty: Optional[float]

@dataclass
class BOConfig:
    """Bayesian optimization configuration."""
    target_property: str  # "bandgap", "formation_energy"
    target_range: Tuple[float, float]  # Desired range
    acquisition_function: str = "ei"  # "ei", "ucb", "poi"
    n_initial_random: int = 10
    n_iterations: int = 50
    batch_size: int = 5

def suggest_candidates(
    target_config: BOConfig,
    existing_data: List[Dict[str, Any]],
    n_suggestions: int = 5
) -> List[Candidate]:
    """
    Suggest new candidates using Bayesian optimization.

    Uses Gaussian Process surrogate model with chosen acquisition
    function (Expected Improvement, Upper Confidence Bound, etc.)
    """
```

**Acquisition Functions**:
- **Expected Improvement (EI)**: Balance exploitation and exploration
- **Upper Confidence Bound (UCB)**: Optimistic sampling
- **Probability of Improvement (POI)**: Simple probability-based

#### 2. Parameter Space Definition

**DesignParameterSpace Schema**:

```python
{
    "parameter_space": {
        "type": "compositional",  # or "structural"

        # For compositional optimization
        "base_structure_id": "uuid",
        "dopant_sites": [0, 1, 2],  # Atom indices
        "dopant_elements": ["Al", "Ga", "In"],
        "dopant_fractions": [0.0, 0.5],  # Range

        # For structural optimization
        "lattice_strain": {
            "a": [0.95, 1.05],  # Fractional change
            "b": [0.95, 1.05],
            "c": [0.95, 1.05]
        },

        # Continuous parameters
        "parameters": {
            "doping_x": {"min": 0.0, "max": 1.0, "type": "continuous"},
            "strain": {"min": -0.05, "max": 0.05, "type": "continuous"}
        }
    }
}
```

#### 3. DesignCampaign Integration

**Extended Campaign Config**:

```python
{
    "strategy": "bayesian_optimization",
    "target_property": "bandgap",
    "target_range": [2.0, 3.0],  # Target bandgap range
    "parameter_space": {...},  # As defined above
    "bo_config": {
        "acquisition_function": "ei",
        "n_initial_random": 10,
        "batch_size": 5
    },
    "max_iterations": 50,
    "simulation_budget": 100  # Max simulations to run
}
```

**Updated `run_design_iteration()` Logic**:

```python
async def run_design_iteration(campaign_id: UUID, db: AsyncSession):
    """
    Run one iteration of design campaign.

    For BO strategy:
    1. Gather all evaluated structures from previous iterations
    2. Build dataset (parameters → property values)
    3. Fit Gaussian Process surrogate model
    4. Use acquisition function to suggest new candidates
    5. Generate structures for candidates
    6. Run ML predictions (fast)
    7. Optionally select subset for expensive DFT simulations
    """
```

#### 4. API Endpoints (Planned)

```python
POST /api/design/campaigns/bo
{
    "name": "Bandgap BO Campaign",
    "material_id": "uuid",
    "strategy": "bayesian_optimization",
    "target_property": "bandgap",
    "target_range": [2.0, 3.0],
    "parameter_space": {...},
    "bo_config": {...}
}

POST /api/design/campaigns/{id}/step
# Run next BO iteration
# Returns: New candidates suggested

GET /api/design/campaigns/{id}/trajectory
# Returns optimization trajectory:
# - Best score vs iteration
# - Hypervolume improvement
# - Pareto frontier (for multi-objective)
```

**Implementation Status**: ✅ **IMPLEMENTED**
- ✅ BO module with GP surrogate (SimpleGP)
- ✅ Acquisition functions (EI, UCB, POI)
- ✅ Parameter space handling
- ✅ Candidate suggestion with BO
- ✅ Pareto front computation for multi-objective
- ⚠️ API endpoints (pending)
- ⚠️ DesignCampaign integration (campaign model already supports BO strategy)

**Notes**:
- Uses simplified Gaussian Process implementation (good for small datasets)
- For production with large datasets, consider GPyTorch or BoTorch
- Gracefully falls back to random sampling when scipy unavailable
- Multi-start optimization for acquisition function maximization

---

## Session 20: Active Learning for Smart Simulation Selection ✅

### Goals

Add uncertainty estimation to ML models and implement active learning to intelligently select which candidates deserve expensive DFT/MD simulations.

### Deliverables

#### 1. Uncertainty Estimation in GNN Models ✅

**Extended `backend/common/ml/models/cgcnn_like.py`**:

```python
class CGCNNModel:
    def predict_with_uncertainty(
        self,
        graph_repr: Dict[str, Any],
        method: str = "mc_dropout"  # or "ensemble"
    ) -> Dict[str, Any]:
        """
        Predict with uncertainty estimate.

        Methods:
        - mc_dropout: Monte Carlo dropout (multiple forward passes)
        - ensemble: Use multiple models and compute variance

        Returns:
            {
                "prediction": float,
                "uncertainty": float,  # Standard deviation
                "predictions_raw": List[float],  # For MC dropout
                "metadata": {...}
            }
        """
```

**MC Dropout Implementation**:
```python
def _predict_mc_dropout(self, graph_repr, n_samples=50):
    """Run multiple forward passes with dropout enabled."""
    predictions = []
    for _ in range(n_samples):
        pred = self._forward_with_dropout(graph_repr)
        predictions.append(pred)

    return {
        "prediction": np.mean(predictions),
        "uncertainty": np.std(predictions),
        "predictions_raw": predictions
    }
```

**Ensemble Implementation**:
```python
def _predict_ensemble(self, graph_repr, model_names):
    """Use multiple models from registry."""
    predictions = []
    for model_name in model_names:
        model = get_model(model_name)
        pred = model.predict(graph_repr)
        predictions.append(pred["prediction"])

    return {
        "prediction": np.mean(predictions),
        "uncertainty": np.std(predictions),
        "model_agreement": np.std(predictions) / np.mean(predictions)
    }
```

#### 2. Active Learning Module (`backend/common/ml/active_learning.py`) ✅

**Selection Strategies**:

```python
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ALCandidate:
    """Candidate with ML prediction and uncertainty."""
    structure_id: str
    predicted_value: float
    predicted_uncertainty: float
    acquisition_score: float  # Combined score for selection

@dataclass
class ALConfig:
    """Active learning configuration."""
    strategy: str = "uncertainty"  # or "expected_improvement", "greedy_uncertainty"
    budget: int = 10  # How many to select for simulation
    min_uncertainty: float = 0.01  # Minimum uncertainty threshold
    exploration_weight: float = 1.0  # Balance exploration/exploitation

def select_candidates_for_simulation(
    candidates: List[Dict[str, Any]],
    budget: int,
    strategy: str = "uncertainty"
) -> List[ALCandidate]:
    """
    Select candidates for expensive simulations using active learning.

    Strategies:
    - uncertainty: Select highest uncertainty
    - greedy_uncertainty: High uncertainty + promising value
    - expected_improvement: BO-style expected improvement
    - diverse_uncertainty: Diverse set with high uncertainty
    """

def compute_acquisition_score(
    candidate: Dict[str, Any],
    strategy: str,
    current_best: float,
    config: ALConfig
) -> float:
    """
    Compute acquisition score for a candidate.

    Combines predicted performance and uncertainty.
    """
```

**Example Acquisition Functions**:

```python
def uncertainty_score(pred_value, uncertainty, config):
    """Pure uncertainty sampling."""
    return uncertainty

def greedy_uncertainty_score(pred_value, uncertainty, target_value, config):
    """Balance performance and uncertainty."""
    value_score = -abs(pred_value - target_value)  # Close to target
    return value_score + config.exploration_weight * uncertainty

def expected_improvement_score(pred_value, uncertainty, current_best, config):
    """Expected improvement over current best."""
    from scipy.stats import norm
    improvement = pred_value - current_best
    z = improvement / (uncertainty + 1e-9)
    ei = improvement * norm.cdf(z) + uncertainty * norm.pdf(z)
    return ei
```

#### 3. DesignCampaign Integration

**Extended Campaign Config**:

```python
{
    "strategy": "bayesian_optimization",  # or "genetic_algorithm"
    "target_property": "bandgap",

    # Session 20: Active learning config
    "active_learning": {
        "enabled": true,
        "strategy": "greedy_uncertainty",
        "simulation_budget_per_iteration": 5,  # How many to simulate
        "min_uncertainty": 0.01,
        "exploration_weight": 1.0
    },

    "max_iterations": 50
}
```

**Updated Iteration Logic**:

```python
async def run_design_iteration_with_al(campaign_id: UUID, db: AsyncSession):
    """
    Run design iteration with active learning.

    1. Generate candidate structures (via BO, GA, etc.)
    2. Run ML predictions with uncertainty for ALL candidates
    3. Use active learning to select subset for DFT simulation
    4. Create SimulationJob entries for selected candidates
    5. For non-selected: keep ML predictions only
    6. Track which candidates were simulated vs predicted-only
    """
```

#### 4. Tracking Label Status

**Extend DesignIteration Model**:

```python
class DesignIteration(Base):
    # ... existing fields ...

    # Session 20: Active learning tracking
    candidates_ml_predicted: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Number of candidates with ML predictions only"
    )

    candidates_simulated: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Number of candidates with DFT/MD simulations"
    )

    simulation_budget_used: Mapped[int] = mapped_column(
        Integer,
        default=0,
        comment="Cumulative simulation budget used"
    )

    active_learning_stats: Mapped[Optional[dict]] = mapped_column(
        JSON,
        nullable=True,
        comment="AL statistics (avg uncertainty, selection criteria, etc.)"
    )
```

#### 5. API Endpoints (Planned)

```python
# Existing endpoint extended
POST /api/ml/gnn/properties
{
    "structure_id": "uuid",
    "gnn_model_name": "cgcnn_bandgap_v1",
    "return_uncertainty": true,  # Session 20
    "uncertainty_method": "mc_dropout",
    "n_samples": 50  # For MC dropout
}

# Response includes uncertainty
{
    "prediction": 2.45,
    "uncertainty": 0.12,  # Standard deviation
    ...
}

# New endpoint
POST /api/design/campaigns/{id}/select_for_simulation
{
    "strategy": "greedy_uncertainty",
    "budget": 10
}

# Returns selected candidates with scores
{
    "selected": [
        {
            "structure_id": "uuid",
            "predicted_value": 2.4,
            "uncertainty": 0.3,
            "acquisition_score": 1.2,
            "reason": "high_uncertainty_promising_value"
        },
        ...
    ],
    "total_candidates": 50,
    "selected_count": 10
}
```

#### 6. Retraining Integration

**Periodic Model Retraining**:

As new DFT simulations complete via active learning, the training dataset grows. Implement periodic retraining:

```python
async def check_and_retrain(db: AsyncSession):
    """
    Check if enough new labeled data has accumulated to warrant retraining.

    Criteria:
    - At least N new labeled structures since last training
    - Significant improvement expected based on validation
    """
    # Get latest model
    latest_model = get_latest_model_for_target(target="bandgap")

    # Count new simulations since model training
    new_simulations = count_simulations_since(latest_model.created_at)

    if new_simulations >= RETRAIN_THRESHOLD:
        # Trigger retraining job (Session 16 infrastructure)
        job = submit_training_job(
            target_property="bandgap",
            model_name=f"cgcnn_bandgap_{next_version}"
        )
```

**Implementation Status**: ✅ **IMPLEMENTED**
- ✅ GNN uncertainty estimation via MC dropout
- ✅ Dropout layers added to CGCNN model
- ✅ Active learning module with multiple selection strategies
- ✅ Candidate classification by confidence
- ✅ Simulation budget tracking
- ✅ API schemas for uncertainty prediction and candidate selection
- ⚠️ API endpoints implementation (pending)
- ⚠️ Retraining triggers (pending)
- ⚠️ DesignCampaign integration (campaign model ready)

**Implemented Features**:
- **MC Dropout**: Multiple forward passes with dropout to estimate uncertainty
- **Selection Strategies**:
  - `uncertainty`: Pure uncertainty sampling
  - `greedy_uncertainty`: Balance value and uncertainty
  - `expected_improvement`: BO-style expected improvement
- **Confidence Classification**: Automatic classification into high/low confidence
- **Budget Management**: SimulationBudget class for tracking usage
- **Dataset Quality Analysis**: Estimate model confidence from uncertainties
- **Information Gain Tracking**: Measure uncertainty reduction from simulations

**Notes**:
- MC dropout requires model trained with dropout layers (dropout=0.1)
- Ensemble method placeholder (can be extended with multiple model ensembles)
- Stub implementation provides simulated uncertainty for testing

---

## File Structure

```
New/Modified Files:
===================

Session 18 (✅ Completed):
  src/api/models/ml_potential.py           (NEW, ~160 lines)
  src/api/models/__init__.py               (MODIFIED: +2 lines, add MLPotential)
  backend/common/ml/potentials.py          (NEW, ~700 lines)
  backend/common/engines/lammps.py         (MODIFIED: +150 lines for ML potentials)

Session 19 (✅ Completed):
  backend/common/ml/bo.py                  (NEW, ~650 lines)
  src/api/models/campaign.py               (OK: already supports BO via config.strategy)
  src/api/routes/design.py                 (PENDING: add BO endpoints)
  src/worker/tasks/design_tasks.py         (PENDING: BO iteration logic)

Session 20 (✅ Completed):
  backend/common/ml/active_learning.py     (NEW, ~450 lines)
  backend/common/ml/models/cgcnn_like.py   (MODIFIED: +130 lines, uncertainty methods + dropout)
  src/api/schemas/ml.py                    (MODIFIED: +230 lines, AL schemas)
  src/api/models/campaign.py               (OK: supports AL via metadata)
  src/api/routes/ml.py                     (PENDING: uncertainty endpoints)

Documentation:
  SESSIONS_18-20_IMPLEMENTATION.md         (UPDATED, this file, ~1100 lines)
```

**Total New/Modified Lines**:
- Session 18: ~1,012 lines
- Session 19: ~650 lines
- Session 20: ~810 lines
- Documentation: ~1,100 lines
- **Grand Total: ~3,572 lines**
```

---

## Integration Roadmap

### Immediate (Session 18)
- ✅ MLPotential model
- ✅ Training data extraction
- ✅ LAMMPS ML potential support
- ⚠️ Hybrid workflow orchestration
- ⚠️ API endpoints for ML potentials
- ⚠️ Worker task for training
- ⚠️ Database migration for MLPotential

### Short-term (Session 19)
- Implement BO module with GP surrogate
- Extend DesignCampaign for BO strategy
- Create parameter space schema
- Add BO iteration logic
- API endpoints for BO campaigns
- Frontend: BO campaign creation and trajectory visualization

### Medium-term (Session 20)
- Implement MC dropout for GNN uncertainty
- Implement ensemble uncertainty
- Create active learning selection module
- Integrate AL with design campaigns
- Track simulation budget usage
- Periodic model retraining triggers

---

## Testing Strategy

### Unit Tests (Planned)

```python
# Test ML potential training
def test_extract_snapshots():
    result = create_mock_simulation_result()
    snapshots = extract_snapshots_from_simulation(result)
    assert len(snapshots) > 0
    assert snapshots[0].energy is not None

def test_build_training_dataset():
    results = [mock_result_1(), mock_result_2()]
    dataset = build_training_dataset(results)
    assert dataset.num_snapshots == expected_count
    assert len(dataset.elements) > 0

def test_train_snap_potential():
    config = PotentialConfig(name="test", descriptor_type="SNAP", ...)
    dataset = create_mock_dataset()
    result = train_ml_potential(config, dataset, output_dir)
    assert result["status"] == "success"
    assert "files" in result

# Test LAMMPS ML potential
def test_lammps_ml_potential_input():
    engine = LAMMPSEngine()
    params = {
        "ml_potential_config": {
            "descriptor_type": "SNAP",
            "elements": ["Si"]
        }
    }
    input_dir = engine.prepare_input(structure, params, work_dir)
    # Check that LAMMPS input contains SNAP commands
    with open(input_dir / "in.lammps") as f:
        content = f.read()
        assert "pair_style snap" in content

# Test BO
def test_bo_suggestion():
    config = BOConfig(target_property="bandgap", target_range=(2, 3))
    existing_data = [...]
    candidates = suggest_candidates(config, existing_data, n_suggestions=5)
    assert len(candidates) == 5
    assert all(c.predicted_score is not None for c in candidates)

# Test Active Learning
def test_active_learning_selection():
    candidates = [
        {"structure_id": "1", "prediction": 2.5, "uncertainty": 0.3},
        {"structure_id": "2", "prediction": 2.6, "uncertainty": 0.1},
    ]
    selected = select_candidates_for_simulation(candidates, budget=1)
    assert len(selected) == 1
    # Should select candidate 1 (higher uncertainty)
    assert selected[0].structure_id == "1"
```

### Integration Tests (Planned)

```python
# Test end-to-end hybrid workflow
async def test_hybrid_dft_ml_workflow():
    # 1. Run DFT jobs
    dft_jobs = await submit_dft_jobs(structures)
    await wait_for_completion(dft_jobs)

    # 2. Extract training data
    results = [job.result.to_dict() for job in dft_jobs]
    dataset = build_training_dataset(results)

    # 3. Train ML potential
    config = PotentialConfig(...)
    training_result = train_ml_potential(config, dataset, output_dir)

    # 4. Register in database
    potential = MLPotential(
        name=config.name,
        files=training_result["files"],
        metrics=training_result["metrics"]
    )
    db.add(potential)
    await db.commit()

    # 5. Run LAMMPS MD with ML potential
    md_job = await submit_lammps_job(
        structure,
        ml_potential_config={
            "descriptor_type": config.descriptor_type,
            "files": training_result["files"],
            "elements": config.elements
        }
    )
    await wait_for_completion([md_job])
    assert md_job.status == "COMPLETED"
```

---

## Known Limitations & Future Work

### Session 18 (ML Potentials)

**Current**:
- ✅ MLPotential model implemented
- ✅ Training data extraction implemented
- ✅ LAMMPS ML potential support implemented
- ⚠️ Training is stub implementation (doesn't actually train)
- ⚠️ Hybrid workflow orchestration not implemented
- ⚠️ API endpoints not implemented
- ⚠️ Worker tasks not implemented

**Future**:
- Integrate actual ML potential trainers (FitSNAP, NequIP, MACE)
- Implement hybrid workflow orchestration (DFT → Train → MD pipeline)
- Add API endpoints for potential management and training
- Create worker tasks for background training
- Add validation and testing suites for trained potentials
- Support transfer learning (fine-tuning existing potentials)
- Implement uncertainty quantification for ML potentials
- Add committee models for robust predictions

### Session 19 (Bayesian Optimization)

**Planned Features**:
- Gaussian Process surrogate modeling
- Multiple acquisition functions (EI, UCB, POI)
- Multi-objective BO (Pareto optimization)
- Constraint handling (e.g., stability constraints)
- Batch BO (suggest multiple candidates at once)
- Transfer learning (use data from similar systems)

**Challenges**:
- Parameter space encoding (how to represent structures as continuous vectors)
- Handling discrete parameters (element choices)
- Scaling to high-dimensional spaces
- Computational cost of GP training for large datasets

### Session 20 (Active Learning)

**Planned Features**:
- Multiple uncertainty estimation methods
- Various selection strategies
- Batch-mode active learning
- Cost-aware selection (account for simulation cost)
- Explore-exploit tradeoffs
- Automatic retraining triggers

**Challenges**:
- Uncertainty calibration (ensure uncertainty estimates are accurate)
- Cold start problem (need initial labeled data)
- Balancing exploration and exploitation
- Computational overhead of uncertainty estimation
- Online learning vs batch retraining

---

## Dependencies

### Python Packages

**Required** (already in requirements):
- sqlalchemy
- alembic
- pydantic
- numpy
- pandas

**Optional** (for full functionality):
- **PyTorch**: For GNN models (already conditional)
- **scikit-learn**: For GP surrogate models, AL strategies
- **GPyTorch** or **GPy**: For scalable Gaussian Processes
- **BoTorch**: For advanced Bayesian optimization
- **scipy**: For statistical functions (already likely installed)

### External Tools

**For ML Potential Training**:
- **FitSNAP**: SNAP potential fitting
- **NequIP**: Equivariant neural network potentials
- **MACE**: Multi-Atomic Cluster Expansion
- **QUIP**: GAP training
- **ACE1pack** or **ACE.jl**: Atomic Cluster Expansion

**For LAMMPS**:
- LAMMPS compiled with:
  - ML-SNAP package (for SNAP potentials)
  - ML-IAP package (for neural network potentials)
  - PACE package (for ACE potentials)
  - QUIP interface (for GAP potentials)

---

## Summary

Sessions 18-20 implement the **autonomous materials discovery loop**:

1. **Session 18**: Train ML potentials from DFT data → Run fast MD with ML potentials
2. **Session 19**: Use Bayesian optimization to intelligently suggest promising candidates
3. **Session 20**: Use active learning to select which candidates deserve expensive simulations

This creates a closed loop:
```
┌────────────────────────────────────────────────────┐
│  Bayesian Optimization suggests candidates         │
└───────────────────┬────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│  ML models predict properties + uncertainties      │
└───────────────────┬────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│  Active Learning selects promising/uncertain ones  │
└───────────────────┬────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│  Run DFT simulations on selected candidates        │
└───────────────────┬────────────────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────┐
│  Retrain ML models with new data                   │
└───────────────────┬────────────────────────────────┘
                    │
                    └─────► (loop back to BO)
```

**Current Completion**:
- Session 18 Core: **100%** ✅
- Session 18 Integration: **30%** ⚠️
- Session 19 Core: **100%** ✅
- Session 19 Integration: **20%** ⚠️
- Session 20 Core: **100%** ✅
- Session 20 Integration: **30%** ⚠️

**Total New Code**: ~3,572 lines (all core modules implemented)

---

**Next Steps**:
1. Complete API Endpoint Integration:
   - ML potential management endpoints (Session 18)
   - BO campaign endpoints (Session 19)
   - Uncertainty prediction endpoints (Session 20)
   - Active learning selection endpoints (Session 20)
2. Complete Worker Task Integration:
   - Hybrid DFT→ML-MD workflow orchestration
   - BO iteration execution
   - Active learning candidate selection
   - Automatic model retraining triggers
3. Database Migration:
   - Create migration for MLPotential table
   - Add indexes for performance
4. End-to-end Testing:
   - Test hybrid workflow (DFT → Train ML potential → MD)
   - Test BO-driven design campaign
   - Test active learning selection
   - Test full autonomous discovery loop
5. Frontend Components:
   - ML potential management UI
   - BO campaign creation and trajectory visualization
   - Active learning candidate review UI
   - Uncertainty visualization
6. Production Integration:
   - Replace stub ML potential training with real trainers (FitSNAP, NequIP, etc.)
   - Optimize GP training for larger datasets (consider GPyTorch/BoTorch)
   - Calibrate uncertainty estimates
   - Add ensemble methods

# NANO-OS Design Loops & Campaigns

## Overview

NANO-OS supports **automated materials design** through iterative loops that combine:
- **Candidate generation** (structure search algorithms)
- **Property evaluation** (ML predictions or simulations)
- **Optimization** (Bayesian optimization, genetic algorithms, active learning)

This is where **AGI/AI agents** can plug in to autonomously discover new materials.

## Design Campaign Concept

A **Design Campaign** is a multi-iteration search process with:
- **Search Space**: What structures to explore (element combinations, topologies, etc.)
- **Objective**: What property to optimize (maximize bandgap, minimize cost, etc.)
- **Constraints**: Requirements that must be satisfied (stability > 0.7, max 20 atoms, etc.)
- **Algorithm**: How to search (random, genetic, Bayesian, etc.)
- **Evaluation Method**: ML predictions or actual simulations

### Campaign Lifecycle

```
CREATE campaign
  ↓
FOR iteration = 1 to max_iterations:
  ↓
  GENERATE batch of candidate structures
    ↓
  EVALUATE candidates (ML or simulation)
    ↓
  RANK candidates by objective
    ↓
  UPDATE search strategy (algorithm-specific)
    ↓
  STORE best candidates
    ↓
  CHECK convergence criteria
    ↓
  IF converged or max_iterations → COMPLETE
  ↓
END
```

## Data Model

```python
class DesignCampaign(Base):
    """
    Represents a multi-iteration materials design campaign.
    """
    id: UUID
    owner_id: UUID
    name: str
    description: str

    # Search configuration
    search_space: Dict  # JSON: element pool, constraints, etc.
    objective: str      # "maximize_bandgap", "minimize_formation_energy", etc.
    config: Dict        # JSON: algorithm-specific parameters

    # Campaign status
    status: str         # "ACTIVE", "PAUSED", "COMPLETED", "FAILED"
    iterations_completed: int
    max_iterations: int

    # Results
    best_candidates: List[Dict]  # JSON: top N structures + properties

    # Metadata
    created_at: datetime
    updated_at: datetime

    # Relationships
    iterations: List[DesignIteration]  # Detailed per-iteration data
```

```python
class DesignIteration(Base):
    """
    Represents a single iteration within a campaign.
    """
    id: UUID
    campaign_id: UUID (FK)
    iteration_number: int

    # Candidates evaluated in this iteration
    candidates_evaluated: int
    best_objective_value: float

    # Algorithm state (for stateful algorithms like Bayesian Optimization)
    algorithm_state: Dict  # JSON: GP hyperparameters, etc.

    # Timing
    started_at: datetime
    completed_at: datetime

    # Relationships
    candidate_structures: List[CandidateStructure]
```

```python
class CandidateStructure(Base):
    """
    A candidate structure generated during a design iteration.
    """
    id: UUID
    iteration_id: UUID (FK)
    structure_id: UUID (FK)  # Links to main Structure table

    # Evaluation results
    predicted_properties: Dict  # JSON: bandgap, stability, etc.
    simulation_job_id: UUID (FK, nullable)  # If DFT refinement was done
    objective_value: float      # Value of the objective function
    rank: int                   # Rank within this iteration

    # Provenance
    generation_method: str      # "mutation", "crossover", "BO_acquisition", etc.
    parent_structure_ids: List[UUID]  # If derived from other structures

    created_at: datetime
```

## Search Algorithms

### 1. Random Search

**Description**: Generate random structures within search space. Baseline method.

**Parameters**:
```python
{
    "algorithm": "random",
    "batch_size": 20,  # Candidates per iteration
    "seed": 42
}
```

**Use Case**: Establishing baseline performance, small search spaces.

---

### 2. Genetic Algorithm

**Description**: Evolve structures through crossover and mutation.

**Parameters**:
```python
{
    "algorithm": "genetic",
    "population_size": 50,
    "elite_fraction": 0.2,        # Top 20% survive
    "crossover_probability": 0.7,
    "mutation_probability": 0.3,
    "mutation_types": ["swap_element", "shift_atom", "change_lattice"]
}
```

**Operations**:
- **Crossover**: Combine two parent structures (e.g., merge supercells)
- **Mutation**: Perturb structure (swap elements, shift atoms, etc.)
- **Selection**: Keep top performers + some random for diversity

**Use Case**: Large search spaces, when structure similarity metrics are available.

---

### 3. Bayesian Optimization

**Description**: Use a Gaussian Process (GP) to model property landscape, select candidates via acquisition function.

**Parameters**:
```python
{
    "algorithm": "bayesian_optimization",
    "acquisition_function": "EI",  # "EI", "UCB", "PI"
    "batch_size": 10,
    "gp_kernel": "RBF",            # or "Matern"
    "exploration_weight": 0.1      # Balance explore vs exploit
}
```

**Workflow**:
1. Train GP on evaluated structures
2. Compute acquisition function over search space
3. Select top candidates by acquisition value
4. Evaluate candidates
5. Update GP and repeat

**Use Case**: Expensive evaluations (DFT), continuous property spaces.

---

### 4. Active Learning

**Description**: Iteratively train ML models to predict properties, use uncertainty to select candidates.

**Parameters**:
```python
{
    "algorithm": "active_learning",
    "batch_size": 20,
    "model_type": "GNN",           # "GNN", "random_forest", etc.
    "uncertainty_metric": "ensemble_std",  # or "dropout_std"
    "retrain_frequency": 2         # Retrain model every N iterations
}
```

**Workflow**:
1. Train ML model on initial dataset
2. Predict properties for candidate pool with uncertainty estimates
3. Select candidates with high uncertainty (exploration) or high objective value (exploitation)
4. Evaluate selected candidates (simulation or experiment)
5. Add to dataset, retrain model, repeat

**Use Case**: Large search spaces, when ML models can be trained quickly.

---

### 5. AGI-Driven Search (Future)

**Description**: External AI agent (GPT-4, Claude, specialized agent) proposes candidates.

**Parameters**:
```python
{
    "algorithm": "agi_agent",
    "agent_endpoint": "https://api.myagi.com/v1/propose",
    "agent_api_key": "sk-...",
    "batch_size": 10,
    "context_window": 100  # Number of past evaluations to send to agent
}
```

**Workflow**:
1. Send campaign objective + past iterations to AGI agent
2. Agent proposes new structures (as CIF/POSCAR or via composition + topology)
3. NANO-OS evaluates candidates
4. Results fed back to agent for next iteration

**Integration**:
```python
# Pseudo-code for AGI agent call
async def agi_propose_candidates(campaign: DesignCampaign) -> List[Structure]:
    history = get_campaign_history(campaign.id)

    response = await httpx.post(
        campaign.config["agent_endpoint"],
        json={
            "objective": campaign.objective,
            "search_space": campaign.search_space,
            "history": history,
            "batch_size": campaign.config["batch_size"]
        },
        headers={"Authorization": f"Bearer {campaign.config['agent_api_key']}"}
    )

    structures = parse_agi_response(response.json())
    return structures
```

**Use Case**: Complex design objectives, leveraging LLM knowledge of materials science.

---

## Objective Functions

### Standard Objectives

```python
OBJECTIVES = {
    "maximize_bandgap": {
        "target_property": "bandgap",
        "direction": "maximize",
        "units": "eV"
    },
    "minimize_formation_energy": {
        "target_property": "formation_energy",
        "direction": "minimize",
        "units": "eV/atom"
    },
    "maximize_stability": {
        "target_property": "stability_score",
        "direction": "maximize",
        "units": "dimensionless"
    }
}
```

### Custom Objectives

Users can define custom objectives:

```python
# Multi-objective: maximize bandgap AND stability
{
    "objective": "custom",
    "formula": "0.6 * bandgap + 0.4 * stability_score",
    "constraints": [
        {"property": "bandgap", "min": 1.0},
        {"property": "stability_score", "min": 0.7}
    ]
}
```

---

## Constraints

### Element Constraints

```python
{
    "allowed_elements": ["Mo", "W", "S", "Se", "Te"],
    "forbidden_elements": ["Hg", "Cd", "Pb"],  # Toxic
    "max_num_elements": 3  # Binary or ternary only
}
```

### Structure Constraints

```python
{
    "dimensionality": 2,           # 2D materials only
    "max_atoms_per_cell": 20,      # Computational feasibility
    "min_atoms_per_cell": 2,
    "symmetry": "hexagonal"        # or null for any
}
```

### Property Constraints

```python
{
    "min_stability": 0.7,
    "max_formation_energy": -2.0,  # eV/atom
    "bandgap_range": [1.0, 3.0]    # eV
}
```

---

## Evaluation Methods

### 1. ML Prediction Only

**Fast**: ~100-1000 candidates/minute

```python
{
    "evaluation_method": "ml_only",
    "ml_model": "cgcnn_bandgap_v1"
}
```

**Workflow**: For each candidate, call `/api/ml/gnn/properties`.

**Use Case**: Initial screening, large search spaces.

---

### 2. DFT Refinement

**Slow**: ~1-10 candidates/hour (depending on size and resources)

```python
{
    "evaluation_method": "dft_refinement",
    "dft_workflow": "DFT_RELAX_QE",
    "dft_params": {
        "ecutwfc": 50,
        "ecutrho": 400
    },
    "ml_prescreen": True,  # Use ML to filter before DFT
    "ml_threshold": 2.0    # Only run DFT if ML predicts bandgap > 2.0 eV
}
```

**Workflow**:
1. ML predictions for all candidates
2. Filter by threshold
3. Submit DFT jobs for top candidates
4. Wait for completion (parallel jobs)
5. Use DFT results as ground truth

**Use Case**: Final validation, small batches, high-accuracy needed.

---

### 3. Hybrid (ML + Selective DFT)

```python
{
    "evaluation_method": "hybrid",
    "ml_model": "cgcnn_bandgap_v1",
    "dft_workflow": "DFT_SCF_QE",
    "dft_fraction": 0.2,  # Run DFT on top 20% of ML predictions
    "dft_trigger": "uncertainty"  # Run DFT on high-uncertainty predictions
}
```

**Use Case**: Balance speed and accuracy.

---

## Campaign API Usage

### Create Campaign

```bash
POST /api/campaigns
{
  "name": "Wide bandgap 2D TMDs",
  "description": "Search for 2D transition metal dichalcogenides with bandgap > 2.5 eV",
  "search_space": {
    "allowed_elements": ["Mo", "W", "S", "Se", "Te"],
    "dimensionality": 2,
    "max_atoms_per_cell": 10
  },
  "objective": "maximize_bandgap",
  "config": {
    "algorithm": "bayesian_optimization",
    "acquisition_function": "EI",
    "batch_size": 20,
    "evaluation_method": "ml_only",
    "ml_model": "cgcnn_bandgap_v1"
  },
  "max_iterations": 10
}
```

**Response**:
```json
{
  "id": "campaign-uuid",
  "status": "ACTIVE",
  "iterations_completed": 0,
  "best_candidates": []
}
```

Campaign starts automatically. Worker task `run_design_iteration()` is triggered.

---

### Monitor Campaign

```bash
GET /api/campaigns/{id}
```

**Response**:
```json
{
  "id": "campaign-uuid",
  "name": "Wide bandgap 2D TMDs",
  "status": "ACTIVE",
  "iterations_completed": 3,
  "max_iterations": 10,
  "best_candidates": [
    {
      "structure_id": "uuid-1",
      "formula": "WS2",
      "bandgap": 2.89,
      "stability_score": 0.92,
      "iteration": 2,
      "rank": 1
    },
    {
      "structure_id": "uuid-2",
      "formula": "MoSe2",
      "bandgap": 2.67,
      "stability_score": 0.88,
      "iteration": 3,
      "rank": 2
    }
  ],
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-01-15T11:30:00Z"
}
```

---

### Get Campaign Statistics

```bash
GET /api/campaigns/{id}/stats
```

**Response**:
```json
{
  "total_candidates_evaluated": 153,
  "best_objective_value": 2.89,
  "convergence_trend": [2.1, 2.5, 2.8, 2.85, 2.87, 2.89, 2.89, 2.89],
  "iteration_times": [45, 48, 52, 51, 49, 50, 48, 47],  // seconds
  "estimated_time_remaining": "10 minutes"
}
```

---

### Pause/Resume Campaign

```bash
POST /api/campaigns/{id}/pause
POST /api/campaigns/{id}/resume
```

Useful for adjusting parameters mid-campaign or debugging.

---

## AGI Agent Integration

### How an AGI Agent Uses NANO-OS

```python
# Pseudo-code for an AGI-driven materials discovery agent

import asyncio
import anthropic  # or openai

async def agi_materials_discovery():
    # 1. Create campaign via API
    campaign = await create_campaign(
        name="AGI-driven superconductor search",
        search_space={
            "allowed_elements": ["Y", "Ba", "Cu", "O", "La", "Sr"],
            "max_atoms_per_cell": 30
        },
        objective="maximize_tc",  # Critical temperature
        config={
            "algorithm": "agi_agent",
            "batch_size": 10,
            "evaluation_method": "dft_refinement",
            "dft_workflow": "DFT_RELAX_QE"
        },
        max_iterations": 20
    )

    # 2. AGI iteration loop
    for iteration in range(20):
        # Get campaign history
        history = await get_campaign_history(campaign["id"])

        # AGI analyzes history and proposes new structures
        ai_prompt = f"""
        You are a materials scientist. Based on the following campaign history,
        propose 10 new candidate structures for high-Tc superconductors.

        Objective: Maximize critical temperature (Tc)
        Elements available: Y, Ba, Cu, O, La, Sr

        Past iterations:
        {format_history(history)}

        Propose structures as CIF format.
        """

        response = anthropic.messages.create(
            model="claude-sonnet-4-5-20250929",
            messages=[{"role": "user", "content": ai_prompt}]
        )

        # Parse AI response into structure files
        candidate_cifs = extract_cif_blocks(response.content)

        # 3. Submit candidates to NANO-OS
        candidate_ids = []
        for cif_text in candidate_cifs:
            struct = await upload_structure(
                material_id=campaign["material_id"],
                file_content=cif_text,
                format="CIF",
                name=f"AGI candidate {iteration}-{len(candidate_ids)}"
            )
            candidate_ids.append(struct["id"])

        # 4. Launch simulations
        job_ids = []
        for struct_id in candidate_ids:
            job = await create_job(
                structure_id=struct_id,
                workflow_template_id="DFT_RELAX_QE",
                priority=10
            )
            job_ids.append(job["id"])

        # 5. Wait for completion
        await wait_for_jobs(job_ids)

        # 6. Collect results and update campaign
        results = await get_job_results(job_ids)
        await update_campaign_iteration(
            campaign["id"],
            iteration=iteration,
            candidates=results
        )

        # Check if converged
        if is_converged(results):
            break

    # 7. Return best material found
    final_campaign = await get_campaign(campaign["id"])
    best = final_campaign["best_candidates"][0]
    return best
```

### AGI Agent Design Patterns

#### Pattern 1: Structure Proposer

AGI acts as a **generative model** for structures.

```python
# AGI gets: objective, constraints, history
# AGI returns: List of structures (as CIF/POSCAR)

agi_response = """
Here are 10 candidate structures optimized for high bandgap:

1. WS2 monolayer (bandgap ~2.0 eV historically)
<CIF>
...
</CIF>

2. MoSe2 with 5% strain (strain increases bandgap)
<CIF>
...
</CIF>

... (8 more)
"""
```

#### Pattern 2: Strategy Advisor

AGI analyzes trends and suggests which regions of search space to explore.

```python
# AGI gets: campaign statistics, convergence plot
# AGI returns: Recommendations

agi_response = """
Analysis of iterations 1-5:
- Tungsten-based TMDs consistently show higher bandgaps than Mo-based
- 2D structures outperform bulk
- Recommendation: Focus on W-S and W-Se binaries, explore ternaries like W-S-Se

Suggested next batch:
- WS2 with different stacking
- WSe2 with vacancies
- W(S0.5Se0.5)2 alloy
"""
```

#### Pattern 3: Multi-Scale Orchestrator

AGI decides when to use ML vs DFT vs experiments.

```python
# AGI gets: candidate list with ML predictions
# AGI returns: Which candidates to validate with DFT

agi_response = """
Candidates 1, 3, 7 have high predicted bandgaps (>2.5 eV) with low ML uncertainty.
Recommendation: Run DFT on these.

Candidates 2, 4, 5 are similar to known materials (graphene derivatives).
Recommendation: Skip DFT, use database values.

Candidate 9 has high uncertainty. Recommendation: Get more training data first.
"""
```

---

## Provenance Tracking

Every candidate structure maintains a provenance chain:

```python
{
  "structure_id": "uuid",
  "generation_method": "genetic_crossover",
  "parent_structure_ids": ["parent-uuid-1", "parent-uuid-2"],
  "campaign_id": "campaign-uuid",
  "iteration": 5,
  "algorithm_params": {
    "crossover_type": "supercell_merge",
    "mutation_applied": False
  },
  "evaluation_results": {
    "ml_prediction": {"bandgap": 2.7},
    "dft_refinement": {"bandgap": 2.5}
  }
}
```

**Use Cases**:
- Reproducibility: Recreate the entire design process
- Analysis: Which generation methods produce best candidates?
- AGI Training: Use provenance as training data for better proposals

---

## Performance Considerations

### Parallel Evaluation

- ML predictions: Batch inference (100+ candidates simultaneously)
- DFT jobs: Parallel execution on worker pool (10-50 jobs depending on resources)

### Caching

- Structure features cached in `StructureFeatures` table
- ML model loaded once per worker process
- Candidate pool precomputed for combinatorial search spaces

### Early Stopping

```python
{
  "convergence_criteria": {
    "type": "no_improvement",
    "patience": 3,  # Stop if no improvement for 3 iterations
    "min_improvement": 0.05  # Must improve by at least 0.05 eV
  }
}
```

---

## Future Enhancements

1. **Multi-Objective Optimization**: Pareto fronts for bandgap vs stability
2. **Transfer Learning**: Use campaigns on one material class to initialize another
3. **Experiment-in-the-Loop**: Integrate with lab automation for synthesis validation
4. **Collaborative Campaigns**: Multiple users contribute to same campaign
5. **Campaign Templates**: Pre-configured campaigns for common use cases
6. **Real-Time Visualization**: Live plots of convergence, structure evolution
7. **Cost Optimization**: Minimize computational cost while meeting objectives

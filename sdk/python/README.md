# NANO-OS Python SDK

Python client library for interacting with the NANO-OS (Nanomaterials Operating System) API.

## Installation

```bash
pip install nano-os
```

Or install from source:

```bash
cd sdk/python
pip install -e .
```

## Quick Start

```python
from nano_os import NanoOSClient

# Initialize client
client = NanoOSClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Create a structure
structure = client.structures.create(
    composition="MoS2",
    lattice_type="hexagonal",
    metadata={"source": "manual_entry"}
)

# Submit a DFT job
job = client.jobs.submit_dft(
    structure_id=structure.id,
    functional="PBE",
    kpoints_density=0.03
)

# Check job status
status = client.jobs.get_status(job.id)
print(f"Job status: {status.status}")

# Run a design campaign
campaign = client.campaigns.create(
    name="TMD Bandgap Optimization",
    config={
        "target_properties": {
            "bandgap": {"value": 2.0, "tolerance": 0.2, "weight": 1.0}
        },
        "constraints": {
            "elements": ["Mo", "W", "S", "Se"],
            "dimensionality": 2
        },
        "max_iterations": 20,
        "candidates_per_iteration": 10
    }
)

# Run campaign iterations
result = client.campaigns.run_iterations(campaign.id, num_iterations=5)
```

## Workflow DSL

Define complex workflows using YAML or JSON:

```yaml
# workflow.yaml
name: "TMD Screening Workflow"
version: "1.0"

steps:
  - name: "generate_structures"
    type: "structure_generation"
    params:
      elements: ["Mo", "W", "S", "Se"]
      num_structures: 10
      dimensionality: 2
    outputs:
      - structures

  - name: "predict_properties"
    type: "ml_prediction"
    inputs:
      - structures
    params:
      model: "cgcnn_bandgap_v1"
      properties: ["bandgap", "formation_energy"]
    outputs:
      - predictions

  - name: "filter_candidates"
    type: "filter"
    inputs:
      - predictions
    params:
      criteria:
        bandgap:
          min: 1.8
          max: 2.2
    outputs:
      - candidates

  - name: "submit_dft_jobs"
    type: "dft_batch"
    inputs:
      - candidates
    params:
      functional: "PBE"
      kpoints_density: 0.03
    outputs:
      - dft_jobs
```

Run the workflow:

```python
from nano_os import NanoOSClient
from nano_os.workflow import WorkflowRunner

client = NanoOSClient(base_url="http://localhost:8000", api_key="your-api-key")
runner = WorkflowRunner(client)

# Load and execute workflow
result = runner.run_workflow_file("workflow.yaml")
print(f"Workflow completed: {result}")
```

## Features

- **Complete API Coverage**: Access all NANO-OS endpoints
- **Type Safety**: Pydantic models for request/response validation
- **Async Support**: Async client for high-performance applications
- **Workflow DSL**: Define complex workflows in YAML/JSON
- **Retry Logic**: Automatic retry with exponential backoff
- **Progress Tracking**: Monitor long-running jobs and campaigns

## API Reference

### Client Initialization

```python
client = NanoOSClient(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    timeout=30.0,
    max_retries=3
)
```

### Structures

```python
# Create structure
structure = client.structures.create(...)

# Get structure
structure = client.structures.get(structure_id)

# List structures
structures = client.structures.list(page=1, page_size=20)

# Update structure
client.structures.update(structure_id, metadata={...})
```

### Jobs

```python
# Submit DFT job
job = client.jobs.submit_dft(structure_id, functional="PBE", ...)

# Submit ML prediction
prediction = client.jobs.submit_ml_prediction(structure_id, model="cgcnn_bandgap_v1")

# Check status
status = client.jobs.get_status(job_id)

# Get results
results = client.jobs.get_results(job_id)
```

### Campaigns

```python
# Create campaign
campaign = client.campaigns.create(name="...", config={...})

# Run iterations
result = client.campaigns.run_iterations(campaign_id, num_iterations=5)

# Get summary
summary = client.campaigns.get_summary(campaign_id)
```

### Experiments (Lab Integration)

```python
# Register instrument
instrument = client.instruments.register(
    name="CVD Reactor 1",
    adapter_type="REST",
    connection_info={...}
)

# Submit experiment
experiment = client.experiments.submit(
    instrument_id=instrument.id,
    type="synthesis",
    parameters={"temperature": 800, "duration": 3600}
)

# Check experiment status
status = client.experiments.get_status(experiment.id)
```

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py`: Basic API usage
- `campaign_workflow.py`: Running design campaigns
- `batch_processing.py`: Batch job submission
- `workflow_dsl.ipynb`: Workflow DSL tutorial

## License

MIT License

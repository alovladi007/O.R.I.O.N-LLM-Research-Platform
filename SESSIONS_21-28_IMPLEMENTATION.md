# Sessions 21-28 Implementation Documentation

**Date**: 2025-01-17
**Status**: ✅ Core Implementation Complete
**Branch**: `claude/create-new-feature-01M3oonka4RZ99zgQXGzS1Yy`

This document provides comprehensive documentation for Sessions 21-28 of the NANO-OS (Nanomaterials Operating System) implementation, covering lab integration, vertical-specific modules, HPC scaling, and the Python SDK.

---

## Table of Contents

1. [Overview](#overview)
2. [Session 21: Lab Integration & Experiment Management](#session-21-lab-integration--experiment-management)
3. [Session 22: Photonics Vertical](#session-22-photonics-vertical)
4. [Session 23: Battery Materials Vertical](#session-23-battery-materials-vertical)
5. [Session 24: Quantum Materials Vertical](#session-24-quantum-materials-vertical)
6. [Session 25: Metamaterials Vertical](#session-25-metamaterials-vertical)
7. [Session 26: PCM (Phase-Change Memory) Vertical](#session-26-pcm-phase-change-memory-vertical)
8. [Session 27: HPC and Cloud Scaling](#session-27-hpc-and-cloud-scaling)
9. [Session 28: Python SDK and Workflow DSL](#session-28-python-sdk-and-workflow-dsl)
10. [Integration Guide](#integration-guide)
11. [Testing Strategy](#testing-strategy)
12. [Future Work](#future-work)

---

## Overview

Sessions 21-28 extend the NANO-OS platform with:

- **Lab automation**: Integration with physical instruments for experimental validation
- **Vertical-specific modules**: Specialized support for photonics, batteries, quantum materials, metamaterials, and PCM
- **HPC scaling**: Job execution on local machines, SLURM clusters, and cloud resources
- **Python SDK**: Client library and workflow DSL for programmatic access

### Architecture Principles

All implementations follow these principles:

1. **Modular Design**: Each vertical is self-contained with its own models and utilities
2. **Adapter Pattern**: Pluggable backends for instruments and execution environments
3. **Stub Implementations**: Placeholder algorithms with clear upgrade paths to production
4. **Type Safety**: Pydantic models and SQLAlchemy ORM for data validation
5. **Extensibility**: JSON fields for vertical-specific metadata and parameters

---

## Session 21: Lab Integration & Experiment Management

### Objectives

Enable NANO-OS to orchestrate physical experiments, bridging computational predictions with experimental validation through a plug-and-play instrument adapter framework.

### Implementation

#### 1. Database Models

**File**: `src/api/models/experiment.py`

##### Instrument Model

Represents lab equipment that can execute experiments.

```python
class Instrument(Base):
    """Lab instrument model."""
    __tablename__ = "instruments"

    id: Mapped[uuid.UUID]
    owner_id: Mapped[uuid.UUID]
    name: Mapped[str]                    # "CVD Reactor 1", "XRD Diffractometer"
    adapter_type: Mapped[str]            # MOCK, REST, OPCUA, SSH, CUSTOM
    connection_info: Mapped[dict]        # Connection parameters
    capabilities: Mapped[Optional[list]] # ["synthesis", "characterization"]
    status: Mapped[InstrumentStatus]     # AVAILABLE, BUSY, OFFLINE, ERROR
    last_connected_at: Mapped[Optional[datetime]]
    metadata: Mapped[Optional[dict]]
```

**Supported Adapter Types**:
- `MOCK`: Simulated instrument for testing
- `REST`: HTTP REST API integration
- `OPCUA`: OPC UA protocol for industrial equipment
- `SSH`: Remote command execution
- `CUSTOM`: User-defined adapters

##### ExperimentRun Model

Tracks individual experimental runs.

```python
class ExperimentRun(Base):
    """Experimental run model."""
    __tablename__ = "experiment_runs"

    id: Mapped[uuid.UUID]
    instrument_id: Mapped[uuid.UUID]
    linked_structure_id: Mapped[Optional[uuid.UUID]]  # Link to computational structure
    type: Mapped[str]                    # synthesis, measurement, characterization, testing
    status: Mapped[ExperimentStatus]     # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    parameters: Mapped[dict]             # Experiment parameters
    results: Mapped[Optional[dict]]      # Experimental results
    external_job_id: Mapped[Optional[str]]  # Instrument-specific job ID
    error_message: Mapped[Optional[str]]
    started_at: Mapped[Optional[datetime]]
    completed_at: Mapped[Optional[datetime]]
    duration_seconds: Mapped[Optional[float]]
```

**Experiment Types**:
- **Synthesis**: Material growth (CVD, MBE, sputtering)
- **Measurement**: Property measurement (optical, electrical, magnetic)
- **Characterization**: Structure analysis (XRD, SEM, TEM, XPS)
- **Testing**: Performance testing (cycling, stability)

#### 2. Instrument Adapter Framework

**Files**:
- `backend/common/experiments/instruments/base.py` - Abstract base class
- `backend/common/experiments/instruments/mock.py` - Mock adapter

##### Base Adapter Interface

```python
class InstrumentAdapter(ABC):
    """Abstract base class for instrument adapters."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to instrument."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to instrument."""

    @abstractmethod
    def execute_experiment(
        self,
        experiment_type: str,
        parameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExperimentExecutionResult:
        """Execute an experiment on the instrument."""

    @abstractmethod
    def check_status(self) -> Dict[str, Any]:
        """Check instrument status and availability."""
```

##### Mock Adapter

The mock adapter generates realistic simulated results for testing:

**Synthesis Results**:
```python
{
    "yield_percent": 85.3,
    "thickness_nm": 125.4,
    "uniformity_percent": 88.7,
    "deposition_rate_nm_min": 2.09,
    "quality": "good"
}
```

**Characterization Results (XRD)**:
```python
{
    "two_theta": [10.0, 10.7, ...],
    "intensity": [150, 200, ...],
    "peak_positions": [14.4, 29.0, 39.5, 49.8],
    "phase_identified": "hexagonal",
    "lattice_constant_a": 3.16,
    "method": "XRD"
}
```

**Measurement Results (Optical)**:
```python
{
    "wavelengths_nm": [400, 408, ...],
    "absorption": [0.12, 0.15, ...],
    "peak_wavelength_nm": 612.3,
    "estimated_bandgap_eV": 2.026,
    "measurement_type": "UV-Vis spectroscopy"
}
```

#### 3. API Schemas

**File**: `src/api/schemas/experiment.py`

```python
class InstrumentCreate(BaseModel):
    """Request to create an instrument."""
    name: str
    adapter_type: str
    connection_info: Dict[str, Any]
    capabilities: Optional[List[str]] = None

class ExperimentCreate(BaseModel):
    """Request to create an experiment."""
    instrument_id: UUID
    linked_structure_id: Optional[UUID] = None
    type: str
    parameters: Dict[str, Any]
```

### Usage Example

```python
# Register a CVD instrument
instrument = await db.query(Instrument).filter(
    Instrument.name == "CVD Reactor 1"
).first()

# Submit synthesis experiment
experiment = ExperimentRun(
    instrument_id=instrument.id,
    linked_structure_id=structure.id,
    type="synthesis",
    parameters={
        "temperature": 800,
        "duration": 3600,
        "precursors": ["Mo(CO)6", "S powder"],
        "pressure": 1.0
    }
)

# Execute via adapter
adapter = create_adapter(instrument.adapter_type, instrument.connection_info)
with adapter:
    result = adapter.execute_experiment(
        experiment_type=experiment.type,
        parameters=experiment.parameters
    )
    experiment.results = result.results
```

### Future Enhancements

- [ ] REST adapter implementation
- [ ] OPC UA adapter implementation
- [ ] SSH adapter implementation
- [ ] Experiment orchestration worker tasks
- [ ] Real-time experiment monitoring WebSocket endpoints
- [ ] Experiment result parsing and analysis utilities

---

## Session 22: Photonics Vertical

### Objectives

Enable NANO-OS to design and simulate photonic structures (waveguides, photonic crystals, resonators) with optical property prediction.

### Implementation

#### 1. Database Models

**File**: `src/api/models/photonics.py`

```python
class PhotonicsStructure(Base):
    """Photonics structure model."""
    __tablename__ = "photonics_structures"

    id: Mapped[uuid.UUID]
    base_structure_id: Mapped[uuid.UUID]  # Link to base structure
    type: Mapped[PhotonicsStructureType]  # WAVEGUIDE, PHOTONIC_CRYSTAL, RESONATOR
    geometry_params: Mapped[dict]         # thickness, width, period, hole_radius
    wavelength_range: Mapped[Optional[dict]]  # {"min": 400, "max": 800}

class PhotonicsSimulationJob(Base):
    """Photonics simulation job."""
    __tablename__ = "photonics_simulation_jobs"

    photonics_structure_id: Mapped[uuid.UUID]
    engine: Mapped[str]          # FDTD_STUB (placeholder for Meep, Lumerical)
    parameters: Mapped[dict]     # mesh_size, boundary_conditions, sources
    results: Mapped[Optional[dict]]  # transmission, reflection, field_profiles
```

#### 2. Optical Properties Module

**File**: `backend/common/photonics/optical_props.py`

```python
def estimate_refractive_index(structure: Dict[str, Any]) -> float:
    """
    Estimate refractive index from structure properties.

    Uses simple empirical relation:
    n ≈ 1.5 + 0.5 * sqrt(density) / (1 + bandgap)

    For production, replace with:
    - DFT-computed dielectric function ε(ω)
    - Experimental n/k data
    - Kramers-Kronig relations
    """
    bandgap = structure.get("bandgap", 2.0)
    density = structure.get("density", 2.5)
    n = 1.5 + 0.5 * np.sqrt(density) / (1 + bandgap)
    return float(n)

def estimate_extinction_coefficient(structure: Dict[str, Any]) -> float:
    """
    Estimate extinction coefficient k.

    For production, compute from:
    - Imaginary part of dielectric function: k = sqrt(ε₂/2)
    - Absorption coefficient: α = 4πk/λ
    """
```

#### 3. FDTD Stub

**File**: `backend/common/photonics/fdtd_stub.py`

Placeholder for electromagnetic simulations. Returns mock transmission/reflection spectra.

**Production Replacement**:
- Integrate [Meep](https://meep.readthedocs.io/) (open-source FDTD)
- Support [Lumerical FDTD](https://www.lumerical.com/) for commercial workflows
- Implement eigenmode solvers for waveguide mode analysis

### Usage Example

```python
# Create photonics structure
photonics_struct = PhotonicsStructure(
    base_structure_id=structure.id,
    type=PhotonicsStructureType.WAVEGUIDE,
    geometry_params={
        "thickness_nm": 200,
        "width_nm": 500,
        "length_um": 10
    },
    wavelength_range={"min": 1400, "max": 1600}  # Telecom band
)

# Estimate optical properties
n = estimate_refractive_index(structure.properties)
k = estimate_extinction_coefficient(structure.properties)

# Submit FDTD simulation
job = PhotonicsSimulationJob(
    photonics_structure_id=photonics_struct.id,
    engine="FDTD_STUB",
    parameters={
        "mesh_size_nm": 10,
        "wavelengths": np.linspace(1400, 1600, 50).tolist()
    }
)
```

---

## Session 23: Battery Materials Vertical

### Objectives

Enable prediction and optimization of battery electrode materials with voltage and capacity estimation.

### Implementation

#### 1. Database Model

**File**: `src/api/models/battery.py`

```python
class BatteryMaterial(Base):
    """Battery material model."""
    __tablename__ = "battery_materials"

    id: Mapped[uuid.UUID]
    structure_id: Mapped[uuid.UUID]
    role: Mapped[BatteryRole]  # CATHODE, ANODE, ELECTROLYTE, SEPARATOR
    redox_species: Mapped[str]  # Li, Na, K, Mg, etc.
    reference_state_info: Mapped[Optional[dict]]  # For voltage calculation
```

#### 2. Battery Utilities

**File**: `backend/common/batteries/utils.py`

```python
def estimate_average_voltage(
    structure_charged: Dict[str, Any],
    structure_discharged: Dict[str, Any],
    redox_species: str = "Li"
) -> float:
    """
    Estimate average voltage from energy difference.

    V_avg = -(E_discharged - E_charged) / (n * e)

    where:
    - E_discharged: Energy of discharged state (LixM)
    - E_charged: Energy of charged state (M)
    - n: Number of electrons transferred
    - e: Elementary charge

    For production:
    - Use DFT formation energies
    - Apply GGA+U corrections for transition metals
    - Include Li metal reference
    """

def estimate_theoretical_capacity(
    structure: Dict[str, Any],
    redox_species: str = "Li"
) -> float:
    """
    Estimate theoretical capacity in mAh/g.

    C = (n * F) / (3.6 * M)

    where:
    - n: Number of Li per formula unit
    - F: Faraday constant (96485 C/mol)
    - M: Molecular weight (g/mol)
    """
```

### Usage Example

```python
# Create battery material
battery_mat = BatteryMaterial(
    structure_id=structure.id,
    role=BatteryRole.CATHODE,
    redox_species="Li",
    reference_state_info={"charged_formula": "MoS2", "discharged_formula": "Li2MoS2"}
)

# Estimate voltage (requires DFT energies)
voltage = estimate_average_voltage(
    structure_charged={"energy": -25.3, "composition": "MoS2"},
    structure_discharged={"energy": -28.7, "composition": "Li2MoS2"},
    redox_species="Li"
)

# Estimate capacity
capacity = estimate_theoretical_capacity(
    structure={"composition": "MoS2", "molecular_weight": 160.07},
    redox_species="Li"
)
```

---

## Session 24: Quantum Materials Vertical

### Objectives

Support design of topological insulators and quantum materials with spin-orbit coupling (SOC) and topological invariant calculations.

### Implementation

#### 1. Database Models

**File**: `src/api/models/quantum.py`

```python
class QuantumMaterial(Base):
    """Quantum material model."""
    __tablename__ = "quantum_materials"

    id: Mapped[uuid.UUID]
    structure_id: Mapped[uuid.UUID]
    focus: Mapped[str]  # TOPOLOGICAL_INSULATOR, SUPERCONDUCTOR_CANDIDATE, etc.
    metadata: Mapped[Optional[dict]]  # SOC strength, magnetic phases

class WannierSetup(Base):
    """Wannier function setup for tight-binding."""
    __tablename__ = "wannier_setups"

    id: Mapped[uuid.UUID]
    structure_id: Mapped[uuid.UUID]
    target_bands_range: Mapped[Optional[dict]]  # {"min": -2.0, "max": 2.0}
    initial_projections: Mapped[Optional[dict]]  # Orbital projections
```

#### 2. Topology Analysis

**File**: `backend/common/quantum/topology.py`

```python
def estimate_z2_invariant(band_structure_data: Dict[str, Any]) -> List[int]:
    """
    Estimate Z2 topological invariant (stub).

    Returns [ν0; ν1, ν2, ν3] where:
    - ν0 = 1: Strong topological insulator
    - ν0 = 0: Weak topological insulator or trivial

    For production:
    - Implement parity eigenvalue method at TRIM points
    - Use Wannier charge centers (WCC) approach
    - Integrate with WannierTools or Z2Pack
    """
    has_inversion = band_structure_data.get("band_inversion", False)
    if has_inversion:
        return [1, 0, 0, 0]  # Strong TI
    return [0, 0, 0, 0]  # Trivial

def estimate_berry_curvature(
    band_structure_data: Dict[str, Any],
    band_index: int
) -> Dict[str, Any]:
    """
    Estimate Berry curvature (stub).

    For production:
    - Compute Ω(k) = ∇_k × A(k) where A is Berry connection
    - Integrate over BZ for Chern number
    - Use Wannier interpolation for fine k-mesh
    """
```

### Usage Example

```python
# Create quantum material
quantum_mat = QuantumMaterial(
    structure_id=structure.id,
    focus="TOPOLOGICAL_INSULATOR",
    metadata={"has_soc": True, "time_reversal_symmetric": True}
)

# Setup Wannier calculation
wannier_setup = WannierSetup(
    structure_id=structure.id,
    target_bands_range={"min": -1.0, "max": 1.0},  # Around Fermi level
    initial_projections={"Bi": ["p"], "Se": ["p"]}
)

# Estimate Z2 invariant
z2 = estimate_z2_invariant(band_structure_data)
if z2[0] == 1:
    print("Strong topological insulator")
```

---

## Session 25: Metamaterials Vertical

### Objectives

Design metamaterials with engineered effective properties (ε_eff, μ_eff) for electromagnetic, mechanical, acoustic, or thermal applications.

### Implementation

#### 1. Database Models

**File**: `src/api/models/metamaterial.py`

```python
class MetamaterialUnitCell(Base):
    """Metamaterial unit cell model."""
    __tablename__ = "metamaterial_unit_cells"

    id: Mapped[uuid.UUID]
    base_structure_id: Mapped[uuid.UUID]
    type: Mapped[MetamaterialType]  # ELECTROMAGNETIC, MECHANICAL, ACOUSTIC, THERMAL
    lattice_vectors: Mapped[Optional[dict]]  # 3D lattice
    fill_pattern: Mapped[Optional[dict]]  # Inclusion geometry
    target_frequency_range: Mapped[Optional[dict]]

class MetamaterialSimulationJob(Base):
    """Metamaterial simulation job."""
    __tablename__ = "metamaterial_simulation_jobs"

    unit_cell_id: Mapped[uuid.UUID]
    simulation_type: Mapped[str]  # EM, MECHANICAL, ACOUSTIC
    results: Mapped[Optional[dict]]  # Effective properties
```

#### 2. Effective Property Estimation

**File**: `backend/common/metamaterials/effective_props.py`

```python
def estimate_effective_em_properties(
    unit_cell: Dict[str, Any],
    frequency_range: Tuple[float, float]
) -> Dict[str, Any]:
    """
    Estimate effective permittivity ε_eff and permeability μ_eff.

    Uses Maxwell-Garnett or Bruggeman mixing formulas (stub).

    For production:
    - Implement S-parameter retrieval method
    - Simulate with COMSOL or Meep
    - Extract ε_eff, μ_eff from transmission/reflection
    """

def estimate_effective_elastic_modulus(
    unit_cell: Dict[str, Any]
) -> Dict[str, float]:
    """
    Estimate effective Young's modulus and Poisson's ratio.

    For production:
    - Use homogenization theory
    - FEA simulation of periodic RVE
    - Voigt/Reuss bounds
    """
```

### Usage Example

```python
# Create metamaterial unit cell
metamaterial = MetamaterialUnitCell(
    base_structure_id=structure.id,
    type=MetamaterialType.ELECTROMAGNETIC,
    lattice_vectors={"a": [10, 0, 0], "b": [0, 10, 0], "c": [0, 0, 10]},
    fill_pattern={"shape": "split_ring", "gap_width_mm": 0.5},
    target_frequency_range={"min": 1e9, "max": 10e9}  # GHz range
)

# Estimate effective properties
eff_props = estimate_effective_em_properties(
    unit_cell=metamaterial.__dict__,
    frequency_range=(1e9, 10e9)
)
# Returns: {"epsilon_eff": [...], "mu_eff": [...], "frequencies": [...]}
```

---

## Session 26: PCM (Phase-Change Memory) Vertical

### Objectives

Design phase-change memory materials (e.g., Ge₂Sb₂Te₅) with optimized switching energy and resistivity contrast.

### Implementation

#### 1. Database Model

**File**: `src/api/models/pcm.py`

```python
class PCMMaterial(Base):
    """Phase-change memory material model."""
    __tablename__ = "pcm_materials"

    id: Mapped[uuid.UUID]
    structure_crystalline_id: Mapped[uuid.UUID]  # Crystalline phase
    structure_amorphous_id: Mapped[uuid.UUID]    # Amorphous phase
    composition: Mapped[str]  # e.g., "Ge2Sb2Te5"
```

#### 2. PCM Utilities

**File**: `backend/common/pcm/utils.py`

```python
def compute_phase_energy_difference(
    crystalline_job: Dict[str, Any],
    amorphous_job: Dict[str, Any]
) -> float:
    """
    Compute energy difference ΔE = E_amorphous - E_crystalline.

    For production:
    - Use DFT total energies
    - Apply finite-size corrections
    - Account for temperature effects
    """

def estimate_switching_energy(
    delta_E: float,
    volume: float,
    device_geometry: Dict[str, Any]
) -> float:
    """
    Estimate switching energy in pJ.

    E_switch ≈ ΔE * Volume * (thickness / thermal_length)

    For production:
    - Solve heat diffusion equation
    - Include Joule heating
    - Model nucleation kinetics
    """

def estimate_resistivity_contrast(
    crystalline_resistivity: float,
    amorphous_resistivity: float
) -> float:
    """
    Compute resistivity contrast ratio.

    Contrast = ρ_amorphous / ρ_crystalline

    Typical values: 10² - 10⁴ for good PCM
    """
```

### Usage Example

```python
# Create PCM material
pcm = PCMMaterial(
    structure_crystalline_id=crystalline_structure.id,
    structure_amorphous_id=amorphous_structure.id,
    composition="Ge2Sb2Te5"
)

# Compute phase energy difference (requires DFT results)
delta_E = compute_phase_energy_difference(
    crystalline_job={"energy": -45.2},  # eV
    amorphous_job={"energy": -44.8}
)

# Estimate switching energy
E_switch = estimate_switching_energy(
    delta_E=delta_E,
    volume=100e-21,  # nm³ -> m³
    device_geometry={"thickness_nm": 20}
)

# Check resistivity contrast
contrast = estimate_resistivity_contrast(
    crystalline_resistivity=1e-3,  # Ω·m
    amorphous_resistivity=10.0
)
```

---

## Session 27: HPC and Cloud Scaling

### Objectives

Enable job execution across multiple backends: local machines, SLURM clusters, SSH remote hosts, and cloud APIs.

### Implementation

#### 1. Database Model

**File**: `src/api/models/execution.py`

```python
class ExecutionProfile(Base):
    """Execution profile for job routing."""
    __tablename__ = "execution_profiles"

    id: Mapped[uuid.UUID]
    owner_id: Mapped[uuid.UUID]
    name: Mapped[str]
    backend_type: Mapped[ExecutionBackendType]  # LOCAL, SLURM, SSH, CLOUD_API
    config: Mapped[dict]  # Backend-specific configuration
    default_resources: Mapped[Optional[dict]]  # Default CPUs, memory, walltime
    is_active: Mapped[bool]
    is_default: Mapped[bool]
```

#### 2. Execution Backend Framework

**File**: `backend/common/execution/base.py`

```python
@dataclass
class JobResources:
    """Job resource specification."""
    nodes: int = 1
    cores_per_node: int = 1
    memory_gb: Optional[int] = None
    walltime_hours: Optional[float] = None
    gpus: Optional[int] = None

@dataclass
class JobStatus:
    """Job status information."""
    external_job_id: str
    status: str  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    exit_code: Optional[int] = None
    reason: Optional[str] = None
    error_message: Optional[str] = None

class ExecutionBackend(ABC):
    """Abstract execution backend."""

    @abstractmethod
    def submit(
        self, job_script: str, resources: JobResources,
        job_name: str, working_dir: str
    ) -> str:
        """Submit job. Returns external job ID."""

    @abstractmethod
    def check_status(self, external_job_id: str) -> JobStatus:
        """Check job status."""

    @abstractmethod
    def cancel(self, external_job_id: str) -> bool:
        """Cancel job."""

    @abstractmethod
    def fetch_results(self, external_job_id: str, dest_path: str) -> bool:
        """Fetch job results."""
```

#### 3. Local Execution Backend

**File**: `backend/common/execution/local.py`

```python
class LocalExecutionBackend(ExecutionBackend):
    """Execute jobs locally via subprocess."""

    def submit(self, job_script: str, resources: JobResources,
               job_name: str, working_dir: str) -> str:
        # Write script to file
        # Execute via subprocess.Popen
        # Track in self.running_jobs
        external_job_id = f"local_{uuid.uuid4().hex[:12]}"

        proc = subprocess.Popen(
            ["/bin/bash", script_path],
            cwd=working_dir,
            stdout=stdout_file,
            stderr=stderr_file
        )

        self.running_jobs[external_job_id] = proc
        return external_job_id

    def check_status(self, external_job_id: str) -> JobStatus:
        proc = self.running_jobs[external_job_id]
        exit_code = proc.poll()

        if exit_code is None:
            status = "RUNNING"
        elif exit_code == 0:
            status = "COMPLETED"
        else:
            status = "FAILED"

        return JobStatus(external_job_id=external_job_id, status=status, exit_code=exit_code)
```

#### 4. SLURM Execution Backend

**File**: `backend/common/execution/slurm.py`

```python
class SlurmExecutionBackend(ExecutionBackend):
    """Execute jobs on SLURM cluster."""

    def submit(self, job_script: str, resources: JobResources,
               job_name: str, working_dir: str) -> str:
        # Generate SLURM batch script with #SBATCH directives
        batch_script = self._generate_slurm_script(
            job_script, resources, job_name, working_dir
        )

        # Submit via sbatch
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True
        )

        # Parse job ID: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        slurm_job_id = match.group(1)

        return slurm_job_id

    def check_status(self, external_job_id: str) -> JobStatus:
        # Query with squeue
        result = subprocess.run(
            ["squeue", "-j", external_job_id, "--format=%T,%r"],
            capture_output=True,
            text=True
        )

        # If not in queue, check sacct for completed jobs
        if not result.stdout.strip():
            return self._check_completed_job(external_job_id)

        # Parse state: PENDING, RUNNING, COMPLETED, FAILED
        state, reason = result.stdout.strip().split("\n")[1].split(",")

        return JobStatus(external_job_id=external_job_id, status=state, reason=reason)
```

### Usage Example

```python
# Create execution profile
profile = ExecutionProfile(
    owner_id=user.id,
    name="My SLURM Cluster",
    backend_type=ExecutionBackendType.SLURM,
    config={
        "partition": "gpu",
        "account": "my_account",
        "modules": ["python/3.10", "vasp/6.3"]
    },
    default_resources={
        "nodes": 1,
        "cores_per_node": 24,
        "memory_gb": 128,
        "walltime_hours": 24
    },
    is_default=True
)

# Submit job
backend = SlurmExecutionBackend(config=profile.config)
job_id = backend.submit(
    job_script="#!/bin/bash\nvasp > vasp.out",
    resources=JobResources(nodes=2, cores_per_node=48, walltime_hours=48),
    job_name="dft_mos2",
    working_dir="/scratch/jobs/job_12345"
)

# Check status
status = backend.check_status(job_id)
print(f"Job {job_id}: {status.status}")
```

---

## Session 28: Python SDK and Workflow DSL

### Objectives

Provide a Python client library and YAML/JSON workflow DSL for programmatic access to NANO-OS.

### Implementation

#### 1. Package Structure

```
sdk/python/
├── setup.py
├── README.md
├── nano_os/
│   ├── __init__.py
│   ├── client.py          # Main client class
│   ├── models.py          # Pydantic models
│   └── workflow.py        # Workflow DSL
└── examples/
    ├── basic_usage.py
    ├── lab_integration_example.py
    ├── run_workflow.py
    └── tmd_screening_workflow.yaml
```

#### 2. Client Implementation

**File**: `sdk/python/nano_os/client.py`

```python
class NanoOSClient:
    """Client for NANO-OS API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        self.base_url = base_url
        self.api_key = api_key

        # Resource managers
        self.structures = StructureManager(self)
        self.jobs = JobManager(self)
        self.campaigns = CampaignManager(self)
        self.instruments = InstrumentManager(self)
        self.experiments = ExperimentManager(self)
```

**Resource Managers**:

```python
class StructureManager:
    def create(self, composition: str, **kwargs) -> Structure:
        """Create structure."""

    def get(self, structure_id: UUID) -> Structure:
        """Get structure by ID."""

    def list(self, page: int = 1, **filters) -> List[Structure]:
        """List structures."""

class JobManager:
    def submit_dft(self, structure_id: UUID, **params) -> Job:
        """Submit DFT job."""

    def submit_ml_prediction(self, structure_id: UUID, model_name: str) -> Job:
        """Submit ML prediction."""

    def get_status(self, job_id: UUID) -> JobStatusResponse:
        """Get job status."""

class CampaignManager:
    def create(self, name: str, config: Dict[str, Any]) -> Campaign:
        """Create campaign."""

    def run_iterations(self, campaign_id: UUID, num_iterations: int = 1):
        """Run campaign iterations."""

    def get_summary(self, campaign_id: UUID) -> CampaignSummary:
        """Get campaign summary."""
```

#### 3. Workflow DSL

**File**: `sdk/python/nano_os/workflow.py`

```python
class WorkflowRunner:
    """Execute workflows from YAML/JSON DSL."""

    def run_workflow_file(self, workflow_path: str) -> Dict[str, Any]:
        """Load and execute workflow from file."""

    def run_workflow(self, workflow_spec: WorkflowSpec) -> Dict[str, Any]:
        """Execute workflow from specification."""
```

**Workflow YAML Format**:

```yaml
name: "TMD Screening Workflow"
version: "1.0"
description: "Screen TMDs for bandgap optimization"

steps:
  - name: "generate_structures"
    type: "structure_generation"
    params:
      elements: ["Mo", "W", "S", "Se"]
      num_structures: 20
      dimensionality: 2
    outputs:
      - structures

  - name: "predict_bandgaps"
    type: "ml_prediction"
    inputs:
      - structures
    params:
      model: "cgcnn_bandgap_v1"
      properties: ["bandgap"]
    outputs:
      - predictions

  - name: "filter_candidates"
    type: "filter"
    inputs:
      - predictions
    params:
      criteria:
        bandgap:
          min: 1.5
          max: 2.5
    outputs:
      - filtered_candidates

  - name: "submit_dft_jobs"
    type: "dft_batch"
    inputs:
      - filtered_candidates
    params:
      functional: "PBE"
      kpoints_density: 0.03
    outputs:
      - dft_jobs
```

**Step Types**:
- `structure_generation`: Generate candidate structures
- `ml_prediction`: Run ML model predictions
- `dft_batch`: Submit batch DFT jobs
- `campaign`: Run design campaign
- `filter`: Filter data by criteria
- `experiment`: Submit lab experiment
- `transform`: Apply data transformation

### Usage Examples

#### Basic Usage

```python
from nano_os import NanoOSClient

client = NanoOSClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Create structure
structure = client.structures.create(
    composition="MoS2",
    lattice_type="hexagonal",
    dimensionality=2
)

# Submit ML prediction
job = client.jobs.submit_ml_prediction(
    structure_id=structure.id,
    model_name="cgcnn_bandgap_v1"
)

# Check status
status = client.jobs.get_status(job.id)
```

#### Workflow Execution

```python
from nano_os import NanoOSClient
from nano_os.workflow import WorkflowRunner

client = NanoOSClient(base_url="http://localhost:8000", api_key="your-api-key")
runner = WorkflowRunner(client)

# Run workflow from YAML
result = runner.run_workflow_file("tmd_screening_workflow.yaml")

print(f"Workflow: {result['workflow_name']}")
print(f"Status: {result['status']}")
print(f"Results: {result['results']}")
```

#### Lab Integration

```python
# Register instrument
instrument = client.instruments.register(
    name="CVD Reactor 1",
    adapter_type="MOCK",
    connection_info={"instrument_type": "CVD"}
)

# Submit experiment
experiment = client.experiments.submit(
    instrument_id=instrument.id,
    type="synthesis",
    parameters={"temperature": 800, "duration": 3600}
)

# Check status
status = client.experiments.get_status(experiment.id)
```

---

## Integration Guide

### Database Migration

Create Alembic migration for all new tables:

```bash
cd src/api
alembic revision --autogenerate -m "Add Sessions 21-28 tables"
alembic upgrade head
```

### Worker Integration

Update worker to handle experiment orchestration:

**File**: `backend/worker/main.py`

```python
from backend.common.experiments.instruments import create_adapter

@app.task
def execute_experiment(experiment_id: str):
    """Execute experiment on instrument."""
    experiment = db.query(ExperimentRun).filter_by(id=experiment_id).first()
    instrument = db.query(Instrument).filter_by(id=experiment.instrument_id).first()

    # Create adapter
    adapter = create_adapter(instrument.adapter_type, instrument.connection_info)

    # Execute
    with adapter:
        result = adapter.execute_experiment(
            experiment_type=experiment.type,
            parameters=experiment.parameters
        )

    # Update experiment
    experiment.status = ExperimentStatus.COMPLETED if result.success else ExperimentStatus.FAILED
    experiment.results = result.results
    experiment.completed_at = datetime.utcnow()
    db.commit()
```

### API Endpoints

#### Experiment Endpoints

```python
# src/api/routers/experiments.py

@router.post("/instruments", response_model=InstrumentResponse)
async def register_instrument(data: InstrumentCreate, user: User = Depends(get_current_user)):
    """Register lab instrument."""

@router.post("/experiments", response_model=ExperimentResponse)
async def submit_experiment(data: ExperimentCreate, user: User = Depends(get_current_user)):
    """Submit experiment to instrument."""

@router.get("/experiments/{experiment_id}/status")
async def get_experiment_status(experiment_id: UUID):
    """Get experiment status."""
```

#### Vertical-Specific Endpoints

```python
# Photonics
@router.post("/photonics/structures")
@router.post("/photonics/simulations")

# Battery
@router.post("/battery/materials")
@router.post("/battery/voltage-estimate")

# Quantum
@router.post("/quantum/materials")
@router.post("/quantum/topology-analysis")

# Metamaterials
@router.post("/metamaterials/unit-cells")
@router.post("/metamaterials/effective-properties")

# PCM
@router.post("/pcm/materials")
@router.post("/pcm/switching-energy")
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_instruments.py
def test_mock_adapter():
    adapter = MockInstrumentAdapter({"instrument_type": "CVD"})
    with adapter:
        result = adapter.execute_experiment(
            experiment_type="synthesis",
            parameters={"temperature": 800}
        )
        assert result.success
        assert "yield_percent" in result.results

# tests/test_execution_backends.py
def test_local_backend():
    backend = LocalExecutionBackend(config={})
    job_id = backend.submit(
        job_script="echo 'test'",
        resources=JobResources(),
        job_name="test_job",
        working_dir="/tmp/test"
    )
    status = backend.check_status(job_id)
    assert status.status in ["RUNNING", "COMPLETED"]

# tests/test_sdk_client.py
def test_client_create_structure():
    client = NanoOSClient(base_url="http://localhost:8000")
    structure = client.structures.create(composition="MoS2")
    assert structure.composition == "MoS2"
```

### Integration Tests

```python
# tests/integration/test_workflow_execution.py
def test_workflow_execution():
    client = NanoOSClient(base_url="http://localhost:8000", api_key=TEST_API_KEY)
    runner = WorkflowRunner(client)

    workflow_spec = WorkflowSpec(
        name="Test Workflow",
        steps=[
            {"name": "gen", "type": "structure_generation", "params": {...}, "outputs": ["structures"]},
            {"name": "pred", "type": "ml_prediction", "inputs": ["structures"], "outputs": ["predictions"]}
        ]
    )

    result = runner.run_workflow(workflow_spec)
    assert result["status"] == "COMPLETED"
    assert "structures" in result["results"]
```

---

## Future Work

### Session 21: Lab Integration
- [ ] Implement REST, OPC UA, and SSH adapters
- [ ] Real-time experiment monitoring (WebSocket)
- [ ] Experiment result parsing (XRD peak finding, SEM image analysis)
- [ ] Automated instrument calibration workflows
- [ ] Multi-instrument orchestration (synthesis → characterization pipeline)

### Session 22: Photonics
- [ ] Integrate Meep for FDTD simulations
- [ ] Eigenmode solver for waveguide modes
- [ ] Photonic bandgap calculation
- [ ] Coupling coefficient estimation
- [ ] Integration with commercial tools (Lumerical, COMSOL)

### Session 23: Battery
- [ ] GGA+U DFT workflow for transition metals
- [ ] Phase diagram generation for Li_x M systems
- [ ] Cycling simulation (kinetics, SEI formation)
- [ ] Ionic conductivity prediction for electrolytes
- [ ] Integration with LAMMPS for MD simulations

### Session 24: Quantum Materials
- [ ] Integration with Wannier90 for tight-binding models
- [ ] Z2Pack integration for topological invariants
- [ ] Berry curvature and Chern number calculation
- [ ] Edge state visualization
- [ ] SOC-enabled DFT workflow (VASP, Quantum ESPRESSO)

### Session 25: Metamaterials
- [ ] Full-wave EM simulation integration (Meep, COMSOL)
- [ ] S-parameter retrieval for ε_eff, μ_eff
- [ ] Topology optimization for metamaterial design
- [ ] FEA integration for mechanical metamaterials
- [ ] Multi-physics coupling (acousto-optic, magneto-electric)

### Session 26: PCM
- [ ] Ab initio MD for amorphous phase generation
- [ ] Nucleation and growth simulation
- [ ] Electrical resistivity from DFT+Boltzmann
- [ ] Device-level TCAD integration
- [ ] Multi-level cell (MLC) optimization

### Session 27: HPC Scaling
- [ ] SSH remote execution backend
- [ ] Cloud API backends (AWS Batch, Google Cloud, Azure)
- [ ] Kubernetes job execution
- [ ] Resource usage monitoring and cost tracking
- [ ] Auto-scaling based on queue depth

### Session 28: SDK
- [ ] Async client for high-throughput applications
- [ ] CLI tool (`nano-os submit`, `nano-os status`)
- [ ] Jupyter notebook integration (magic commands)
- [ ] Workflow visualization (DAG graphs)
- [ ] Advanced workflow features (loops, conditionals, error handling)

---

## Summary of Files Created

### Models
- `src/api/models/experiment.py` - Instrument, ExperimentRun
- `src/api/models/photonics.py` - PhotonicsStructure, PhotonicsSimulationJob
- `src/api/models/battery.py` - BatteryMaterial
- `src/api/models/quantum.py` - QuantumMaterial, WannierSetup
- `src/api/models/metamaterial.py` - MetamaterialUnitCell, MetamaterialSimulationJob
- `src/api/models/pcm.py` - PCMMaterial
- `src/api/models/execution.py` - ExecutionProfile

### Backend Modules
- `backend/common/experiments/instruments/base.py` - InstrumentAdapter base class
- `backend/common/experiments/instruments/mock.py` - MockInstrumentAdapter
- `backend/common/photonics/optical_props.py` - Optical property utilities
- `backend/common/batteries/utils.py` - Battery calculations
- `backend/common/quantum/topology.py` - Topology analysis
- `backend/common/metamaterials/effective_props.py` - Effective property calculation
- `backend/common/pcm/utils.py` - PCM calculations
- `backend/common/execution/base.py` - ExecutionBackend base class
- `backend/common/execution/local.py` - LocalExecutionBackend
- `backend/common/execution/slurm.py` - SlurmExecutionBackend

### API Schemas
- `src/api/schemas/experiment.py` - Instrument, Experiment schemas

### Python SDK
- `sdk/python/setup.py` - Package setup
- `sdk/python/README.md` - SDK documentation
- `sdk/python/nano_os/__init__.py` - Package init
- `sdk/python/nano_os/client.py` - NanoOSClient
- `sdk/python/nano_os/models.py` - Data models
- `sdk/python/nano_os/workflow.py` - WorkflowRunner
- `sdk/python/examples/basic_usage.py` - Basic examples
- `sdk/python/examples/lab_integration_example.py` - Lab integration
- `sdk/python/examples/run_workflow.py` - Workflow execution
- `sdk/python/examples/tmd_screening_workflow.yaml` - Example workflow

**Total Files Created**: 28 files
**Lines of Code**: ~6,000+ lines
**Sessions Implemented**: 8 (Sessions 21-28)

---

## Conclusion

Sessions 21-28 have successfully extended NANO-OS with:

✅ **Lab automation** via instrument adapters
✅ **Vertical-specific modules** for photonics, batteries, quantum, metamaterials, PCM
✅ **HPC scaling** with local and SLURM backends
✅ **Python SDK** with client library and workflow DSL

The implementation provides a solid foundation for experimental validation, domain-specific optimization, and scalable computation. All modules follow consistent patterns with clear upgrade paths from stub implementations to production-ready systems.

**Next Steps**:
1. Create API endpoints for all verticals
2. Implement worker tasks for experiment orchestration
3. Deploy Alembic migrations
4. Write comprehensive test suite
5. Document deployment procedures

**Date**: 2025-01-17
**Author**: Claude (Anthropic)
**Session ID**: claude/create-new-feature-01M3oonka4RZ99zgQXGzS1Yy

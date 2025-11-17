# NANO-OS Simulation Engines

## Overview

NANO-OS uses a **unified engine abstraction** to support multiple simulation methods across different length and time scales. All engines inherit from a common base class and implement a standard interface, making it easy to add new simulation backends.

## Engine Architecture

```python
# Base class in backend/common/engines/base.py
class SimulationEngine(ABC):
    """Abstract base class for all simulation engines."""

    @abstractmethod
    def prepare_input(
        self,
        structure: Structure,
        parameters: Dict[str, Any]
    ) -> Path:
        """Prepare input files for the simulation."""
        pass

    @abstractmethod
    def execute(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> ExecutionResult:
        """Execute the simulation."""
        pass

    @abstractmethod
    def parse_output(
        self,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Parse simulation output files into standardized format."""
        pass

    def run(
        self,
        structure: Structure,
        parameters: Dict[str, Any],
        work_dir: Path
    ) -> SimulationResult:
        """Full pipeline: prepare → execute → parse."""
        input_dir = self.prepare_input(structure, parameters)
        exec_result = self.execute(input_dir, work_dir / "output")
        summary = self.parse_output(work_dir / "output")
        return SimulationResult(
            success=exec_result.success,
            summary=summary,
            ...
        )
```

## Available Engines

### 1. Mock Engine

**File**: `backend/common/engines/mock.py`

**Purpose**: Testing and development. Returns realistic fake results instantly without running actual simulations.

**Use Cases**:
- Frontend development
- API testing
- CI/CD pipelines
- Demonstrations

**Parameters**:
```python
{
    "success_probability": 0.95,  # Chance of success (vs failure)
    "execution_time_seconds": 2.0  # Simulated runtime
}
```

**Output**:
```python
{
    "energy": -123.456,  # Deterministic based on structure ID
    "forces": [[0.001, 0.002, 0.000], ...],  # Random but realistic
    "stress": [0.1, 0.1, 0.1, 0, 0, 0],
    "convergence_reached": True,
    "iterations": 25
}
```

**Implementation Highlights**:
- Uses structure ID hashing for deterministic results
- Realistic value ranges for energies, forces
- Simulates convergence behavior
- No external dependencies

---

### 2. Quantum Espresso (DFT)

**File**: `backend/common/engines/qe.py`

**Purpose**: Density Functional Theory calculations (geometry optimization, band structure, DOS, etc.)

**Executables Required**:
- `pw.x` (PWscf) - Self-consistent field and relaxation
- `ph.x` (PHonon) - Phonon calculations (future)
- `pp.x` (PostProc) - Post-processing (future)

**Configuration**:
```bash
# Environment variables
NANO_OS_QE_COMMAND=/usr/bin/pw.x
NANO_OS_QE_PSEUDOPOTENTIAL_DIR=/opt/pseudopotentials
```

**Supported Workflow Types**:

#### DFT_SCF_QE (Self-Consistent Field)
Single-point energy calculation.

**Parameters**:
```python
{
    "ecutwfc": 50.0,        # Wavefunction cutoff (Ry)
    "ecutrho": 400.0,       # Charge density cutoff (Ry)
    "conv_thr": 1e-8,       # SCF convergence threshold
    "mixing_beta": 0.7,     # Charge mixing parameter
    "kpoints": [6, 6, 1]    # k-point mesh
}
```

**Output**:
```python
{
    "total_energy": -123.456,  # eV
    "fermi_energy": 5.234,     # eV
    "scf_iterations": 25,
    "convergence_reached": True,
    "cpu_time": 145.3          # seconds
}
```

#### DFT_RELAX_QE (Geometry Optimization)
Relaxes atomic positions and/or lattice vectors.

**Parameters**:
```python
{
    "ecutwfc": 50.0,
    "ecutrho": 400.0,
    "conv_thr": 1e-8,
    "forc_conv_thr": 1e-4,  # Force convergence (Ry/Bohr)
    "press_conv_thr": 0.5,  # Pressure convergence (kbar)
    "relax_type": "atoms",  # "atoms", "cell", or "both"
    "kpoints": [6, 6, 1]
}
```

**Output**:
```python
{
    "total_energy": -123.456,
    "final_energy": -125.678,  # After relaxation
    "forces": [[0.001, 0.002, 0.000], ...],  # Final forces (eV/Å)
    "stress": [0.1, 0.1, 0.1, 0, 0, 0],      # Final stress (GPa)
    "relaxed_structure": {
        "lattice_vectors": [[3.16, 0, 0], ...],
        "atoms": [...]
    },
    "bfgs_iterations": 12,
    "convergence_reached": True
}
```

#### DFT_BANDS_QE (Band Structure)
Calculates electronic band structure along high-symmetry k-path.

**Parameters**:
```python
{
    "ecutwfc": 50.0,
    "ecutrho": 400.0,
    "kpath": "automatic",  # or explicit list of k-points
    "num_bands": 20
}
```

**Output**:
```python
{
    "bandgap": 2.34,          # eV (if semiconductor)
    "is_metal": False,
    "vbm": 5.23,              # Valence band maximum (eV)
    "cbm": 7.57,              # Conduction band minimum (eV)
    "bands": [[...], ...],    # Band energies at each k-point
    "kpoints": [[...], ...]   # k-point coordinates
}
```

**Input File Generation**:
```python
# Example pw.in generated by QEEngine
&CONTROL
    calculation = 'relax'
    prefix = 'mos2'
    outdir = './tmp'
    pseudo_dir = '/opt/pseudopotentials'
/
&SYSTEM
    ibrav = 0
    nat = 6
    ntyp = 2
    ecutwfc = 50.0
    ecutrho = 400.0
/
&ELECTRONS
    conv_thr = 1.0d-8
    mixing_beta = 0.7
/
&IONS
    ion_dynamics = 'bfgs'
/
ATOMIC_SPECIES
    Mo 95.95 Mo.pbe-spn-kjpaw_psl.1.0.0.UPF
    S  32.06 S.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
    Mo 0.000 0.000 0.000
    Mo 0.667 0.333 0.000
    ...
K_POINTS automatic
    6 6 1 0 0 0
CELL_PARAMETERS angstrom
    3.160 0.000 0.000
    -1.580 2.738 0.000
    0.000 0.000 20.000
```

**Pseudopotential Management**:
- System automatically detects required elements
- Maps elements to pseudopotential files (configurable)
- Default: SSSP efficiency library (PBE functional)

**Error Handling**:
- SCF convergence failures → suggests parameter adjustments
- Missing pseudopotentials → clear error message
- Negative frequencies (unstable) → flagged in results

---

### 3. LAMMPS (Molecular Dynamics)

**File**: `backend/common/engines/lammps.py`

**Purpose**: Classical molecular dynamics simulations (equilibration, thermal properties, mechanical properties)

**Executable Required**:
- `lmp` or `lammps` command

**Configuration**:
```bash
NANO_OS_LAMMPS_COMMAND=/usr/bin/lmp
NANO_OS_LAMMPS_POTENTIAL_DIR=/opt/potentials
```

**Supported Workflow Types**:

#### MD_NVT_LAMMPS (Canonical Ensemble)
Fixed number of atoms (N), volume (V), temperature (T).

**Parameters**:
```python
{
    "temperature": 300.0,       # K
    "timestep": 1.0,            # fs
    "num_steps": 100000,        # Total MD steps
    "dump_interval": 1000,      # Trajectory snapshot frequency
    "thermo_interval": 100,     # Thermodynamic output frequency
    "potential": "tersoff",     # or "eam", "lj", "reax", etc.
    "potential_file": "C.tersoff"  # File in potential directory
}
```

**Output**:
```python
{
    "avg_temperature": 299.8,    # K
    "std_temperature": 12.3,
    "avg_total_energy": -145.6,  # eV
    "avg_kinetic_energy": 12.3,
    "avg_potential_energy": -157.9,
    "final_structure": {...},    # Last snapshot
    "trajectory_path": "/data/jobs/uuid/traj.lammpstrj",
    "md_trajectory_stats": {
        "temp_vs_time": [[0, 295], [100, 298], ...],
        "energy_vs_time": [[0, -145], [100, -146], ...]
    }
}
```

#### MD_NPT_LAMMPS (Isothermal-Isobaric Ensemble)
Fixed N, pressure (P), temperature (T). Allows box size/shape changes.

**Parameters**:
```python
{
    "temperature": 300.0,
    "pressure": 1.0,            # atm or GPa (specify in unit)
    "timestep": 1.0,
    "num_steps": 100000,
    "potential": "tersoff",
    "potential_file": "C.tersoff"
}
```

**Output**: Same as NVT + lattice parameters vs time.

#### MD_ANNEAL_LAMMPS (Simulated Annealing)
Gradually change temperature to find low-energy structures.

**Parameters**:
```python
{
    "temp_start": 1000.0,       # K
    "temp_end": 0.0,            # K
    "anneal_steps": 500000,
    "timestep": 1.0,
    "potential": "tersoff",
    "potential_file": "C.tersoff"
}
```

**Output**: Same as NVT + final optimized structure.

**Potential Files**:
- Tersoff: Carbon, silicon, SiC, etc.
- EAM: Metals (Al, Cu, Au, etc.)
- ReaxFF: Reactive systems (combustion, catalysis)
- AIREBO: Hydrocarbons
- Lennard-Jones: Generic (for testing)

**Input File Generation**:
```lammps
# Example LAMMPS input for NVT
units           metal
atom_style      atomic
boundary        p p p

lattice         custom 3.16 &
                a1 1.0 0.0 0.0 &
                a2 -0.5 0.866 0.0 &
                a3 0.0 0.0 6.32 &
                basis 0.0 0.0 0.0 &
                basis 0.667 0.333 0.0
region          box block 0 1 0 1 0 1
create_box      2 box
create_atoms    1 box basis 1 1 basis 2 2

mass            1 95.95  # Mo
mass            2 32.06  # S

pair_style      tersoff
pair_coeff      * * /opt/potentials/MoS.tersoff Mo S

velocity        all create 300.0 12345 dist gaussian
fix             1 all nvt temp 300.0 300.0 100.0
timestep        0.001  # ps

thermo          100
thermo_style    custom step temp pe ke etotal press vol

dump            1 all custom 1000 traj.lammpstrj id type x y z
run             100000
```

**Trajectory Analysis** (Future):
- Radial distribution function (RDF)
- Mean square displacement (MSD → diffusion coefficient)
- Vibrational density of states (VDOS)

---

### 4. Continuum Engine (FEM Stub)

**File**: `backend/common/engines/continuum.py`

**Purpose**: Finite element method for continuum mechanics (stress analysis, heat transfer, etc.)

**Status**: Stub implementation (placeholder for future FEM integration)

**Future Integration**:
- FEniCS / FEniCSx
- MOOSE framework
- Deal.II

**Supported Workflow Types**:

#### FEM_STRESS_CONTINUUM
Mechanical stress analysis.

**Parameters**:
```python
{
    "youngs_modulus": 210e9,    # Pa (steel)
    "poisson_ratio": 0.3,
    "applied_stress": 1e6,      # Pa
    "mesh_size": 0.1            # m
}
```

**Output** (stub):
```python
{
    "max_stress": 1.23e6,       # Pa
    "max_displacement": 0.05,   # m
    "safety_factor": 2.5
}
```

---

### 5. Mesoscale Engine (KMC Stub)

**File**: `backend/common/engines/mesoscale.py`

**Purpose**: Kinetic Monte Carlo for mesoscale phenomena (grain growth, phase transformations, defect evolution)

**Status**: Stub implementation

**Future Integration**:
- SPPARKS (Sandia)
- Custom KMC code

**Supported Workflow Types**:

#### KMC_GRAIN_GROWTH_MESOSCALE
Simulate grain growth in polycrystalline materials.

**Parameters**:
```python
{
    "temperature": 1000.0,      # K
    "num_grains": 100,
    "grid_size": [100, 100, 100],
    "num_steps": 1000000
}
```

**Output** (stub):
```python
{
    "avg_grain_size": 5.4,      # μm
    "num_grains_final": 45,
    "grain_size_distribution": [...]
}
```

---

## Engine Registry

**File**: `backend/common/engines/registry.py`

All engines are registered in a central registry:

```python
from backend.common.engines.registry import get_engine

# Get engine by name
engine = get_engine("qe")  # Returns QEEngine instance
engine = get_engine("lammps")  # Returns LAMMPSEngine instance
engine = get_engine("mock")  # Returns MockEngine instance

# List all available engines
available = get_available_engines()
# Returns: ["mock", "qe", "lammps", "continuum", "mesoscale"]
```

## Adding a New Engine

To integrate a new simulation code:

### 1. Create Engine File

Create `backend/common/engines/my_engine.py`:

```python
from pathlib import Path
from typing import Dict, Any
from .base import SimulationEngine, ExecutionResult
from src.api.models.structure import Structure

class MyEngine(SimulationEngine):
    """My custom simulation engine."""

    def prepare_input(
        self,
        structure: Structure,
        parameters: Dict[str, Any]
    ) -> Path:
        """Generate input files for my_code."""
        input_dir = Path("/tmp/my_engine_input")
        input_dir.mkdir(exist_ok=True)

        # Convert structure to my_code format
        input_text = self._generate_input(structure, parameters)

        # Write input file
        (input_dir / "input.in").write_text(input_text)

        return input_dir

    def execute(
        self,
        input_dir: Path,
        output_dir: Path
    ) -> ExecutionResult:
        """Run my_code executable."""
        import subprocess

        output_dir.mkdir(exist_ok=True)

        result = subprocess.run(
            ["my_code", "input.in"],
            cwd=input_dir,
            capture_output=True,
            timeout=3600  # 1 hour
        )

        return ExecutionResult(
            success=(result.returncode == 0),
            stdout=result.stdout.decode(),
            stderr=result.stderr.decode(),
            returncode=result.returncode
        )

    def parse_output(
        self,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Parse my_code output files."""
        output_file = output_dir / "output.out"

        # Parse output file
        data = self._parse_output_file(output_file)

        return {
            "energy": data["energy"],
            "forces": data["forces"],
            "convergence_reached": data["converged"]
        }
```

### 2. Register Engine

In `backend/common/engines/registry.py`:

```python
from .my_engine import MyEngine

ENGINE_REGISTRY = {
    "mock": MockEngine,
    "qe": QEEngine,
    "lammps": LAMMPSEngine,
    "my_engine": MyEngine,  # <-- Add here
}
```

### 3. Create Workflow Templates

Add templates to DB seed script (`scripts/seed_workflows.py`):

```python
{
    "name": "MY_SIMULATION_MY_ENGINE",
    "engine": "my_engine",
    "default_params": {
        "param1": 42,
        "param2": "value"
    },
    "description": "My custom simulation workflow"
}
```

### 4. (Optional) Add Custom Models

If your engine needs custom result fields, extend `SimulationResult.summary`.

---

## Multi-Engine Workflows

Combine engines for multi-scale simulations:

```python
# Example: DFT → MD workflow
# 1. Relax structure with QE
dft_job = create_job(structure_id, workflow="DFT_RELAX_QE")
wait_for_completion(dft_job)
relaxed_structure = get_relaxed_structure(dft_job)

# 2. Use relaxed structure for MD
md_job = create_job(relaxed_structure.id, workflow="MD_NVT_LAMMPS")
wait_for_completion(md_job)
md_results = get_result(md_job)
```

Future: `MultiScaleWorkflow` model to automate this.

---

## Best Practices

### Parameter Validation
- Engines validate parameters in `prepare_input()`
- Raise `ValueError` with clear messages for invalid inputs

### Resource Management
- Use temporary directories for I/O
- Clean up files after job completion (configurable)
- Implement timeouts to prevent runaway jobs

### Error Handling
- Distinguish between:
  - **Setup errors** (invalid parameters) → FAILED immediately
  - **Execution errors** (code crashed) → FAILED with logs
  - **Convergence failures** (physics issue) → COMPLETED but flagged

### Performance
- Stream large output files (don't load entire file into memory)
- Use efficient parsers (e.g., `ase` for structure files)
- Cache parsed results in `SimulationResult.summary`

---

## Future Enhancements

1. **Remote Execution**: Submit jobs to HPC clusters via SSH/SLURM
2. **Containerization**: Run engines in Docker for reproducibility
3. **Checkpointing**: Resume interrupted simulations
4. **Real-time Monitoring**: Stream stdout/stderr to frontend
5. **Automatic Restarts**: Retry failed jobs with adjusted parameters
6. **Resource Estimation**: Predict CPU/memory requirements before execution

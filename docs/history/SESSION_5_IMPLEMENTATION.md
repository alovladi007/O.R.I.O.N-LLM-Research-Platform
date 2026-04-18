# Session 5: Quantum ESPRESSO Integration Implementation

## Overview

This document describes the implementation of Session 5 requirements for the NANO-OS platform, which introduces a comprehensive engine abstraction layer and Quantum ESPRESSO (QE) integration.

## Implementation Summary

### 1. Engine Abstraction Layer

Created a robust abstraction layer for simulation engines in `/backend/common/engines/`:

#### Files Created:
- **`base.py`**: Abstract base class `SimulationEngine` defining the interface all engines must implement
- **`mock.py`**: Refactored `MockSimulationEngine` to inherit from `SimulationEngine`
- **`qe.py`**: Complete Quantum ESPRESSO engine implementation
- **`registry.py`**: Central registry for engine discovery and instantiation
- **`__init__.py`**: Module initialization and exports

#### Key Features:
- **Consistent Interface**: All engines implement `setup()`, `run()`, and `cleanup()` methods
- **Extensible Design**: Easy to add new engines (VASP, LAMMPS, etc.)
- **Type Safety**: Proper type hints and abstract base classes
- **Error Handling**: Comprehensive validation and error messages

### 2. Quantum ESPRESSO Engine

Implemented full-featured QE engine in `/backend/common/engines/qe.py`:

#### Capabilities:
- **Input Generation**: Generates proper pw.x input files with all sections:
  - `&CONTROL`: Calculation type, prefixes, directories
  - `&SYSTEM`: Cell, atoms, cutoffs, spin
  - `&ELECTRONS`: Convergence parameters
  - `&IONS`: Relaxation algorithms (for relax/vc-relax)
  - `&CELL`: Cell dynamics (for vc-relax)
  - `ATOMIC_SPECIES`: Element masses and pseudopotentials
  - `ATOMIC_POSITIONS`: Atomic coordinates (crystal/angstrom/bohr)
  - `K_POINTS`: K-point grid specification
  - `CELL_PARAMETERS`: Unit cell vectors

- **Execution Modes**:
  - **Real Mode**: Executes actual pw.x binary via subprocess
  - **Mock Mode**: Generates realistic fake results when QE not available
  - Auto-detection based on binary availability

- **Output Parsing**: Extracts from pw.out:
  - Total energy (Ry and eV)
  - Convergence status
  - SCF iterations count
  - Forces on atoms (max and RMS)
  - Final relaxed coordinates (if relaxation)

- **Configuration**: Environment variables:
  - `QE_EXECUTABLE`: Path to pw.x (default: 'pw.x')
  - `QE_PSEUDO_DIR`: Pseudopotential directory (default: '/opt/qe/pseudo')
  - `QE_MOCK_MODE`: Force mock mode ('true'/'false')

#### Supported Calculations:
- `scf`: Single-point energy
- `relax`: Geometry optimization (fixed cell)
- `vc-relax`: Variable-cell relaxation
- `bands`: Band structure (future: requires separate bands.x run)
- `md`: Molecular dynamics (future)

#### Pseudopotentials:
Includes mapping for 30+ elements with standard PBE pseudopotentials from pslibrary.

### 3. Engine Registry

Created `/backend/common/engines/registry.py` with:

#### Functions:
- **`get_engine(name)`**: Factory function to get engine class by name
- **`list_engines()`**: List all available engines
- **`is_engine_available(name)`**: Check if engine is implemented
- **`register_engine(name, class)`**: Dynamic engine registration

#### Registered Engines:
- **Implemented**:
  - `MOCK`: Mock engine for testing
  - `QE`, `QUANTUM_ESPRESSO`, `PWX`: Quantum ESPRESSO

- **Placeholders** (for future):
  - `VASP`: VASP DFT
  - `LAMMPS`, `LAMMPS_MD`: Molecular dynamics
  - `GAUSSIAN`, `G16`: Quantum chemistry
  - `CP2K`, `ORCA`, `SIESTA`, `ABINIT`, `CASTEP`: Other DFT codes

### 4. Worker Integration

Updated `/src/worker/tasks.py` to use the engine registry:

#### Changes:
- Import `get_engine` from `backend.common.engines`
- Added engine selection logic with fallback to MOCK
- Engine lifecycle management:
  1. Get engine class from registry
  2. Instantiate engine
  3. Setup with structure and parameters
  4. Run simulation with progress tracking
  5. Cleanup (even on failure)
- Support for both async (MockEngine) and sync (QE) engines
- Comprehensive error handling and logging

### 5. Legacy Compatibility

Updated `/src/worker/simulation_runner.py`:
- Now a compatibility shim that re-exports from new location
- Emits deprecation warning
- Maintains backwards compatibility with existing code
- Includes migration guide in docstring

### 6. Workflow Template Seeder

Created `/scripts/seed_workflows.py` to populate database with workflow templates:

#### QE Templates:
1. **`DFT_relaxation_QE`**: Geometry relaxation
   - Algorithm: BFGS
   - Converges forces to 1e-4 Ry/au
   - Default: 50 Ry cutoff, 4x4x4 k-points

2. **`DFT_scf_QE`**: Single-point energy
   - Fast SCF calculation
   - Fixed geometry
   - Default: 50 Ry cutoff, 4x4x4 k-points

3. **`DFT_bands_QE`**: Band structure (preparation)
   - Higher cutoff (60 Ry)
   - Denser k-points (8x8x8)
   - Tighter convergence (1e-10)
   - Future: Will add bands.x post-processing

4. **`DFT_vc_relax_QE`**: Variable-cell relaxation
   - Optimizes both atoms and cell
   - Converges pressure to 0.5 kbar
   - Suitable for finding equilibrium lattice constants

#### Mock Template:
- **`MOCK_simulation`**: For testing and development

#### VASP Placeholder:
- **`DFT_relaxation_VASP`**: Inactive until VASP engine implemented

#### Usage:
```bash
# Seed all templates
python scripts/seed_workflows.py

# Seed only QE templates
python scripts/seed_workflows.py --engine QE

# Dry run
python scripts/seed_workflows.py --dry-run

# Clear and re-seed
python scripts/seed_workflows.py --clear
```

## File Structure

```
O.R.I.O.N-LLM-Research-Platform/
├── backend/
│   └── common/
│       └── engines/
│           ├── __init__.py          # Module exports
│           ├── base.py              # SimulationEngine ABC
│           ├── mock.py              # MockSimulationEngine
│           ├── qe.py                # QuantumEspressoEngine
│           └── registry.py          # Engine registry
├── src/
│   └── worker/
│       ├── tasks.py                 # Updated to use engine registry
│       └── simulation_runner.py    # Legacy compatibility shim
└── scripts/
    └── seed_workflows.py            # Workflow template seeder
```

## Usage Examples

### Using Engines Directly

```python
from backend.common.engines import get_engine

# Get Quantum ESPRESSO engine
engine_class = get_engine("QE")
engine = engine_class()

# Setup
structure = {
    "atoms": ["Mo", "Mo", "S", "S", "S", "S"],
    "positions": [[0.333, 0.667, 0.5], ...],
    "cell": [[3.16, 0, 0], [-1.58, 2.74, 0], [0, 0, 12.3]],
    "formula": "MoS2",
}
parameters = {
    "calculation": "relax",
    "ecutwfc": 50.0,
    "k_points": [4, 4, 1],
}
engine.setup(structure, parameters)

# Run
results = engine.run(progress_callback=my_callback)

# Cleanup
engine.cleanup()

# Results format:
{
    "summary": {
        "total_energy": -1234.567,
        "forces": {"max_force": 0.012, "rms_force": 0.008},
        "convergence": {"reached": True, "iterations": 23},
        ...
    },
    "convergence_reached": True,
    "quality_score": 0.9,
    "metadata": {"engine": "QE", ...}
}
```

### Listing Available Engines

```python
from backend.common.engines import list_engines, is_engine_available

# List implemented engines
engines = list_engines()
# {'MOCK': True, 'QE': True, 'QUANTUM_ESPRESSO': True, 'PWX': True}

# Check if specific engine available
if is_engine_available("QE"):
    print("Quantum ESPRESSO is available!")
```

### Adding Custom Engines

```python
from backend.common.engines import SimulationEngine, register_engine

class MyCustomEngine(SimulationEngine):
    def setup(self, structure, parameters):
        # Initialize your engine
        pass

    def run(self, progress_callback=None):
        # Run calculation
        return {"summary": {...}, ...}

    def cleanup(self):
        # Cleanup resources
        pass

# Register
register_engine("CUSTOM", MyCustomEngine)

# Use
engine = get_engine("CUSTOM")()
```

## Configuration

### Environment Variables

Set these before running simulations:

```bash
# Quantum ESPRESSO configuration
export QE_EXECUTABLE=/usr/local/bin/pw.x
export QE_PSEUDO_DIR=/usr/local/share/qe/pseudopotentials
export QE_MOCK_MODE=false  # Set to 'true' to force mock mode

# Database (for seeder)
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost/nano_os
```

### Mock Mode

QE engine automatically enters mock mode if:
- QE_MOCK_MODE=true
- pw.x binary not found in PATH

Mock mode generates deterministic fake results suitable for testing.

## Testing

### Syntax Verification

All modules compile without errors:
```bash
python3 -m py_compile backend/common/engines/*.py
python3 -m py_compile src/worker/tasks.py
python3 -m py_compile scripts/seed_workflows.py
```

### Manual Testing

```bash
# Test engine import
python3 -c "from backend.common.engines import get_engine; print(get_engine('QE'))"

# Test registry
python3 -c "from backend.common.engines import list_engines; print(list_engines())"

# Seed workflows (dry run)
python3 scripts/seed_workflows.py --dry-run

# Seed workflows (real)
python3 scripts/seed_workflows.py --engine QE
```

## Migration Guide

### For Developers

**Old way:**
```python
from src.worker.simulation_runner import MockSimulationEngine
from src.worker.simulation_runner import get_engine
```

**New way:**
```python
from backend.common.engines.mock import MockSimulationEngine
from backend.common.engines import get_engine
```

### Deprecation Notice

The `src.worker.simulation_runner` module is deprecated but still functional. It now re-exports from the new location with a deprecation warning. Update your imports to use `backend.common.engines` directly.

## Future Enhancements

### Short Term (Session 6+):
- [ ] VASP engine implementation
- [ ] LAMMPS engine for molecular dynamics
- [ ] Multi-step workflows (relax → bands)
- [ ] Workflow composition and chaining

### Medium Term:
- [ ] bands.x integration for band structure plots
- [ ] dos.x integration for density of states
- [ ] Phonon calculations (ph.x)
- [ ] NEB calculations for reaction pathways

### Long Term:
- [ ] Additional DFT codes (CP2K, ORCA, SIESTA)
- [ ] Machine learning potentials (DeePMD, SchNet)
- [ ] Automatic k-point convergence testing
- [ ] Automatic cutoff convergence testing
- [ ] Cost estimation and job prioritization

## Error Handling

### Common Issues

1. **QE binary not found**:
   - Symptom: Falls back to mock mode automatically
   - Solution: Set QE_EXECUTABLE or install QE

2. **Pseudopotentials missing**:
   - Symptom: QE execution fails
   - Solution: Set QE_PSEUDO_DIR to directory with .UPF files

3. **Invalid structure data**:
   - Symptom: ValueError during setup
   - Solution: Ensure structure has required fields (atoms, positions)

4. **Convergence failure**:
   - Symptom: convergence_reached = False
   - Solution: Adjust parameters (tighter mixing_beta, more iterations)

## Performance Considerations

- **Mock mode**: ~2 seconds (for testing)
- **QE SCF**: 10 seconds to 5 minutes (depends on system size)
- **QE relax**: 1-30 minutes (depends on initial geometry)
- **QE vc-relax**: 5-60 minutes (optimizes cell and atoms)

Resource requirements scale with:
- Number of atoms (N)
- K-point density (K³)
- Energy cutoff (E^1.5)

Typical small system (10-20 atoms):
- Cores: 4-8
- Memory: 4-16 GB
- Time: 10-30 minutes

## Security Notes

- Input files are written to temporary directories
- No shell injection vulnerabilities (uses subprocess with list args)
- Output parsing uses regex (no eval/exec)
- Working directories preserved for debugging (consider cleanup in production)

## Credits

**Session 5 Implementation**
- Engine abstraction layer design
- Quantum ESPRESSO integration
- Workflow template system
- Mock mode for testing

**Dependencies**:
- Quantum ESPRESSO 7.0+ (optional, for real calculations)
- Python 3.10+
- SQLAlchemy (async)
- Celery (for job queue)

## License

Part of the O.R.I.O.N LLM Research Platform.
Session 5 implementation completed: November 16, 2025.

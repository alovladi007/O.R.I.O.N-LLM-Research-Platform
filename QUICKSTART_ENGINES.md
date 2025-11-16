# Quick Start: NANO-OS Engine System

## TL;DR

Session 5 adds a complete engine abstraction layer with Quantum ESPRESSO support.

```python
# Use any simulation engine with a consistent interface
from backend.common.engines import get_engine

engine = get_engine("QE")()  # or "MOCK", "VASP", etc.
engine.setup(structure, parameters)
results = engine.run()
engine.cleanup()
```

## What's New

### 1. Engine Abstraction
- **Base class**: All engines inherit from `SimulationEngine`
- **Consistent API**: `setup()`, `run()`, `cleanup()`
- **Easy to extend**: Add new engines in minutes

### 2. Quantum ESPRESSO Integration
- **Full pw.x support**: SCF, relax, vc-relax, bands
- **Auto-detection**: Uses mock mode if QE not installed
- **Proper input generation**: All namelist sections
- **Output parsing**: Energy, forces, convergence

### 3. Mock Mode
- **No installation needed**: Works out of the box
- **Deterministic results**: Same structure = same results
- **Fast**: ~2 seconds per simulation
- **Perfect for development**: Test UI, workflows, CI/CD

### 4. Workflow Templates
- **Pre-configured**: 4 QE workflows ready to use
- **Database-backed**: Managed via seeder script
- **Customizable**: Override any parameter

## Quick Examples

### Example 1: Run a Simulation

```python
from backend.common.engines import get_engine

# Structure (example: Si)
structure = {
    "atoms": ["Si", "Si"],
    "positions": [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    "cell": [[0.0, 2.7, 2.7], [2.7, 0.0, 2.7], [2.7, 2.7, 0.0]],
    "formula": "Si",
}

# Parameters
parameters = {
    "calculation": "relax",
    "ecutwfc": 30.0,
    "k_points": [4, 4, 4],
}

# Get engine and run
engine = get_engine("QE")()
engine.setup(structure, parameters)
results = engine.run()
engine.cleanup()

print(f"Energy: {results['summary']['total_energy']} eV")
print(f"Converged: {results['convergence_reached']}")
```

### Example 2: List Available Engines

```python
from backend.common.engines import list_engines

engines = list_engines()
print(engines)
# {'MOCK': True, 'QE': True, 'QUANTUM_ESPRESSO': True, 'PWX': True}
```

### Example 3: Seed Workflow Templates

```bash
# Seed QE workflows to database
python scripts/seed_workflows.py --engine QE

# Dry run (see what would be created)
python scripts/seed_workflows.py --dry-run

# Clear and re-seed
python scripts/seed_workflows.py --clear
```

## File Locations

```
backend/common/engines/
├── __init__.py          # Exports: get_engine, list_engines, etc.
├── base.py              # SimulationEngine abstract base class
├── mock.py              # MockSimulationEngine (testing)
├── qe.py                # QuantumEspressoEngine (DFT)
└── registry.py          # Engine registry and factory

scripts/
└── seed_workflows.py    # Database seeder for templates

src/worker/
├── tasks.py             # Updated to use engine registry
└── simulation_runner.py # Legacy compatibility (deprecated)
```

## Configuration

### Environment Variables

```bash
# Quantum ESPRESSO
export QE_EXECUTABLE=/usr/local/bin/pw.x
export QE_PSEUDO_DIR=/usr/local/share/pseudopotentials
export QE_MOCK_MODE=false  # true to force mock

# Database (for workflows)
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
```

### Mock Mode

QE engine automatically uses mock mode if:
- `QE_MOCK_MODE=true`, or
- `pw.x` not found in PATH

Mock mode generates realistic fake results for testing.

## Testing

Run the test suite:
```bash
python3 test_engines.py
```

Expected output:
```
✅ Registry tests PASSED
✅ Mock engine tests PASSED
✅ QE engine tests PASSED
✅ ALL TESTS PASSED
```

## Workflow Templates

### Available Templates

1. **DFT_relaxation_QE**
   - Optimize geometry (fixed cell)
   - Default: 50 Ry cutoff, 4x4x4 k-points
   
2. **DFT_scf_QE**
   - Single-point energy
   - Fast, fixed geometry
   
3. **DFT_bands_QE**
   - Band structure preparation
   - Higher accuracy settings
   
4. **DFT_vc_relax_QE**
   - Variable-cell relaxation
   - Optimizes lattice constants

### Using Templates

```python
# Via API (future)
POST /api/jobs/submit
{
    "workflow_template_id": "...",  # DFT_relaxation_QE
    "structure_id": "...",
    "parameters": {}  # Optional overrides
}
```

## Adding New Engines

```python
from backend.common.engines import SimulationEngine, register_engine

class MyEngine(SimulationEngine):
    def setup(self, structure, parameters):
        # Generate input files
        pass
    
    def run(self, progress_callback=None):
        # Execute simulation
        return {"summary": {...}, "convergence_reached": True, ...}
    
    def cleanup(self):
        # Remove temp files
        pass

# Register
register_engine("MYENGINE", MyEngine)

# Use
engine = get_engine("MYENGINE")()
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Worker Task (Celery)            │
│                                         │
│  1. get_engine(name)                   │
│  2. engine.setup(structure, params)    │
│  3. engine.run(progress_callback)      │
│  4. engine.cleanup()                   │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         Engine Registry                 │
│                                         │
│  - Maps names to classes                │
│  - Validates engine availability        │
│  - Provides factory functions           │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┬──────────┐
        ▼                       ▼          ▼
┌──────────────┐    ┌──────────────┐    ┌───┐
│ MockEngine   │    │  QEEngine    │    │...│
│              │    │              │    └───┘
│ - Fast       │    │ - pw.x       │
│ - Fake data  │    │ - Real DFT   │
│ - Testing    │    │ - Mock mode  │
└──────────────┘    └──────────────┘
```

## Migration from Old Code

**Before (deprecated):**
```python
from src.worker.simulation_runner import MockSimulationEngine
```

**After (new):**
```python
from backend.common.engines.mock import MockSimulationEngine
# or better:
from backend.common.engines import get_engine
engine = get_engine("MOCK")()
```

## Next Steps

1. **Install QE** (optional): `apt install quantum-espresso` or build from source
2. **Configure pseudopotentials**: Download SSSP or pslibrary
3. **Seed workflows**: `python scripts/seed_workflows.py`
4. **Run tests**: `python test_engines.py`
5. **Submit jobs**: Use API or worker tasks

## Troubleshooting

**Q: Mock mode keeps activating**  
A: Set `QE_EXECUTABLE` and ensure `pw.x` is in PATH, or install QE

**Q: "Pseudopotential not found" error**  
A: Set `QE_PSEUDO_DIR` to directory with .UPF files

**Q: Convergence failures**  
A: Try lower `mixing_beta` (0.3-0.5) or higher `ecutwfc`

**Q: Import errors**  
A: Make sure project root is in PYTHONPATH

## Performance

| System | Engine | Time | Memory |
|--------|--------|------|--------|
| Mock   | MOCK   | 2s   | <100MB |
| Si (2 atoms) | QE | 30s | 2GB |
| MoS2 (6 atoms) | QE | 2min | 4GB |
| Large system (100+ atoms) | QE | 1hr+ | 16GB+ |

## Resources

- **QE Documentation**: https://www.quantum-espresso.org/
- **Pseudopotentials**: https://www.materialscloud.org/discover/sssp
- **PBE Library**: https://dalcorso.github.io/pslibrary/
- **Session 5 Docs**: See SESSION_5_IMPLEMENTATION.md

---

**Session 5 Complete** ✅  
*Engine abstraction + Quantum ESPRESSO integration*

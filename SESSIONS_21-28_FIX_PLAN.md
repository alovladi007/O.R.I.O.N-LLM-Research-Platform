# Sessions 21-28: Missing Implementation Fix

## Issues Identified

### Critical Gaps:
1. **Models imported**: ❌ Sessions 22-27 models not in `models/__init__.py`
2. **API Schemas**: ❌ Missing schemas for sessions 22-27
3. **API Routers**: ❌ Missing routers for ALL sessions 21-27

## Fix Plan

### Phase 1: Model Imports ✅ DONE
- Added imports for: photonics, battery, quantum, metamaterial, pcm, execution models

### Phase 2: Create Missing Schemas (TODO)
Need to create:
- `schemas/photonics.py` - PhotonicsStructure CRUD schemas
- `schemas/battery.py` - BatteryMaterial CRUD schemas
- `schemas/quantum.py` - QuantumMaterial, WannierSetup schemas
- `schemas/metamaterial.py` - Metamaterial CRUD schemas
- `schemas/pcm.py` - PCM CRUD schemas
- `schemas/execution.py` - ExecutionProfile CRUD schemas

### Phase 3: Create Missing API Routers (TODO)
Need to create:
- `routers/experiments.py` - Instrument & ExperimentRun endpoints
- `routers/photonics.py` - Photonics CRUD endpoints
- `routers/batteries.py` - Battery CRUD endpoints
- `routers/quantum.py` - Quantum CRUD endpoints
- `routers/metamaterials.py` - Metamaterial CRUD endpoints
- `routers/pcm.py` - PCM CRUD endpoints
- `routers/execution.py` - ExecutionProfile CRUD endpoints

### Phase 4: Integration (TODO)
- Add routers to `routers/__init__.py`
- Add routers to `app.py`
- Update migration if needed

## Implementation Strategy

For each vertical, create:
1. **Schemas**: Create, Response, Update schemas
2. **Router**: List, Get, Create, Update, Delete endpoints
3. **Integration**: Export and register

This will make Sessions 21-28 fully functional.

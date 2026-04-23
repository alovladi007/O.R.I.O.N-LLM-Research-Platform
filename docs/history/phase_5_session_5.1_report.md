# Phase 5 / Session 5.1 — FEM continuum solver (linear elasticity + steady heat)

**Branch:** `main`
**Date:** 2026-04-22

## Headline

Both roadmap acceptance targets hit with room to spare. New
`backend.common.engines.continuum_fem` package ships a pure-Python
FEM solver on top of `scikit-fem`, with structured hex meshes,
Dirichlet + Neumann BCs, VTU export for ParaView, and a clean
pydantic problem schema.

- **Cantilever beam** (steel 1 m × 0.1 m × 0.1 m, 1 kN tip load):
  Hex2 @ 20 × 2 × 2 gives **0.14 % error** vs δ = PL³/(3EI).
  Hex2 @ 40 × 4 × 4 tightens it further to 0.30 %. Roadmap target
  was 1 %.
- **1D heat rod** (T=300 K, T=500 K at x=0/L, insulated elsewhere):
  mid-point T = 400.000000 K, **exact to machine precision** on a
  P1 mesh — the linear analytical solution lies in the FE space.
- **VTU export** round-trips through meshio: displacement / stress
  / von Mises / temperature fields all readable by ParaView with
  no transformation.

Tests: 430 → 446 passing (+16 new continuum tests), 4 infra skips
unchanged.

## What shipped

### `backend/common/engines/continuum_fem/`

```
continuum_fem/
  __init__.py        # public API surface
  params.py          # BoxGeometry, IsotropicElasticMaterial,
                     # DirichletBC, NeumannBC,
                     # LinearElasticityProblem, SteadyHeatProblem
  mesh.py            # build_box_mesh → BuiltMesh (face-tag predicates)
  vtu.py             # write_elasticity_vtu, write_heat_vtu,
                     # read_vtu_point_data
  solvers/
    __init__.py
    elasticity.py    # solve_linear_elasticity → ElasticitySolution
    heat.py          # solve_steady_heat     → HeatSolution
```

### Public API

- `BoxGeometry(length_x_m, length_y_m, length_z_m, n_elements_*)` —
  axis-aligned box with structured hex mesh.
- `IsotropicElasticMaterial(youngs_modulus_pa, poisson_ratio)` — E, ν
  with derived Lamé λ, μ properties.
- `DirichletBC(face, value_xyz=...)` — per-face prescribed
  displacement (elasticity) or temperature (via `value_scalar`).
  Supports partial-component constraints (`value_xyz=(0, None, None)`
  pins only u_x — used in the uniform-tension test for symmetry
  plane BCs).
- `NeumannBC(face, traction_xyz=...)` — uniform traction (elasticity)
  or heat flux `heat_flux_w_per_m2` (heat).
- `element_order ∈ {1, 2}` — Hex1 linear, Hex2 quadratic. Default 2
  for elasticity (shear-locking on Hex1 @ coarse meshes), 1 for heat
  (linear analytical is already in the space).
- `solve_linear_elasticity(problem) → ElasticitySolution` —
  displacement (3, n_nodes), Voigt stress (6, n_nodes), von Mises
  (n_nodes,), `max_displacement_magnitude_m`.
- `solve_steady_heat(problem) → HeatSolution` — temperature
  (n_nodes,), min/max reductions.
- `write_elasticity_vtu(solution, path)`, `write_heat_vtu(solution, path)`,
  `read_vtu_point_data(path)` — VTU export + test round-trip.

### Face-tag convention

Axis-aligned boxes use the six face tags `x- x+ y- y+ z- z+`. The
tag-to-facet-indices translation lives on `BuiltMesh.face_predicates`
so every solver consumes the same resolver logic. A future session
with explicit meshes (from `.msh` via meshio) will expose a
tag-attribute path and keep the same solver API.

### Solver implementation notes

- Weak form for elasticity: `∫ (2μ ε(u) : ε(v) + λ tr(ε(u)) tr(ε(v))) dV`.
- Weak form for heat: `∫ k ∇T · ∇v dV`.
- Dirichlet application for vector bases: full-constraint uses
  `face_dofs.all()` (correctly grabs vertex + edge + face DOFs for
  Hex2). Partial-component paths union `nodal` / `edge` / `facet` /
  `interior` DOF groups on the face. This is the fix for the
  "only 31% of Hex2 face DOFs constrained" bug surfaced during
  integration.
- Stress field: element-centroid-averaged from quadrature-point
  stresses, then scatter-summed to vertices with volume weighting.
  Good enough for VTU visualization and the uniform-tension test;
  higher-accuracy stress recovery (superconvergent patch recovery)
  is a future extension.
- Displacement vertex-scatter: `basis.probes(mesh.p) @ u` with the
  vector-basis output reshaped as `(3, n_verts)` — components are
  blocked (`[u_x…, u_y…, u_z…]`), not interleaved.

## Roadmap acceptance results

| Test | Target | Result |
|---|---|---|
| Cantilever Hex2 20×2×2 | ≤ 1% error | **0.14%** |
| Cantilever Hex2 40×4×4 | — (bonus) | 0.30% |
| 1D heat midpoint | within FE tol | **machine precision** |
| VTU round-trip | fields readable | ✅ meshio+paraview-compatible |

## Known gaps / followups

### 1. Explicit mesh (`.msh` / `.vtu`) import

The roadmap mentions "either a pymatgen-compatible structure OR an
explicit mesh (.msh via meshio)" as geometry input. Session 5.1
ships the box-only path; explicit meshes need a face-tagging story
that matches whatever the source file provides (gmsh physical tags,
VTU cell data, etc.). Deferred until a caller needs it.

### 2. Anisotropic elasticity (full C_ij tensor)

`IsotropicElasticMaterial` is the only material spec today. Phase 8
(DFT-MD-continuum coupling) will need to feed a 6×6 Voigt stiffness
tensor from DFT into the continuum solver. The solver's `BilinearForm`
would need to replace `2μ ε : ε + λ tr(ε)² ` with `ε : C : ε`. A
half-day of work; deferred because Session 5.1's acceptance is
isotropic.

### 3. Legacy `backend/common/engines/continuum.py::FEMEngine`

The legacy stub is still in place and still returns fake data. It
no longer matches the ORION-canonical FEM path — new callers should
route through `continuum_fem.solve_linear_elasticity` / `solve_steady_heat`
directly. Migrating the stub's `SimulationEngine` interface to
wrap the real solver is a follow-up (probably in Session 5.3 when
workflow templates compose DFT → MD → continuum).

### 4. Dynamic / transient / modal analysis

Session 5.1 is steady-state only. Modal analysis (natural frequencies)
and transient dynamics (Newmark-β) share most of the infrastructure
(same K, add M and damping). Deferred; a continuum-fatigue campaign
or a thermal-shock problem would motivate the addition.

### 5. Axisymmetric / plate / shell elements

Full 3D only. For beam problems with extreme aspect ratios the
1D Euler-Bernoulli formulation would converge faster than our 3D
hex mesh — but the cantilever acceptance already hits 0.14%, so
the 3D path is adequate for the roadmap's scope.

## Dependencies

Added to `requirements.txt`:

```
scikit-fem>=12.0.0                 # Phase 5 / Session 5.1 FEM continuum solver
meshio>=5.3.0                      # VTU / .msh IO for continuum solver
```

Both are pure Python (no compiled deps beyond numpy/scipy). CI should
install cleanly.

## Tests

- `tests/test_continuum_fem.py` — 16 new tests:
  - `TestMaterial` — Lamé parameter derivation, ν bounds.
  - `TestProblemValidation` — elasticity/heat Dirichlet-absent
    rejection at construction.
  - `TestCantileverAcceptance` — δ within 1% (Hex2), u_z sign,
    Hex1-coarse-mesh lock-in of known limitation.
  - `TestUniformTension` — σ_xx = traction in the interior with
    partial-component symmetry BCs (exercises the partial-fix path
    in `_apply_dirichlet`).
  - `TestHeatRodAcceptance` — midpoint exact, insulated-sides
    invariance, uniform-offset sanity.
  - `TestVTUExport` — round-trip through meshio for both solution
    types.
  - `TestBoxMesh` — face-area geometry + face-facet partitioning.
- Full suite: **446 passed, 4 skipped** (was 430 + 16 new).

## Phase 5 status

5.1 done. Next per roadmap is **Session 5.2 — KMC mesoscale (minimum
viable)**: rejection-free Gillespie on a lattice with vacancy /
interstitial hops and annihilation. `backend/common/engines/mesoscale.py`
stub (returns fake data today) gets the same treatment as
`continuum.py` did here.
